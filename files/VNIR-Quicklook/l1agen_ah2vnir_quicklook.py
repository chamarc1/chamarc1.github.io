"""
VNIR Quicklook Generation - Entry Point

__author__ = cmarc

This module serves as the entry-point wrapper for the AirHARP2 VNIR L1A quicklook workflow.
It provides a command-line interface for generating metadata quicklook visualizations from
NetCDF4 L1A files.

WORKFLOW OVERVIEW:
    1. Parse command-line arguments (L1A path, output path, time windows, etc.)
    2. Search for VNIR L1A NetCDF4 files in the specified directory
    3. Extract metadata and create a stacked metadata structure
    4. Generate quicklook PNG visualizations for specified time windows
    5. Save outputs to the specified directory

ARCHITECTURE:
    The implementation is split into composable, single-responsibility modules:
    
    - ql_domain.py
        • Core data structures (TimeWindow, QuickplotConfig)
        • Domain-specific types and value objects
    
    - ql_transforms.py
        • Data transformation utilities (datetime parsing, array operations)
        • Pure functions for data processing pipelines
    
    - ql_io_netcdf.py
        • NetCDF4 file I/O operations
        • File search, metadata extraction, time coverage parsing
    
    - ql_plotting.py
        • Visualization generation using matplotlib
        • Quicklook image creation and file output

COMMAND-LINE USAGE:
    python l1agen_ah2vnir_quicklook.py \\
        --l1a_path <directory> \\
        --save_path <output_directory> \\
        [--preprocessing_file <granule_file>] \\
        [--log_file <timings_file>] \\
        [--start_time <ISO_datetime>] \\
        [--end_time <ISO_datetime>] \\
        [--line_index <row_index>] \\
        [--sensor_temp] \\
        [--show] \\
        [--verbose]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import logging
import os
import re
import shlex
import sys
from collections import defaultdict
from pathlib import Path

from netCDF4 import Dataset
import matplotlib

matplotlib.use("Agg")  # non-interactive backend; required for safe rendering across worker processes

from ql_modules.ql_domain import TimeWindow
from ql_modules.ql_io_netcdf import (
    create_vnir_metadata_stack,
    default_pushbroom_line_indices,
    get_time_coverage,
    search_vnir_l1a,
)
from ql_modules.ql_plotting import quickplot_vnir_metadata
from ql_modules.ql_transforms import parse_iso_datetime

logger = logging.getLogger(__name__)


class _RedirectAwareStreamHandler(logging.StreamHandler):
    """StreamHandler that resolves sys.stdout at emit time.

    A plain StreamHandler binds its stream once at construction, so it would
    keep writing to the original console stdout even after `--log_output_path`
    redirects stdout to a per-process file. Resolving dynamically mirrors how
    print() behaves under contextlib.redirect_stdout.
    """

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, _value):
        pass


def configure_logging(verbose: bool) -> None:
    """Configure logging once: DEBUG progress output when verbose, else INFO+.

    Only this app's own loggers (this module and `ql_modules`) are raised to
    DEBUG; the root logger stays at INFO so third-party libraries (e.g.
    matplotlib's noisy `findfont`/`locator` DEBUG chatter) stay quiet.
    """

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(handler, _RedirectAwareStreamHandler) for handler in root.handlers):
        handler = _RedirectAwareStreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"))
        root.addHandler(handler)

    app_level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger(__name__).setLevel(app_level)
    logging.getLogger("ql_modules").setLevel(app_level)


def leg_output_name(file_path: Path) -> str:
    """Convert a granule filename ending in ``_L##G###.nc`` to ``_L##.png``."""

    match = re.search(r"_L(?P<leg>\d+)G\d+\.nc$", file_path.name, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Cannot derive leg output name from filename: {file_path.name}")
    return f"{file_path.name[:match.start()]}_L{match.group('leg')}.png"


def default_output_name(file_path: Path, leg_id: int | None = None) -> str:
    """Build the standard quicklook name from the first granule and leg."""

    match = re.search(
        r"^INSPYRE-AH2VNIR-L1A_ER2_(?P<date>\d{8})T\d{6}_(?P<revision>[^.]+)\.nc$",
        file_path.name,
        re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"Cannot derive standard output name from filename: {file_path.name}")
    leg_description = f"Leg{leg_id:02d}" if leg_id is not None else "Leg00"
    return (
        f"INSPYRE-AH2VNIR-L1A_ER2_{match.group('date')}_"
        f"{match.group('revision')}_{leg_description}-quicklook.png"
    )


class custom_argparser(argparse.ArgumentParser):
    """Argument parser supporting ``@file`` inputs with ``--option = value`` lines."""

    def convert_arg_line_to_args(self, arg_line: str) -> list[str]:
        tokens = shlex.split(arg_line, comments=True)
        if not tokens:
            return []
        if len(tokens) == 1:
            return tokens
        if len(tokens) == 3 and tokens[1] == "=":
            return [tokens[0], tokens[2]]
        return tokens


def parse_time_windows_from_log(log_file: Path) -> list[TimeWindow]:
    """Parse comma-separated start/end datetime rows into time windows.

    The parser skips blank lines, comment lines starting with `#`, malformed
    rows, and rows with invalid datetime values. Only valid windows are
    returned.
    """

    # Read user-provided window definitions from a simple CSV-like log file.
    logger.debug("Parsing time windows from log file: %s", log_file)
    windows: list[TimeWindow] = []
    with open(log_file, "r", encoding="utf-8") as fobj:
        for line in fobj:
            line = line.strip()
            # Skip whitespace and commented lines so notes can live in the same file.
            if (not line) or line.startswith("#"):
                continue

            parts = [item.strip() for item in line.split(",")]
            if len(parts) < 2:
                continue

            try:
                # Each valid row contributes exactly one [start, end] processing window.
                windows.append(TimeWindow(start=parse_iso_datetime(parts[0]), end=parse_iso_datetime(parts[1])))
            except ValueError:
                # Ignore malformed datetime rows and keep parsing the rest of the file.
                continue

    logger.debug("Parsed valid time windows: %d", len(windows))
    return windows


def parse_leg_windows_from_preprocessing_file(
    preprocessing_file: Path,
) -> list[tuple[int, list[TimeWindow]]]:
    """Parse preprocessing rows into ordered leg plans.

    Expected row format is `leg_id, granule_id, start_time, end_time, ...`.
    Rows are grouped by leg, sorted by granule id within each leg, and then
    returned as `(leg_id, [TimeWindow, ...])` in ascending leg order.
    """

    # Build a leg->granules mapping so each flight leg can be processed as one output product.
    logger.debug("Parsing leg definitions from preprocessing file: %s", preprocessing_file)
    leg_map: dict[int, list[tuple[int, TimeWindow]]] = defaultdict(list)

    with open(preprocessing_file, "r", encoding="utf-8") as fobj:
        for line in fobj:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue

            parts = [item.strip() for item in line.split(",")]
            if len(parts) < 4:
                continue

            try:
                leg_id = int(parts[0])
                granule_id = int(parts[1])
                start_time = parse_iso_datetime(parts[2])
                end_time = parse_iso_datetime(parts[3])
            except ValueError:
                # Any non-numeric leg/granule id or invalid datetime is treated as a skipped row.
                continue

            leg_map[leg_id].append((granule_id, TimeWindow(start=start_time, end=end_time)))

    ordered_legs: list[tuple[int, list[TimeWindow]]] = []
    # Preserve scientific acquisition order: legs ascending, then granules ascending.
    for leg_id in sorted(leg_map):
        granules = [window for _granule_id, window in sorted(leg_map[leg_id], key=lambda item: item[0])]
        if granules:
            ordered_legs.append((leg_id, granules))

    logger.debug("Parsed legs: %d", len(ordered_legs))
    return ordered_legs


def leg_log_output_path(log_output_path: Path, leg_id: int) -> Path:
    """Append a zero-padded leg suffix to a log output stem."""

    return log_output_path.parent / f"{log_output_path.stem}_L{leg_id:02d}"


def _process_leg(
    leg_id: int,
    granule_windows: list[TimeWindow],
    l1a_path: Path,
    save_path: Path,
    line_indices,
    metadata_panels: list[str] | None,
    leg_output_naming: bool,
    show: bool,
    verbose: bool,
    log_output_path: Path | None,
) -> Path | None:
    """Search, stack, and plot a single leg; module-level so it can run in a worker process."""

    with ExitStack() as output_stack:
        if log_output_path is not None:
            # Each worker process has a distinct PID, so parallel legs land in separate log files.
            process_log_path = log_output_path.parent / str(os.getpid())
            process_log_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_file = output_stack.enter_context(process_log_path.with_suffix(".out").open("a", encoding="utf-8"))
            stderr_file = output_stack.enter_context(process_log_path.with_suffix(".err").open("a", encoding="utf-8"))
            output_stack.enter_context(redirect_stdout(stdout_file))
            output_stack.enter_context(redirect_stderr(stderr_file))

        configure_logging(verbose)
        sep = "=" * 80
        logger.debug(sep)
        logger.debug("Processing leg %d with %d granules", leg_id, len(granule_windows))
        leg_files: list[Path] = []

        # Collect candidate files by scanning each granule time window for this leg.
        for granule_window in granule_windows:
            files = search_vnir_l1a(l1a_path, granule_window.start, granule_window.end)
            if not files:
                logger.debug("  no files found for granule window %s to %s", granule_window.start, granule_window.end)
                continue
            leg_files.extend(files)

        if not leg_files:
            logger.debug("Skipping leg %d: no matching files found", leg_id)
            return None

        # Re-sort matched files by metadata coverage start to ensure deterministic stack order.
        file_coverage: list[tuple[Path, object]] = []
        for file_path in leg_files:
            with Dataset(file_path, "r") as ds:
                coverage_start, _coverage_end = get_time_coverage(ds)
            if coverage_start is None:
                continue
            file_coverage.append((file_path, coverage_start))

        leg_files = [file_path for file_path, _coverage_start in sorted(file_coverage, key=lambda item: item[1])]

        # Build one metadata stack per leg and render a single combined quicklook image.
        leg_window = TimeWindow(start=granule_windows[0].start, end=granule_windows[-1].end)
        stack = create_vnir_metadata_stack(leg_files, line_indices=line_indices)
        if stack.seconds_of_day.size == 0:
            logger.debug("Skipping leg %d: selected files contained zero frames", leg_id)
            logger.debug(sep)
            return None
        out_path = quickplot_vnir_metadata(
            stack,
            leg_window,
            save_path,
            output_label=f"Leg{leg_id}",
            output_name=(
                leg_output_name(leg_files[0])
                if leg_output_naming
                else default_output_name(leg_files[0], leg_id)
            ),
            show=show,
            metadata_panels=metadata_panels,
        )
        logger.info("Saved quicklook for leg %d: %s", leg_id, out_path)
        logger.debug(sep)
        return out_path


def run_quicklooks_vnir(
    l1a_path: Path,
    save_path: Path,
    log_file: Path | None = None,
    preprocessing_file: Path | None = None,
    leg_id: int | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    line_index: int | None = None,
    sensor_temp: bool = True,
    leg_output_naming: bool = False,
    show: bool = False,
    verbose: bool = False,
    log_output_path: Path | None = None,
    max_workers: int | None = None,
) -> list[Path]:
    """Run the VNIR quicklook workflow and return generated PNG paths.

    Supports three processing modes:
    1. `preprocessing_file` mode for leg-wise processing.
    2. `log_file` or explicit `start_time`/`end_time` window mode.
    3. Full-coverage mode inferred from first/last file metadata.

    For each selected leg/window, matching L1A files are searched, stacked,
    and plotted as metadata quicklook images.

    In `preprocessing_file` mode, legs are independent of one another and are
    processed in parallel worker processes (via `max_workers`) unless `show`
    is set, since interactive figure windows require a single process.
    """

    configure_logging(verbose)
    logger.debug("Starting VNIR metadata quicklook run...")
    logger.debug("L1A path: %s", l1a_path)
    logger.debug("Save path: %s", save_path)

    # Quicklook pushbrooms are fixed to the VNIR wavelength rows defined in ql_io_netcdf.
    line_indices = default_pushbroom_line_indices()
    metadata_panels = ["sensor_temperature"] if sensor_temp else None
    if line_index is not None:
        logger.debug(
            "line_index is deprecated for VNIR quicklook pushbrooms; using the four fixed wavelength rows instead."
        )

    # Guardrails: preprocessing mode is mutually exclusive with legacy window selection arguments.
    if preprocessing_file is not None and (log_file is not None or start_time is not None or end_time is not None):
        raise ValueError("Use preprocessing_file by itself, or use the legacy log_file/start_time/end_time modes.")

    # MODE 1: Leg-wise processing driven by preprocessing rows.
    if preprocessing_file is not None:
        leg_plans = parse_leg_windows_from_preprocessing_file(preprocessing_file)
        if not leg_plans:
            raise ValueError(f"No valid leg rows parsed from {preprocessing_file}")
        if leg_id is not None:
            leg_plans = [plan for plan in leg_plans if plan[0] == leg_id]
            if not leg_plans:
                raise ValueError(f"Leg {leg_id} was not found in {preprocessing_file}")
        logger.debug("Total legs to process: %d", len(leg_plans))

        # Sequential by default (preserves single-process behavior); opt into parallel legs via max_workers.
        worker_count = 1 if show else max(1, min(max_workers or 1, len(leg_plans)))
        if show and max_workers not in (None, 1):
            logger.debug("Ignoring max_workers=%s because --show requires sequential processing", max_workers)
        logger.debug("Processing %d leg(s) with %d worker process(es)", len(leg_plans), worker_count)

        if worker_count > 1:
            results: dict[int, Path | None] = {}
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(
                        _process_leg,
                        plan_leg_id,
                        granule_windows,
                        l1a_path,
                        save_path,
                        line_indices,
                        metadata_panels,
                        leg_output_naming,
                        show,
                        verbose,
                        log_output_path,
                    ): plan_leg_id
                    for plan_leg_id, granule_windows in leg_plans
                }
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
            # Preserve ascending leg order regardless of completion order.
            outputs = [
                results[plan_leg_id] for plan_leg_id, _granule_windows in leg_plans if results[plan_leg_id] is not None
            ]
        else:
            outputs = []
            for plan_leg_id, granule_windows in leg_plans:
                out_path = _process_leg(
                    plan_leg_id,
                    granule_windows,
                    l1a_path,
                    save_path,
                    line_indices,
                    metadata_panels,
                    leg_output_naming,
                    show,
                    verbose,
                    log_output_path,
                )
                if out_path is not None:
                    outputs.append(out_path)

        logger.debug("Quicklook run complete. Files generated: %d", len(outputs))
        return outputs

    # MODE 2/3: Window-based processing from log rows, explicit CLI bounds, or inferred full coverage.
    if log_file is not None:
        windows = parse_time_windows_from_log(log_file)
        if not windows:
            raise ValueError(f"No valid time windows parsed from {log_file}")
    elif start_time is not None and end_time is not None:
        windows = [TimeWindow(start=parse_iso_datetime(start_time), end=parse_iso_datetime(end_time))]
    else:
        # Fallback mode: infer a single full-coverage window from first and last file metadata.
        all_files = search_vnir_l1a(l1a_path, None, None)
        if not all_files:
            raise ValueError(f"No VNIR L1A files found in {l1a_path}")

        with Dataset(all_files[0], "r") as ds0, Dataset(all_files[-1], "r") as ds1:
            win_start = get_time_coverage(ds0)[0]
            win_end = get_time_coverage(ds1)[1]

        if win_start is None or win_end is None:
            raise ValueError("Could not infer global time coverage from file metadata.")
        windows = [TimeWindow(start=win_start, end=win_end)]

    logger.debug("Total time windows to process: %d", len(windows))

    outputs: list[Path] = []
    for window in windows:
        logger.debug("Processing window: %s to %s", window.start, window.end)
        # Find all files overlapping the requested window and assemble a stack for plotting.
        files = search_vnir_l1a(l1a_path, window.start, window.end)
        if not files:
            logger.info("No files found for %s to %s", window.start, window.end)
            continue

        stack = create_vnir_metadata_stack(files, line_indices=line_indices)
        if stack.seconds_of_day.size == 0:
            logger.debug("Skipping window %s to %s: selected files contained zero frames", window.start, window.end)
            continue
        out_path = quickplot_vnir_metadata(
            stack,
            window,
            save_path,
            output_name=(
                leg_output_name(files[0])
                if leg_output_naming
                else default_output_name(files[0])
            ),
            show=show,
            metadata_panels=metadata_panels,
        )
        logger.info("Saved quicklook: %s", out_path)
        outputs.append(out_path)

    logger.debug("Quicklook run complete. Files generated: %d", len(outputs))
    return outputs


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for VNIR quicklook generation options.

    Defines required input/output paths plus optional filters and behavior
    flags (time bounds, preprocessing file, plotting options, verbosity).
    """

    # Keep all CLI options centrally defined so behavior stays discoverable in --help.
    parser = custom_argparser(
        fromfile_prefix_chars="@",
        description="Generate AirHARP2 VNIR L1A metadata quicklooks from NetCDF4 files."
    )
    parser.add_argument(
        "--l1a_path",
        required=True,
        type=Path,
        help="Directory containing AH2VNIR L1A NetCDF4 files.",
    )
    parser.add_argument(
        "--save_path",
        required=True,
        type=Path,
        help="Output directory for PNG quicklook files.",
    )
    parser.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help="Optional file containing comma-separated start_time,end_time per line.",
    )
    parser.add_argument(
        "--preprocessing_file",
        type=Path,
        default=None,
        help="Optional preprocessing granule file with Leg#, Granule#, start/end rows.",
    )
    parser.add_argument(
        "--start_time",
        type=str,
        default=None,
        help="Optional ISO datetime start bound (used if --log_file is omitted).",
    )
    parser.add_argument(
        "--end_time",
        type=str,
        default=None,
        help="Optional ISO datetime end bound (used if --log_file is omitted).",
    )
    parser.add_argument(
        "--line_index",
        type=int,
        default=None,
        help="Deprecated; multi-wavelength pushbrooms use the four fixed wavelength rows.",
    )
    parser.add_argument(
        "--sensor_temp",
        action="store_true",
        help="Include sensor temperature metadata panel in the quicklook.",
    )
    parser.add_argument(
        "--leg_output_naming",
        action="store_true",
        help="Use legacy naming after the first granule's leg, replacing _L##G###.nc with _L##.png.",
    )
    parser.add_argument(
        "--log_output_path",
        type=Path,
        default=None,
        help="Directory for captured logs; writes stdout to <process ID>.out and stderr to <process ID>.err.",
    )
    parser.add_argument(
        "--max_workers",
        "-j",
        type=int,
        default=None,
        help=(
            "Number of worker processes for parallel leg processing in --preprocessing_file mode "
            "(default: 1/sequential; ignored with --show)."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figure window in addition to saving PNG.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print step-by-step progress to console.",
    )
    return parser


def main() -> None:
    """Command-line entry point.

    Parses user-provided arguments and forwards them to the workflow runner.
    Kept intentionally thin so orchestration logic remains testable in
    `run_quicklooks_vnir`.
    """
    # Thin CLI wrapper: parse args once and hand off to the testable workflow function.
    parser = build_parser()
    args = parser.parse_args()

    run_quicklooks_vnir(
        l1a_path=args.l1a_path,
        save_path=args.save_path,
        log_file=args.log_file,
        preprocessing_file=args.preprocessing_file,
        start_time=args.start_time,
        end_time=args.end_time,
        line_index=args.line_index,
        sensor_temp=args.sensor_temp,
        leg_output_naming=args.leg_output_naming,
        show=args.show,
        verbose=args.verbose,
        log_output_path=args.log_output_path,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
