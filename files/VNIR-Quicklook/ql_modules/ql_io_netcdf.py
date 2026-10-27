from __future__ import annotations

import datetime as dt
import logging
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
from netCDF4 import Dataset

from .ql_domain import MetadataChunk, MetadataStack, TimeWindow
from .ql_transforms import as_1d_float, concat_chunks, parse_iso_datetime, window_contains

logger = logging.getLogger(__name__)


PUSHBROOM_WAVELENGTH_ROWS: tuple[tuple[float, int], ...] = (
    (763.5, 1478),
    (589.2, 1194),
    (567.7, 1181),
    (546.1, 1141),
)


def default_pushbroom_line_indices() -> list[int]:
    """Return the zero-based line indices used for the four wavelength pushbrooms."""

    return [line_number - 700 for _wavelength, line_number in PUSHBROOM_WAVELENGTH_ROWS]


def pushbroom_wavelength_labels() -> list[float]:
    """Return the wavelength labels associated with the default pushbrooms."""

    return [wavelength for wavelength, _line_number in PUSHBROOM_WAVELENGTH_ROWS]


def pushbroom_line_number_labels() -> list[int]:
    """Return ROI-equivalent line labels for the default pushbrooms.

    The VNIR ROI starts at full-sensor line 700, so labels are expressed as
    (full_sensor_line - 700) to match ROI indexing used in quicklooks.
    """

    return [line_number - 700 for _wavelength, line_number in PUSHBROOM_WAVELENGTH_ROWS]


def list_l1a_candidates(l1a_dir: Path) -> list[Path]:
    """Return sorted candidate VNIR L1A files from directory.
    
    Searches for files matching the pattern "*AH2VNIR-L1A_*.nc" (AirHARP2 VNIR L1A naming).
    Returns results in sorted order to ensure consistent, predictable ordering for
    downstream time-ordered processing.
    
    Args:
        l1a_dir: Directory path to search
    
    Returns:
        Sorted list of Path objects for matching NetCDF4 files
    """

    return _cached_l1a_candidates(l1a_dir)


@lru_cache(maxsize=None)
def _cached_l1a_candidates(l1a_dir: Path) -> list[Path]:
    # Cached per directory so repeated per-granule searches within a run
    # don't re-scan the filesystem or re-log the same candidate count.
    return sorted(l1a_dir.glob("*AH2VNIR-L1A_*.nc"))


def get_time_coverage(ds: Dataset) -> tuple[dt.datetime | None, dt.datetime | None]:
    """Read global time_coverage_start/end if present and valid.
    
    Extracts the time_coverage_start and time_coverage_end global attributes
    from a NetCDF4 file, parsing them as ISO datetime objects. Returns (None, None)
    if attributes are missing or invalid.
    
    Args:
        ds: Open NetCDF4 Dataset object
    
    Returns:
        Tuple of (start_datetime, end_datetime) or (None, None) if not available
    """

    start_str = getattr(ds, "time_coverage_start", "")
    end_str = getattr(ds, "time_coverage_end", "")

    start = parse_iso_datetime(start_str) if start_str else None
    end = parse_iso_datetime(end_str) if end_str else None
    return start, end


def get_frame_count(ds: Dataset) -> int | None:
    """Return the frame count from image_data when available.

    Some VNIR files can exist with valid time coverage metadata but zero image
    frames. Returning ``None`` indicates frame count could not be inferred and
    callers may decide whether to keep or skip those files.
    """

    image_data = ds.groups.get("image_data")
    if image_data is None:
        return None

    if "frames" in image_data.dimensions:
        return len(image_data.dimensions["frames"])

    if "seconds_of_day" in image_data.variables:
        try:
            return int(image_data["seconds_of_day"].shape[0])
        except (TypeError, ValueError, IndexError):
            return None

    return None


def search_vnir_l1a(
    l1a_dir: Path,
    start_time: dt.datetime | None = None,
    end_time: dt.datetime | None = None,
) -> list[Path]:
    """Find VNIR L1A files whose coverage is fully inside [start_time, end_time].
    
    Searches for L1A candidates in the directory, opens each to read its time_coverage
    global attributes, and filters for files whose coverage is completely contained
    within the requested time window.
    
    If start_time and end_time are both None, all candidate files are returned
    (no filtering applied).
    
    Args:
        l1a_dir: Directory containing VNIR L1A files
        start_time: Window start (inclusive); if None, no start filtering
        end_time: Window end (inclusive); if None, no end filtering
    
    Returns:
        List of Path objects for files fully contained in the time window
    """

    files = list_l1a_candidates(l1a_dir)
    selected: list[Path] = []

    window = None
    if (start_time is not None) and (end_time is not None):
        window = TimeWindow(start=start_time, end=end_time)

    for file_path in files:
        with Dataset(file_path, "r") as ds:
            fstart, fend = get_time_coverage(ds)
            frame_count = get_frame_count(ds)

        if fstart is None or fend is None:
            continue

        include = True if window is None else window_contains(fstart, fend, window)
        if include:
            if frame_count == 0:
                logger.debug("  skipping empty file: %s (%s to %s)", file_path.name, fstart, fend)
                continue
            selected.append(file_path)
            frame_text = ""
            if frame_count is not None:
                frame_text = f", frames={frame_count}"
            logger.debug("  included: %s (%s to %s%s)", file_path.name, fstart, fend, frame_text)

    logger.debug(
        "Searched %s (%d candidates) for window %s..%s: %d selected",
        l1a_dir, len(files), start_time, end_time, len(selected),
    )
    return selected


def read_metadata_from_file(
    file_path: Path,
    line_indices: Sequence[int] | None = None,
) -> MetadataChunk:
    """Read key metadata vectors and a sensor strip from one VNIR L1A file.
    
    Extracts all scalar metadata (temperatures, integration time, frame IDs, etc.) and
    a stack of 2D pushbroom image strips from wavelength-specific sensor rows.
    
    Args:
        file_path: Path to the VNIR L1A NetCDF4 file
        line_indices: Sensor row indices (0-based) to extract for visualization.
                 If None, defaults to the four wavelength rows defined by
                 PUSHBROOM_WAVELENGTH_ROWS after subtracting the zero-point offset.
                 Each index is clipped to the valid range [0, n_lines-1].
    
    Returns:
        MetadataChunk with all extracted metadata and sensor strips.
        Masked/invalid pixels are converted to NaN for downstream handling.
    """

    logger.debug("Reading metadata: %s", file_path)
    with Dataset(file_path, "r") as ds:
        img = ds.groups["image_data"]

        sensor1_var = img["sensor1"]
        n_lines = sensor1_var.shape[1]
        if line_indices is None:
            line_indices = default_pushbroom_line_indices()

        # Index each line directly on the netCDF variable so only the needed
        # lines are read from disk, instead of loading the full sensor cube.
        sensor_strips = []
        for line_index in line_indices:
            idx_line = int(np.clip(line_index, 0, n_lines - 1))
            sensor_strip = np.ma.asarray(sensor1_var[:, idx_line, :]).astype(float)
            sensor_strips.append(np.ma.filled(sensor_strip, np.nan))

        sensor_strips_array = np.stack(sensor_strips, axis=0)

        return MetadataChunk(
            seconds_of_day=as_1d_float(img["seconds_of_day"][:, 0]),
            frame_id=as_1d_float(img["frame_ID"][:, 0]),
            sensor_temperature=as_1d_float(img["sensor_temperature"][:, 0]),
            camera_temperature=as_1d_float(img["camera_temperature"][:, 0]),
            power_supply_temperature=as_1d_float(img["power_supply_temperature"][:, 0]),
            integration_time=as_1d_float(img["integration_time"][:, 0]),
            image_type=as_1d_float(img["image_type"][:, 0]),
            sensor_strips=sensor_strips_array,
        )


def create_vnir_metadata_stack(
    file_list: Sequence[Path],
    line_indices: Sequence[int] | None = None,
) -> MetadataStack:
    """Concatenate metadata vectors across all files in temporal order.
    
    Reads metadata from each file in sequence and combines them into a single
    MetadataStack. All 1D arrays are concatenated along the frame axis, and
    cumulative frame boundaries are recorded for visualization purposes.
    
    Args:
        file_list: Sequence of Paths to VNIR L1A files (should be in temporal order)
        line_indices: Sensor rows to extract from all files (same rows used for all files)
    
    Returns:
        MetadataStack containing concatenated data from all files
    
    Raises:
        ValueError: If file_list is empty
    """

    if not file_list:
        raise ValueError("No files provided to create_vnir_metadata_stack.")

    logger.debug("Stacking metadata from %d files...", len(file_list))
    chunks = [read_metadata_from_file(fpath, line_indices=line_indices) for fpath in file_list]
    stack = concat_chunks(chunks)
    logger.debug("Stack complete. Total frames: %d", stack.seconds_of_day.shape[0])
    return stack
