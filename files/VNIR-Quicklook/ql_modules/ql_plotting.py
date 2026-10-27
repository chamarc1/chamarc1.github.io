from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Sequence

import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from .ql_domain import MetadataStack, TimeWindow
from .ql_io_netcdf import pushbroom_line_number_labels, pushbroom_wavelength_labels
from .ql_transforms import format_dt_for_fname, safe_vlims, seconds_to_labels

logger = logging.getLogger(__name__)

FIGURE_WIDTH_IN = 16.0
MIN_PUSHBROOM_HEIGHT_IN = 7.0
PUSHBROOM_HEIGHT_PER_ASPECT = 5.5
METADATA_PANEL_HEIGHT_IN = 2.0
LAYOUT_PADDING_HEIGHT_IN = 2.8
MIN_FIGURE_HEIGHT_IN = 10.0
MAX_FIGURE_HEIGHT_IN = 42.0
COLORBAR_HEIGHT_RATIO = 0.35


def quickplot_vnir_metadata(
    metadata: MetadataStack,
    window: TimeWindow,
    save_path: Path,
    output_label: str | None = None,
    output_name: str | None = None,
    show: bool = False,
    metadata_panels: Sequence[str] | None = None,
) -> Path:
    """Generate and save a VNIR quicklook figure.
    
    Creates a multi-panel figure with:

    TOP ROW (Large):
        Side-by-side pushbroom image strips from configured sensor rows.
        - X-axis: Spectral pixel index
        - Y-axis: Frame progression (flipped to match MAP quicklook direction)
        - Color scale: DN, normalized to mean ± 3σ for visual clarity

    BOTTOM PANELS (Optional, Stacked):
        Time series plots of instrument housekeeping.
        Panels are only shown when explicitly enabled via metadata_panels.

    All time series share the same x-axis (UTC date/time) aligned with the pushbroom frame index.
    
    Args:
        metadata: MetadataStack with all extracted data
        window: TimeWindow defining the time span shown in figure title
        save_path: Output directory where PNG will be saved
        output_label: Optional filename label (e.g., "Leg1") used in the title.
        output_name: Explicit output PNG filename. If provided, takes precedence
            over generated naming conventions.
        show: If True, display figure in window (blocking). If False, save only.
        metadata_panels: Optional sequence of metadata field names to show below
            pushbroom panels. If omitted or empty, no metadata panel is shown.
    
    Returns:
        Path to the saved PNG file
    """

    logger.debug("Building quicklook figure...")
    logger.debug("Time range: %s to %s", window.start, window.end)
    logger.debug("Output directory: %s", save_path)

    # Pulling raw data arrays and configuration labels needed for the plot
    # Each sensor strip is indexed by acquisition frame, which supplies the time axis.
    seconds = metadata.seconds_of_day
    metadata_times = [
        dt.datetime.combine(window.start.date(), dt.time()) + dt.timedelta(seconds=float(value))
        for value in seconds
    ]
    sensor_strips = metadata.sensor_strips
    wavelength_labels = pushbroom_wavelength_labels()
    line_number_labels = pushbroom_line_number_labels()
    vlims = [safe_vlims(strip) for strip in sensor_strips]
    ytick, ylabel = seconds_to_labels(seconds, metadata.segment_ticks, window.start.date())
    
    # Measuring the dimensions of our image data to handle variable flight-pass lengths
    strip_shapes = [strip.shape for strip in sensor_strips]
    h = max(shape[0] for shape in strip_shapes) # Max number of frames (vertical height of flight pass)
    w = max(shape[1] for shape in strip_shapes) # Max cross-track pixels (horizontal width)
    strip_ratio = h / max(w, 1) # Computes how tall the image is compared to its width

    # List of available instrument telemetry metrics that can be plotted as bottom panels
    panels = [
        ("sensor_temperature", "Sensor Temp (C)"),
        ("camera_temperature", "Camera Temp (C)"),
        ("power_supply_temperature", "PS Temp (C)"),
        ("integration_time", "Integration (ms)"),
        ("image_type", "Image Type"),
    ]

    # Map requested metadata field names to their display labels, ignoring unrequested ones
    panel_lookup = {field: (field, ylabel) for field, ylabel in panels}
    if metadata_panels is None:
        active_panels: list[tuple[str, str]] = []
    else:
        active_panels = [panel_lookup[field] for field in metadata_panels if field in panel_lookup]

    metadata_rows = len(active_panels)
    n_pushbrooms = len(sensor_strips)
    
    # --- DYNAMIC CANVAS SIZING CALCULATIONS ---
    # Width is locked to a fixed 16 inches since we always have 4 pushbrooms side-by-side
    fig_w = FIGURE_WIDTH_IN
    
    # Height scales automatically: longer flight passes (taller aspect ratios) generate larger pushbroom heights
    pushbroom_height = max(MIN_PUSHBROOM_HEIGHT_IN, strip_ratio * PUSHBROOM_HEIGHT_PER_ASPECT)
    
    # Add 2.0 inches of vertical space for each active metadata graph stacked at the bottom
    metadata_height = METADATA_PANEL_HEIGHT_IN * metadata_rows
    
    # Total canvas height combines image height, telemetry height, and UI padding. 
    # Bounded between 10.0 and 42.0 inches to prevent tiny compressed plots or system-crashing giant images.
    fig_h = min(
        MAX_FIGURE_HEIGHT_IN,
        max(MIN_FIGURE_HEIGHT_IN, pushbroom_height + metadata_height + LAYOUT_PADDING_HEIGHT_IN),
    )

    # Font sizing configuration for the figure layout
    suptitle_fs = 22
    panel_title_fs = 14
    axis_label_fs = 13
    tick_label_fs = 11

    # Initialize the main Matplotlib figure canvas
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150, constrained_layout=True)
    leg_number = output_label[3:] if output_label and output_label.startswith("Leg") else output_label
    leg_title = f"Leg #{leg_number or 'N/A'}"
    fig.suptitle(
        (
            "AirHARP2 VNIR L1A Metadata Quicklook "
            f"{leg_title}\n"
            f"Start: {window.start.isoformat(sep=' ')} "
            f"End: {window.end.isoformat(sep=' ')}"
        ),
        fontsize=suptitle_fs,
    )
    
    # Create a grid layout: columns match the number of pushbrooms; rows include
    # the images, one colorbar row, and any active metadata graphs.
    gs = gridspec.GridSpec(
        2 + metadata_rows,
        n_pushbrooms,
        figure=fig,
        height_ratios=[pushbroom_height, COLORBAR_HEIGHT_RATIO] + [1.0] * metadata_rows,
        hspace=0.02,
        wspace=0.12,
    )

    # Loop through each pushbroom sensor and plot its image strip side-by-side
    pushbroom_share_ax = None
    for idx, (strip, (vmin, vmax), wavelength, line_number) in enumerate(
        zip(sensor_strips, vlims, wavelength_labels, line_number_labels, strict=True)
    ):
        ax = fig.add_subplot(gs[0, idx], sharey=pushbroom_share_ax)
        if pushbroom_share_ax is None:
            pushbroom_share_ax = ax

        # Plot detector pixel index across the x-axis and acquisition time/frame vertically.
        # This is temporal alignment, not geographic or aircraft-location alignment.
        # imshow (with origin="lower" to match pcolormesh's row orientation) is much
        # faster than pcolormesh for large regular grids like this.
        image = ax.imshow(
            np.flipud(strip), vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest", origin="lower"
        )
        ax.set_title(f"Line {line_number} ({wavelength:.1f} nm)", fontsize=panel_title_fs)

        # Put each horizontal colorbar in the matching GridSpec column so its
        # width is exactly the same as the pushbroom x-axis.
        colorbar_ax = fig.add_subplot(gs[1, idx])
        colorbar = fig.colorbar(image, cax=colorbar_ax, orientation="horizontal")
        colorbar.set_label("DN", fontsize=tick_label_fs)
        colorbar.ax.tick_params(labelsize=tick_label_fs)

        # Only display UTC time labels on the leftmost panel to avoid visual clutter
        if idx == 0:
            # All pushbroom panels use the same frame-time labels because they share frames.
            ax.set_yticks(ytick, ylabel)
            ax.set_ylabel("UTC (YYYY-MM-DD HH:MM:SS)", fontsize=axis_label_fs)
            ax.tick_params(axis="y", labelsize=tick_label_fs)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)

        if idx != (len(sensor_strips) - 1):
            ax.tick_params(axis="x", labelbottom=True, labelsize=tick_label_fs)
        else:
            ax.tick_params(axis="x", labelsize=tick_label_fs)
        ax.set_xlabel("Cross-track pixel index", fontsize=axis_label_fs)

    # If metadata panels are requested, stack them neatly underneath the pushbroom row
    if metadata_rows > 0:
        share_ax = None
        for idx, (field, ylabel_txt) in enumerate(active_panels):
            ax = fig.add_subplot(gs[idx + 2, :], sharex=share_ax)
            if share_ax is None:
                share_ax = ax

            # Extract housekeeping time series data and plot as a scatter plot aligned with frame indices
            series = getattr(metadata, field)
            ax.scatter(metadata_times, series, s=4, linewidth=0)
            ax.set_ylabel(ylabel_txt, fontsize=axis_label_fs)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
            ax.tick_params(axis="both", labelsize=tick_label_fs)
            ax.grid(True, alpha=0.3)

            # Only show the x-axis label on the very bottom panel to keep things clean
            if idx != (len(active_panels) - 1):
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("UTC date/time (frame axis)", fontsize=axis_label_fs)

    # Automatically generate folder structure and handle output file naming logic
    save_path.mkdir(parents=True, exist_ok=True)
    if output_name is not None:
        out_name = output_name
    else:
        out_name = f"{format_dt_for_fname(window.start)}_{format_dt_for_fname(window.end)}.png"
    
    out_path = save_path / out_name
    
    # Render and save out the final high-resolution PNG image
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.15)
    logger.debug("Figure saved: %s", out_path)

    # Display window or clean up memory depending on execution flags
    if show:
        logger.debug("Displaying figure window (--show enabled)...")
        plt.show()
    else:
        plt.close(fig)

    return out_path