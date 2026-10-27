from __future__ import annotations

import datetime as dt
from typing import Iterable, Sequence

import numpy as np

from .ql_domain import MetadataChunk, MetadataStack, TimeWindow


def parse_iso_datetime(value: str) -> dt.datetime:
    """Parse datetime strings with optional trailing 'Z'.
    
    Handles ISO 8601 datetime strings with optional UTC indicator ('Z').
    Used for parsing time_coverage_start/end from NetCDF global attributes.
    
    Args:
        value: ISO datetime string (e.g., "2024-09-10T18:23:40" or "2024-09-10T18:23:40Z")
    
    Returns:
        datetime.datetime object
    """

    return dt.datetime.fromisoformat(value.strip().replace("Z", ""))


def format_dt_for_fname(value: dt.datetime) -> str:
    """Create YYYYmmddTHHMMSS string for output file names.
    
    Formats a datetime into a compact, sortable filename component.
    Used for standardized quicklook PNG output naming.
    
    Args:
        value: datetime.datetime object
    
    Returns:
        Formatted string like "20240910T182340"
    """

    return value.strftime("%Y%m%dT%H%M%S")


def window_contains(file_start: dt.datetime, file_end: dt.datetime, window: TimeWindow) -> bool:
    """Pure predicate for inclusion of a file coverage interval in requested window.
    
    Checks if a file's entire time coverage falls within the requested selection window.
    Used to filter L1A files based on user-specified time bounds.
    
    Args:
        file_start: Start time of file coverage
        file_end: End time of file coverage
        window: User-requested TimeWindow for processing
    
    Returns:
        True if file coverage is fully contained within window, False otherwise
    """

    return (file_start >= window.start) and (file_end <= window.end)


def as_1d_float(values: np.ndarray) -> np.ndarray:
    """Convert NetCDF variable slices to 1D float array with NaNs for masked values.
    
    NetCDF4 masked arrays need conversion to plain float arrays for downstream
    processing. Masked values are replaced with NaN to indicate missing/invalid data.
    
    Args:
        values: NumPy array or masked array from NetCDF4
    
    Returns:
        1D float array with shape (n_elements,), masked values converted to NaN
    """

    arr = np.ma.asarray(values)
    return np.ma.filled(arr.astype(float), np.nan).reshape(-1)


def concat_chunks(chunks: Sequence[MetadataChunk]) -> MetadataStack:
    """Concatenate metadata vectors across all files in temporal order.
    
    Combines per-file metadata chunks into one stacked structure. All 1D metadata
    arrays are concatenated along the frame axis. The cumulative segment_ticks are
    computed to mark where each file's data ends, enabling visualization of
    granule/file boundaries in the quicklook plot.
    
    The segment_ticks are used to overlay vertical separation lines on the metadata
    plots and can be referenced to identify data gaps or timing anomalies between files.
    
    Args:
        chunks: Sequence of MetadataChunk objects (should be in temporal order)
    
    Returns:
        MetadataStack with all arrays concatenated and segment boundaries recorded
    """

    seconds_of_day = np.concatenate([chunk.seconds_of_day for chunk in chunks], axis=0)
    frame_id = np.concatenate([chunk.frame_id for chunk in chunks], axis=0)
    sensor_temperature = np.concatenate([chunk.sensor_temperature for chunk in chunks], axis=0)
    camera_temperature = np.concatenate([chunk.camera_temperature for chunk in chunks], axis=0)
    power_supply_temperature = np.concatenate([chunk.power_supply_temperature for chunk in chunks], axis=0)
    integration_time = np.concatenate([chunk.integration_time for chunk in chunks], axis=0)
    image_type = np.concatenate([chunk.image_type for chunk in chunks], axis=0)
    sensor_strips = np.concatenate([chunk.sensor_strips for chunk in chunks], axis=1)

    cumulative = 0
    segment_ticks: list[int] = []
    for chunk in chunks:
        cumulative += chunk.seconds_of_day.shape[0]
        segment_ticks.append(cumulative)

    return MetadataStack(
        seconds_of_day=seconds_of_day,
        frame_id=frame_id,
        sensor_temperature=sensor_temperature,
        camera_temperature=camera_temperature,
        power_supply_temperature=power_supply_temperature,
        integration_time=integration_time,
        image_type=image_type,
        sensor_strips=sensor_strips,
        segment_ticks=segment_ticks,
    )


def safe_vlims(arr: np.ndarray) -> tuple[float, float]:
    """Return robust plotting limits around mean +/- 3 sigma.
    
    Computes visualization color scale limits using robust statistics to handle
    outliers and ensure good contrast across a range of image characteristics.
    Falls back to (0.0, 1.0) if the array is empty, all NaN, or has zero variance.
    
    Args:
        arr: 2D image array (may contain NaN values)
    
    Returns:
        Tuple of (vmin, vmax) for use with matplotlib.pcolormesh()
    """

    if arr.size == 0:
        return 0.0, 1.0

    mu = np.nanmean(arr)
    sigma = np.nanstd(arr)
    if np.isnan(mu) or np.isnan(sigma) or sigma == 0.0:
        return 0.0, 1.0
    return mu - 3.0 * sigma, mu + 3.0 * sigma


def seconds_to_labels(
    seconds: np.ndarray,
    ticks: Iterable[int],
    start_date: dt.date | None = None,
) -> tuple[list[int], list[str]]:
    """Build y-axis ticks/labels for granule boundaries.
    
    Creates tick positions and formatted labels for the pushbroom image y-axis,
    marking where each L1A granule/file boundary occurs in the concatenated data.
    Times are displayed as UTC date and HH:MM:SS. If start_date is omitted,
    labels retain the legacy HH:MM:SS-only format.
    
    Args:
        seconds: 1D array of seconds_of_day values (one per frame)
        ticks: Iterable of cumulative frame indices at granule boundaries
        start_date: Calendar date associated with seconds_of_day midnight
    
    Returns:
        Tuple of (ytick_positions, ylabel_strings) for use with ax.set_yticks()
    """

    def format_seconds(value: float) -> str:
        if start_date is None:
            return str(dt.timedelta(seconds=int(value)))
        timestamp = dt.datetime.combine(start_date, dt.time()) + dt.timedelta(seconds=int(value))
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")

    if seconds.size == 0:
        return [0], [format_seconds(0.0)]

    ytick = [0]
    ylabel = [format_seconds(seconds[0])]

    tick_list = list(ticks)
    for tick in tick_list[:-1]:
        ytick.append(tick - 1)
        ylabel.append(format_seconds(seconds[tick - 1]))
        ytick.append(tick)
        ylabel.append(format_seconds(seconds[tick]))

    ytick.append(tick_list[-1] - 1)
    ylabel.append(format_seconds(seconds[-1]))

    return ytick, list(np.flipud(np.array(ylabel, dtype=object)))
