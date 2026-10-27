from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimeWindow:
    """Immutable time-window domain model for selection and plotting.
    
    Used to specify which time intervals should be processed and included
    in quicklook generation. The window bounds define the temporal scope for
    file selection and metadata extraction.
    """

    start: dt.datetime  # Start of the observation window (inclusive)
    end: dt.datetime    # End of the observation window (inclusive)


@dataclass(frozen=True)
class MetadataChunk:
    """Per-file metadata aligned on frame dimension.
    
    Represents raw metadata and image data extracted from a single VNIR L1A file.
    Each field is a 1D array indexed by frame number. The sensor_strips field is a 3D array
    of shape (n_pushbrooms, n_frames, n_pixels_per_line) extracted from the configured
    wavelength-specific sensor rows and will be transposed for visualization.
    
    Attributes:
        seconds_of_day: Acquisition time for each frame (UTC seconds within the day)
        frame_id: Sequential frame counter from the instrument
        sensor_temperature: Sensor temperature monitoring (Celsius)
        camera_temperature: Camera electronics temperature (Celsius)
        power_supply_temperature: Power supply module temperature (Celsius)
        integration_time: Sensor exposure time per frame (milliseconds)
        image_type: Image classification code (e.g., science, dark, flat, etc.)
        sensor_strips: 3D pushbroom image stack
                   Shape: (n_pushbrooms, n_frames, n_spectral_pixels)
                   Each slice will be transposed to (n_spectral_pixels, n_frames)
                   for visualization
    """

    seconds_of_day: np.ndarray
    frame_id: np.ndarray
    sensor_temperature: np.ndarray
    camera_temperature: np.ndarray
    power_supply_temperature: np.ndarray
    integration_time: np.ndarray
    image_type: np.ndarray
    sensor_strips: np.ndarray


@dataclass(frozen=True)
class MetadataStack:
    """Concatenated metadata across all selected files.
    
    Represents the complete metadata and image data for a time window,
    formed by concatenating MetadataChunks from all files in temporal order.
    All 1D arrays have matching length (total_frames across all files).
    
    The segment_ticks list records the cumulative frame count at each file
    boundary, allowing visualization of granule transitions in the final plot.
    
    Attributes:
        seconds_of_day: All acquisition times concatenated
        frame_id: All frame IDs concatenated
        sensor_temperature: All sensor temps concatenated
        camera_temperature: All camera temps concatenated
        power_supply_temperature: All power supply temps concatenated
        integration_time: All integration times concatenated
        image_type: All image types concatenated
        sensor_strips: Concatenated pushbroom image data
                   Shape: (n_pushbrooms, total_frames, n_pixels)
        segment_ticks: Cumulative frame counts marking file boundaries
                      Used for plotting granule separation lines on visualization
    """

    seconds_of_day: np.ndarray
    frame_id: np.ndarray
    sensor_temperature: np.ndarray
    camera_temperature: np.ndarray
    power_supply_temperature: np.ndarray
    integration_time: np.ndarray
    image_type: np.ndarray
    sensor_strips: np.ndarray
    segment_ticks: list[int]  # Cumulative frame indices at granule boundaries
