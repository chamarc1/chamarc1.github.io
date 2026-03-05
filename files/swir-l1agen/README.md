# SWIR L1A Generation (AirHARP2)

This folder summarizes the SWIR L1A generation workflow used to convert AirHARP2 SWIR raw image captures into Level 1A NetCDF4 products.

## Pipeline Overview

The original implementation included four core modules:

- `SWIR_Granule_Controller.py`
  - Enumerates image files in chronological order
  - Detects flight legs from timestamp gaps
  - Segments data into 5-minute (300 second) granules
  - Supports parallel processing orchestration

- `SWIR_Image_Data_TEC_Metadata_Matcher.py`
  - Locates `.tif` images and `.meta` metadata files
  - Matches records by CFC capture ID
  - Exports aligned metadata fields (TEC, filter position, integration time)

- `SWIR_Img_Processor.py`
  - Filters corrupted files by minimum file size
  - Parses CFC IDs from filenames
  - Builds matched image + metadata structures
  - Supports chronological retrieval and filter-position subsets

- `SWIR_NC_Processor.py`
  - Creates Level 1A NetCDF4 output structure
  - Writes image frames, timing, instrument state, and navigation data
  - Stores key fields including frame IDs, timestamps, JD, seconds of day,
    filter wheel position, integration time, and sensor temperature

## Data Product Focus

The workflow is designed for calibration-aware, reproducible SWIR Level 1A generation with robust metadata traceability and mission-oriented granule structure.
