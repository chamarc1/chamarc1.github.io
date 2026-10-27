# AirHARP2 VNIR Quicklook

Generates PNG quicklook images from AirHARP2 VNIR L1A NetCDF4 files. Each output image shows pushbroom imagery at four fixed wavelengths and, optionally, sensor temperature — useful for rapidly reviewing data quality and coverage from a flight campaign.

## Table of Contents

- [Input Data Expectations](#input-data-expectations)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Mode 1 — Single time window](#mode-1--single-time-window)
  - [Mode 2 — Legwise quicklooks from a preprocessing file](#mode-2--legwise-quicklooks-from-a-preprocessing-file)
  - [Mode 3 — Multiple windows from a log file](#mode-3--multiple-windows-from-a-log-file)
  - [Mode 4 — Full coverage (no time filtering)](#mode-4--full-coverage-no-time-filtering)
- [All Options](#all-options)
- [Outputs](#outputs)
- [Code Overview](#code-overview)

---

## Input Data Expectations

### L1A NetCDF4 files

- Format: NetCDF4 (`.nc`)
- Source: AirHARP2 VNIR L1A data products
- Location: all `.nc` files should be in a single flat directory (subdirectories are not searched)
- Required metadata: each file must contain `time_coverage_start` and `time_coverage_end` global attributes in ISO 8601 format
- Files with valid time metadata but zero frames in `image_data/frames` are skipped automatically

### Preprocessing granule file (optional, for legwise mode)

A plain-text file listing granules grouped by flight leg. One granule per line, comma-separated. Lines beginning with `#` are treated as comments and ignored.

**Required columns (in order):** `Leg#`, `Granule#`, `Granule_start_time`, `Granule_end_time`

```text
# Leg#, Granule#, Granule_start_time, Granule_end_time, Granule_duration
1, 1, 2026-07-27 17:16:44, 2026-07-27 17:19:51, 03:07
1, 2, 2026-07-27 18:25:50, 2026-07-27 18:29:06, 03:16
2, 1, 2026-07-27 18:31:50, 2026-07-27 18:36:50, 05:00
```

### Log file (optional, for multi-window mode)

A plain-text file with one time window per line, comma-separated. Lines beginning with `#` are ignored.

**Required columns:** `start_time`, `end_time` (ISO 8601 format)

```text
# start_time, end_time
2026-03-23T12:00:00, 2026-03-23T12:30:00
2026-03-23T13:15:00, 2026-03-23T13:45:00
```

---

## Quick Start

**Step 1 — Set up your Python environment (one time only)**

Open a terminal in this folder and run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> On Windows, replace `source .venv/bin/activate` with `.venv\Scripts\activate`.

**Step 2 — Activate the environment (every new terminal session)**

```bash
source .venv/bin/activate
```

**Step 3 — Run the script**

The two required arguments are always:
- `--l1a_path`: the folder containing your `.nc` L1A files
- `--save_path`: the folder where PNG files will be saved (created automatically if it does not exist)

The simplest run — process all files in a folder with no time filtering:

```bash
python l1agen_ah2vnir_quicklook.py \
    --l1a_path /path/to/vnir/l1a/files \
    --save_path /path/to/output/pngs
```

Arguments can also be loaded from a file by prefixing its path with `@`:

```bash
python l1agen_ah2vnir_quicklook.py @quicklook_args.txt
```

The argument file uses one option per line. Values may be quoted, and lines
beginning with `#` are ignored:

```text
--l1a_path = "/path/to/vnir/l1a/files"
--save_path = "/path/to/output/pngs"
--preprocessing_file = "/path/to/GRANULE.txt"
--sensor_temp
--leg_output_naming
# --log_output_path = "/path/to/logs/vnir_quicklook"
```

To process a single flight leg using a preprocessing file (the most common use case):

```bash
python l1agen_ah2vnir_quicklook.py \
    --l1a_path /path/to/vnir/l1a/files \
    --save_path /path/to/output/pngs \
    --preprocessing_file /path/to/GRANULE.txt \
    --sensor_temp \
    --leg_output_naming \
    --verbose
```

---

## Usage

### Mode 1 — Single time window

Use `--start_time` and `--end_time` to generate one quicklook for a specific time range.

```bash
python l1agen_ah2vnir_quicklook.py \
    --l1a_path /path/to/vnir/l1a/files \
    --save_path /path/to/output/pngs \
    --start_time 2026-03-23T12:00:00 \
    --end_time 2026-03-23T12:30:00
```

### Mode 2 — Legwise quicklooks from a preprocessing file

Generates one PNG per flight leg. All granules belonging to the same leg are stitched into a single image.

```bash
python l1agen_ah2vnir_quicklook.py \
    --l1a_path /path/to/vnir/l1a/files \
    --save_path /path/to/output/pngs \
    --preprocessing_file /path/to/GRANULE.txt
```

> **Note:** `--preprocessing_file` cannot be combined with `--log_file`, `--start_time`, or `--end_time`.

Example from a recent campaign run:

```bash
python l1agen_ah2vnir_quicklook.py \
    --l1a_path /stor/z101/Data/INSPYRE/GreatFalls/L1_Data/VNIR/V0/2026-07-27 \
    --save_path /data/ESI/User/cmarc/Projects/AirHARP2/VNIR_output \
    --preprocessing_file /stor/z101/Data/INSPYRE/GreatFalls/L1_Data/PreProcessing/V0/GRANULE.2026-07-27.txt \
    --sensor_temp \
    --verbose
```

### Mode 3 — Multiple windows from a log file

Generates one PNG per time window listed in the log file.

```bash
python l1agen_ah2vnir_quicklook.py \
    --l1a_path /path/to/vnir/l1a/files \
    --save_path /path/to/output/pngs \
    --log_file /path/to/windows.txt
```

### Mode 4 — Full coverage (no time filtering)

If none of `--preprocessing_file`, `--log_file`, `--start_time`, or `--end_time` are provided, the script infers a single window from the first and last file's metadata and processes all files in the directory.

```bash
python l1agen_ah2vnir_quicklook.py \
    --l1a_path /path/to/vnir/l1a/files \
    --save_path /path/to/output/pngs
```

---

## All Options

| Argument | Required | Description |
|---|---|---|
| `--l1a_path` | Yes | Directory containing AirHARP2 VNIR L1A `.nc` files |
| `--save_path` | Yes | Output directory for PNG files |
| `--preprocessing_file` | No | Preprocessing granule file for leg-wise processing (see [Mode 2](#mode-2--legwise-quicklooks-from-a-preprocessing-file)) |
| `--log_file` | No | Log file with one `start_time, end_time` pair per line (see [Mode 3](#mode-3--multiple-windows-from-a-log-file)) |
| `--start_time` | No | ISO 8601 start time, e.g. `2026-03-23T12:00:00` (used with `--end_time`) |
| `--end_time` | No | ISO 8601 end time, e.g. `2026-03-23T12:30:00` (used with `--start_time`) |
| `--sensor_temp` | No | Add a sensor temperature panel to the quicklook image |
| `--leg_output_naming` | No | Use legacy naming from the first granule, replacing `_L##G###.nc` with `_L##.png` |
| `--show` | No | Open each plot in a window on screen in addition to saving it |
| `--verbose` / `-v` | No | Print step-by-step progress to the console |
| `--line_index` | No | Deprecated; accepted but ignored |
| `--log_output_path` | No | Directory for captured stdout/stderr; files are named `<process ID>.out` and `<process ID>.err` |
| `@argument_file` | No | Read command-line options from a file using one `--option = value` entry per line |

---

## Outputs

- **Format:** PNG image files saved to `--save_path`
- **Filename:** by default, `INSPYRE-AH2VNIR-L1A_ER2_YYYYMMDD_RA_LegNN-quicklook.png`, with the date and revision read from the first granule and `NN` the zero-padded leg number; non-leg windows use `Leg00`; use `--leg_output_naming` for the legacy `_L##.png` form
- **Logs:** with `--log_output_path /path/to/logs`, runs create `/path/to/logs/<process ID>.out` and `/path/to/logs/<process ID>.err`
- **One file per leg or window:** one PNG is created for each flight leg (preprocessing mode) or each time window (single/log/full-coverage mode)
- **Image contents:** four pushbroom panels at fixed wavelengths. Each panel shows measured
    sensor values as a function of detector cross-track pixel index and acquisition time/frame;
    the cross-track index is not a geographic location. Time is displayed on the vertical axis:
  - 763.5 nm
  - 589.2 nm
  - 567.7 nm
  - 546.1 nm
- **Optional panel:** sensor temperature vs. time (enabled with `--sensor_temp`)
- **Skipped outputs:** legs or windows where all matching files contain zero frames produce no PNG and a log message is printed instead

---

## Code Overview

| File | Purpose |
|---|---|
| [l1agen_ah2vnir_quicklook.py](l1agen_ah2vnir_quicklook.py) | Entry point, CLI argument parsing, workflow orchestration |
| [ql_modules/ql_io_netcdf.py](ql_modules/ql_io_netcdf.py) | NetCDF4 file discovery, metadata reads, time coverage parsing |
| [ql_modules/ql_transforms.py](ql_modules/ql_transforms.py) | Pure data transforms (datetime parsing, array operations) |
| [ql_modules/ql_plotting.py](ql_modules/ql_plotting.py) | Matplotlib rendering and PNG output |
| [ql_modules/ql_domain.py](ql_modules/ql_domain.py) | Immutable dataclasses (`TimeWindow`, `QuickplotConfig`) |
