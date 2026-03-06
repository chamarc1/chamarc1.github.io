# SWIR L1A Generator

Tools to process AirHARP2 SWIR images + metadata into L1A NetCDF4 products.

## What this repo does

- `l1agen_SWIR_NC_Processor.py`: processes one image range into one L1A NetCDF4 file.
- `l1agen_SWIR_Granule_Controller.py`: splits by granule time ranges and launches many NC processor runs in parallel.

## Requirements

- Python 3.9+
- Packages: `numpy`, `pandas`, `netCDF4`, `Pillow`

### Recommended setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy pandas netCDF4 Pillow
```

If your `requirements.txt` is maintained for your environment, you can also use:

```bash
pip install -r requirements.txt
```

## Input data expectations

### SWIR images

- Directory passed with `--img-dir` (used by both scripts)
- Expected naming pattern: `cfc_capture_<capture_id>.tif` or `.tiff`
- The capture ID must be parseable as an integer from the filename
- Corrupted/small images can be skipped by internal validation

### SWIR metadata files

- Directory passed with `--meta-dir`
- Files should end with `.meta`
- Filename timestamp format used by controller logic: `YYYYMMDD_HHMMSS_SWIR.meta`
- Metadata lines are expected in ACQ-style format, e.g.:
  - `ACQ:<capture_id>:<tec_temp>:<filter_pos>:<integration_time_ms>ms`

### AH2 IMU NetCDF

- Required by both scripts:
  - NC processor flag: `--ah2_imu_ncfile`
  - Controller flag: `--ah2-imu-ncfile`

### Optional ER2 navigation inputs

When `--include-er2-nav` is set, both must be supplied:

- `--er2-infile` (ER2 CSV/log)
- `--er2-xmlfile` (ER2 XML header)

## Quick start (single output file)

Run the NetCDF processor directly:

```bash
python l1agen_SWIR_NC_Processor.py \
  --img-dir /path/to/swir/images \
  --meta-dir /path/to/metadata \
  --output_dir /path/to/output \
  --start-img cfc_capture_1719352629638.tiff \
  --end-img cfc_capture_1719352929638.tiff \
  --level 1A \
  --ah2_imu_ncfile /path/to/PACEPAX_AH2.IMU.yyyymmdd.nc \
  -v
```

With ER2 navigation:

```bash
python l1agen_SWIR_NC_Processor.py \
  --img-dir /path/to/swir/images \
  --meta-dir /path/to/metadata \
  --output_dir /path/to/output \
  --start-img cfc_capture_1719352629638.tiff \
  --end-img cfc_capture_1719352929638.tiff \
  --level 1A \
  --ah2_imu_ncfile /path/to/PACEPAX_AH2.IMU.yyyymmdd.nc \
  --include-er2-nav \
  --er2-infile /path/to/IWG1_10HZ.csv \
  --er2-xmlfile /path/to/IWG1.xml \
  -v
```

## Batch processing with granules

Run the controller to process many granules:

```bash
python l1agen_SWIR_Granule_Controller.py \
  --img-dir /path/to/swir/images \
  --meta-dir /path/to/metadata \
  --output-dir /path/to/output \
  --ah2-imu-ncfile /path/to/PACEPAX_AH2.IMU.yyyymmdd.nc \
  --nc-processor-script /full/path/to/l1agen_SWIR_NC_Processor.py \
  --granule-file /path/to/GRANULE.DATE.txt \
  --execution-backend auto \
  --num-processes 4 \
  -v
```

### Important current behavior

- `--granule-file` is effectively required right now.
- Automatic granule calculation path exists, but controller currently exits if `--granule-file` is not provided.

### Execution backends

- `auto`: uses `slurm` if `sinfo` is available, otherwise `multiprocessing`
- `multiprocessing`: processes granules locally in Python workers
- `slurm`: writes/submits a batch script (or only writes it with `--disable-autorun`)
- `gnu-parallel`: writes/runs a GNU parallel script

Useful shell-script options for `slurm` / `gnu-parallel`:

- `--shell-output-dir` (default `./runs`)
- `--shell-output-fname` (default `l1agen_swir_batch.sh`)
- `--shell-log-output-dir` (default `./runs/log-%u`)
- `--disable-autorun` to generate scripts without launching

### Run the controller to process many granules using Slurm

Generate a Slurm batch script without auto-submitting:

```bash
python l1agen_SWIR_Granule_Controller.py \
  --img-dir /path/to/swir/images \
  --meta-dir /path/to/metadata \
  --output-dir /path/to/output \
  --ah2-imu-ncfile /path/to/PACEPAX_AH2.IMU.yyyymmdd.nc \
  --nc-processor-script /full/path/to/l1agen_SWIR_NC_Processor.py \
  --granule-file /path/to/GRANULE.DATE.txt \
  --execution-backend slurm \
  --disable-autorun \
  --shell-output-dir ./runs \
  --shell-output-fname l1agen_swir_batch.sh \
  --shell-log-output-dir ./runs/log-%u \
  --include-er2-nav \
  --er2-infile /path/to/infile \
  --er2-xmlfile /path/to/xmlfile \
  -v
```

Submit the generated script manually:

```bash
sbatch ./runs/l1agen_swir_batch.sh
```

If you want the controller to submit the Slurm job immediately, remove `--disable-autorun`.

## GRANULE file format

Expected CSV-style rows (comments with `#` are ignored):

```text
# Leg#, Granule#, Granule_start_time, Granule_end_time, Granule_duration
1, 1, 2024-09-10 20:57:00, 2024-09-10 21:02:00, 300
1, 2, 2024-09-10 21:02:00, 2024-09-10 21:07:00, 300
```

Timestamp format must be: `YYYY-MM-DD HH:MM:SS`.

## Outputs

- NetCDF files are written to the directory passed by `--output_dir` (NC processor) or `--output-dir` (controller).
- Controller-generated run scripts/logs default under `runs/`.
- Missing image-to-ACQ matches are appended to `missing_cfc_capture_ids.err` in output/log locations.

## Troubleshooting

- **No images found in a granule**: check `missing_cfc_capture_ids.err` and verify image capture IDs match metadata ACQ IDs.
- **ER2 nav errors**: when using `--include-er2-nav`, confirm both ER2 files exist and are readable.
- **Granule parse failures**: verify `GRANULE.DATE.txt` date-time format exactly matches `YYYY-MM-DD HH:MM:SS`.
- **Script path issues in controller**: pass an absolute path to `--nc-processor-script`.

## Project layout

```text
SWIR-L1AGen/
  l1agen_SWIR_NC_Processor.py
  l1agen_SWIR_Granule_Controller.py
  modules_l1a_SWIR/
  runs/
  tests/
  requirements.txt
```
