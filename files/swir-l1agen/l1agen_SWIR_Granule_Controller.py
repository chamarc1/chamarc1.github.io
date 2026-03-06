"""
SWIR Granule Controller for AirHARP2 SWIR L1A Processing

This controller orchestrates SWIR image processing into L1A granules.
It handles:
1. Metadata file catalog building (extracting timestamps and ACQ IDs)
2. Pre-processed granule time ranges (from GRANULE.DATE.txt file)
3. Metadata-to-granule matching based on filename timestamps
4. Image selection via CFC capture ID matching
5. Parallel job generation and submission

Key Features:
- Adjusted ACQ ID timestamp-based granule matching (with time offset correction)
- Pre-processed granule file support (GRANULE.DATE.txt)
- Robust error handling for mismatched data
- Parallel execution support
- Detailed logging and progress tracking

Usage to use Python's built in multiprocessing (slow):
    python /data/ESI/User/cmarc/Projects/AirHARP2/SWIR_Projects/SWIR-L1AGen/l1agen_SWIR_Granule_Controller.py \
        --img-dir /stor/z101/Data/PACE-PAX/Shared/PACE-PAX_FlightData_20240910/SWIR \
        --meta-dir /stor/z101/Data/PACE-PAX/Shared/PACE-PAX_FlightData_20240910/SWIR/MetadataGroups \
        --output-dir /data/ESI/User/cmarc/Projects/AirHARP2/SWIR_output/nc_output_dir/20240910 \
        --ah2-imu-ncfile /stor/z101/Data/PACE-PAX/Shared/L1Data/PreL1A/20240910/PACEPAX_AH2.IMU.20240910.nc \
        --nc-processor-script //data/ESI/User/cmarc/Projects/AirHARP2/SWIR_Projects/SWIR-L1AGen/l1agen_SWIR_NC_Processor.py \
        --include-er2-nav \
        --granule-file /stor/z101/Data/PACE-PAX/Shared/L1Data/PreL1A/20240910/GRANULE.2024-09-10.txt \
        --er2-infile /stor/z101/Data/PACE-PAX/Shared/L1Data/ER2-IMU/2024-09-10/IWG1_10HZ.10Sep2024-2057 \
        --er2-xmlfile /stor/z101/Data/PACE-PAX/Shared/L1Data/ER2-IMU/2024-09-10/IWG1.xml \
        -v

Usage to generate shell script:
    python /data/ESI/User/cmarc/Projects/AirHARP2/SWIR_Projects/SWIR-L1AGen/l1agen_SWIR_Granule_Controller.py \
        --img-dir /stor/z101/Data/PACE-PAX/Shared/PACE-PAX_FlightData_20240910/SWIR \
        --meta-dir /stor/z101/Data/PACE-PAX/Shared/PACE-PAX_FlightData_20240910/SWIR/MetadataGroups \
        --output-dir /data/ESI/User/cmarc/Projects/AirHARP2/SWIR_output/nc_output_dir/20240910 \
        --ah2-imu-ncfile /stor/z101/Data/PACE-PAX/Shared/L1Data/PreL1A/20240910/PACEPAX_AH2.IMU.20240910.nc \
        --nc-processor-script /data/home/cmarc/Projects/AirHARP2/SWIR_Projects/SWIR-L1AGen/l1agen_SWIR_NC_Processor.py \
        --granule-file /stor/z101/Data/PACE-PAX/Shared/L1Data/PreL1A/20240910/GRANULE.2024-09-10.txt \
        --execution-backend slurm \
        --disable-autorun \
        --shell-output-dir ./runs \
        --shell-output-fname l1agen_swir_batch.sh \
        --shell-log-output-dir ./runs/log-%u \
        --include-er2-nav \
        --er2-infile /stor/z101/Data/PACE-PAX/Shared/L1Data/ER2-IMU/2024-09-10/IWG1_10HZ.10Sep2024-2057 \
        --er2-xmlfile /stor/z101/Data/PACE-PAX/Shared/L1Data/ER2-IMU/2024-09-10/IWG1.xml -v

Shell Usage:
    sbatch /path/to/l1agen_swir_batch.sh

Configuration:
    Required command-line arguments (see --help for full list)

__name__ =      l1agen_SWIR_Granule_Controller.py
__author__ =    "Charlemagne Marc"
__copyright__ = "Copyright 2025, ESI SWIR Project"
__credits__ =   ["Charlemagne Marc"]
__version__ =   "1.6.5"
__maintainer__ ="Charlemagne Marc"
__email__ =     "chamrc1@umbc.edu"
__status__ =    "Production"
"""

#----------------------------------------------------------------------------
#-- IMPORT STATEMENTS
#----------------------------------------------------------------------------
import os
import sys
import datetime as dt
import numpy as np
import argparse
import subprocess
import re
import shlex
from pathlib import Path
from multiprocessing import Pool
from functools import partial

# Import time offset functions from the matcher module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modules_l1a_SWIR'))
from l1agen_SWIR_Image_Data_TEC_Metadata_Matcher import (
    CalculateTimeOffset,
    ApplyTimeOffsetToAcqIDs,
    ConvertTimestampToUTC
)

#----------------------------------------------------------------------------
#-- FUNCTION DEFINITIONS
#----------------------------------------------------------------------------

def parse_input_args():
    """
    Define input argument parser for this script
    
    :return args: argparse.Namespace class instance, which holds arguments and their values
    """
    #-- create argument parser
    parser = argparse.ArgumentParser(description='Controller for SWIR L1A granule processing')
    
    #-- required arguments
    parser.add_argument('--img-dir', type=str, required=True, help='Directory containing SWIR image files (.tif)')
    parser.add_argument('--meta-dir', type=str, required=True, help='Directory containing SWIR metadata files (.meta)')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for NetCDF4 files')
    parser.add_argument('--ah2-imu-ncfile', type=str, required=True, help='Full path to AH2 IMU netcdf file')
    parser.add_argument('--nc-processor-script', type=str, required=True, help='Full path to l1agen_SWIR_NC_Processor.py script')
    
    #-- optional arguments
    parser.add_argument('--level', type=str, default='1A', help='Processing level (default: 1A)')
    parser.add_argument('--time-tolerance', type=int, default=5, help='Gap threshold in seconds to identify flight leg breaks (default: 5)')
    parser.add_argument('--granule-duration', type=int, default=300, help='Granule duration in seconds (default: 300 = 5 minutes)')
    parser.add_argument('--granule-file', type=str, default=None, help='Path to pre-processed granule file (GRANULE.DATE.txt). If provided, overrides automatic granule calculation')
    parser.add_argument('--num-processes', type=int, default=4, help='Number of parallel processes (default: 4)')
    parser.add_argument('--execution-backend', type=str, choices=['auto', 'multiprocessing', 'slurm', 'gnu-parallel'], default='auto',
                        help='Execution backend (default: auto = slurm if available, else multiprocessing)')
    parser.add_argument('--include-er2-nav', action='store_true', help='Include ER2 navigation data')
    parser.add_argument('--er2-infile', type=str, default=None, help='Full path to ER2 log file (required if --include-er2-nav is set)')
    parser.add_argument('--er2-xmlfile', type=str, default=None, help='Full path to ER2 XML file (required if --include-er2-nav is set)')
    parser.add_argument('--shell-output-fname', type=str, default='l1agen_swir_batch.sh',
                        help='Shell script output filename for slurm/gnu-parallel backends')
    parser.add_argument('--shell-output-dir', type=str, default='./runs',
                        help='Shell script output directory for slurm/gnu-parallel backends')
    parser.add_argument('--shell-log-output-dir', type=str, default='./runs/log-%u',
                        help='Shell log output directory for slurm/gnu-parallel backends')
    parser.add_argument('--disable-autorun', action='store_true',
                        help='Generate script only; do not submit/run automatically for slurm/gnu-parallel backends')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    
    #-- return arguments
    return parser.parse_args()



def extract_metadata_file_timestamp(meta_fname):
    """
    Extract timestamp from metadata filename.
    
    Expected metadata filename format: YearMonthDay_HourMinuteSecond_SWIR.meta
    Example: 20240904_172948_SWIR.meta
    
    :param meta_fname: string metadata filename
    :return: datetime object, or None if extraction fails
    """
    try:
        # Extract the date and time portion (e.g., "20240904_172948")
        match = re.match(r'(\d{8})_(\d{6})', meta_fname)
        if match:
            date_str = match.group(1)  # YYYYMMDD
            time_str = match.group(2)  # HHMMSS
            
            # Parse into datetime
            datetime_str = f"{date_str}{time_str}"
            meta_time = dt.datetime.strptime(datetime_str, '%Y%m%d%H%M%S')
            return meta_time
    except (ValueError, AttributeError):
        pass
    return None


def is_hidden_or_sidecar_file(fname):
    """
    Return True for hidden/system sidecar files that should be ignored.

    This primarily filters macOS AppleDouble files (e.g., ._filename.ext)
    and other dot-prefixed entries.

    :param fname: string filename
    :return: bool
    """
    return fname.startswith('.')


def build_metadata_catalog(meta_dir):
    """
    Build a catalog of metadata files with their timestamps and ACQ IDs
    
    Uses adjusted ACQ ID times (after time offset correction) instead of filename timestamps
    for accurate granule matching.
    
    :param meta_dir: directory containing metadata files
    :return: dict mapping metadata filenames to {time: datetime, acq_ids: list, acq_times: list}
    """
    print("\n" + "="*70)
    print("BUILDING METADATA CATALOG")
    print("="*70)
    
    catalog = {}
    
    try:
        # Find all visible .meta files (skip hidden/system sidecar files)
        meta_files = [
            f for f in os.listdir(meta_dir)
            if f.endswith('.meta') and not is_hidden_or_sidecar_file(f)
        ]
        
        if not meta_files:
            print("WARNING: No metadata files found")
            return catalog
        
        meta_files.sort()
        
        for meta_file in meta_files:
            # Read ACQ IDs from metadata file
            meta_path = os.path.join(meta_dir, meta_file)
            acq_ids = []
            
            try:
                with open(meta_path, 'r') as f:
                    for line in f:
                        if 'ACQ' in line:
                            line = line.replace(':', ';').replace(' ', '').rstrip('\n').split(';')
                            acq_id = int(line[1])
                            acq_ids.append(acq_id)
            except (IOError, ValueError, IndexError) as e:
                print(f"WARNING: Error reading {meta_file}: {e}")
                continue
            
            if not acq_ids:
                print(f"WARNING: No ACQ IDs found in {meta_file}, skipping")
                continue
            
            # Calculate time offset using first ACQ ID and filename timestamp
            first_acq_id = acq_ids[0]
            offset_ms = CalculateTimeOffset(first_acq_id, meta_file)
            
            if offset_ms is None:
                print(f"WARNING: Could not calculate time offset for {meta_file}, skipping")
                continue
            
            # Apply time offset to all ACQ IDs
            adjusted_acq_ids = ApplyTimeOffsetToAcqIDs(np.array(acq_ids), offset_ms)
            
            # Convert adjusted ACQ IDs to datetime objects
            acq_times = []
            for adj_acq_id in adjusted_acq_ids:
                time_str = ConvertTimestampToUTC(adj_acq_id)
                if time_str:
                    acq_time = dt.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                    acq_times.append(acq_time)
            
            if not acq_times:
                print(f"WARNING: Could not convert ACQ IDs to times for {meta_file}, skipping")
                continue
            
            # Store metadata file info with both original and adjusted ACQ IDs
            # Use the middle time as representative time for the file
            middle_time = acq_times[len(acq_times) // 2]
            
            catalog[meta_file] = {
                'time': middle_time,  # Representative time (middle of ACQ times)
                'acq_ids': acq_ids,  # Original (raw) ACQ IDs for image matching
                'adjusted_acq_ids': list(adjusted_acq_ids),  # Adjusted ACQ IDs
                'acq_times': acq_times,  # Adjusted ACQ times as datetime objects
                'time_range': (min(acq_times), max(acq_times))  # (earliest, latest) time
            }
        
        print(f"Metadata files cataloged: {len(catalog)}")
        if catalog:
            all_times = [info['time_range'][0] for info in catalog.values()]
            first_time = min(all_times)
            all_times = [info['time_range'][1] for info in catalog.values()]
            last_time = max(all_times)
            
            first_file = [f for f, info in catalog.items() if info['time_range'][0] == first_time][0]
            last_file = [f for f, info in catalog.items() if info['time_range'][1] == last_time][0]
            
            print(f"  First: {first_file} @ {first_time.strftime('%Y-%m-%d %H:%M:%S')} (adjusted ACQ time)")
            print(f"  Last:  {last_file} @ {last_time.strftime('%Y-%m-%d %H:%M:%S')} (adjusted ACQ time)")
        
        return catalog
        
    except Exception as e:
        print(f"ERROR: Failed to build metadata catalog: {e}")
        import traceback
        traceback.print_exc()
        return {}


def extract_cfc_capture_id_from_filename(fname):
    """
    Extract CFC capture ID from SWIR image filename
    
    Expected format: cfc_capture_<capture_id>.tiff
    
    :param fname: string filename
    :return: capture ID (int), or None if cannot parse
    """
    try:
        # Remove file extension
        base_name = fname.rsplit('.', 1)[0]
        
        # Try format: cfc_capture_<capture_id>
        if base_name.startswith("cfc_capture_"):
            capture_id = int(base_name.split("_")[-1])
            return capture_id
    except (ValueError, IndexError):
        pass
    
    return None


def enumerate_swir_images(img_dir, start_img=None, end_img=None):
    """
    Enumerate all SWIR image files with their CFC capture IDs
    
    :param img_dir:   string path to SWIR image directory
    :param start_img: optional start image filename
    :param end_img:   optional end image filename
    :return: dict mapping CFC capture IDs to filenames and paths
    """
    
    print("\n" + "="*70)
    print("ENUMERATING SWIR IMAGES")
    print("="*70)
    
    # List and sort all visible TIFF files (skip hidden/system sidecar files)
    all_files = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith(('.tif', '.tiff')) and not is_hidden_or_sidecar_file(f)
    ]
    all_files.sort()
    
    if not all_files:
        print(f"ERROR: No TIFF files found in {img_dir}")
        sys.exit(1)
    
    # Apply start/end filters if specified
    if start_img and end_img:
        try:
            idx_start = all_files.index(start_img)
            idx_end = all_files.index(end_img)
            all_files = all_files[idx_start:idx_end + 1]
            print(f"Filtered: {len(all_files)} files from {start_img} to {end_img}")
        except ValueError as e:
            print(f"ERROR: Start/end image not found: {e}")
            sys.exit(1)
    
    # Build mapping of CFC capture IDs to image files
    img_catalog = {}
    
    for fname in all_files:
        capture_id = extract_cfc_capture_id_from_filename(fname)
        
        if capture_id is None:
            print(f"WARNING: Could not parse CFC capture ID from {fname}, skipping")
            continue
        
        img_catalog[capture_id] = {
            'fname': fname,
            'fpath': os.path.join(img_dir, fname)
        }
    
    print(f"Total SWIR images enumerated: {len(img_catalog)}")
    if img_catalog:
        first_id = min(img_catalog.keys())
        last_id = max(img_catalog.keys())
        print(f"  First: {img_catalog[first_id]['fname']} (CFC ID: {first_id})")
        print(f"  Last:  {img_catalog[last_id]['fname']} (CFC ID: {last_id})")
    
    return img_catalog


def identify_flight_legs(img_finfo, time_tolerance_sec=5):
    """
    Identify flight legs by detecting time gaps between images
    
    :param img_finfo:         dict with image info (fnames, ftimes, etc)
    :param time_tolerance_sec: threshold gap to identify leg boundaries (default: 5 seconds)
    :return: list of tuples (leg_start_time, leg_end_time)
    """
    
    print("\n" + "="*70)
    print("IDENTIFYING FLIGHT LEGS")
    print("="*70)
    
    if not img_finfo["ftimes"]:
        print("ERROR: No image times available")
        sys.exit(1)
    
    flight_legs = []
    leg_start = img_finfo["ftimes"][0]
    prev_time = leg_start
    
    for current_time in img_finfo["ftimes"][1:]:
        time_gap = (current_time - prev_time).total_seconds()
        
        # If gap exceeds threshold, end current leg and start new one
        if time_gap > time_tolerance_sec:
            flight_legs.append((leg_start, prev_time))
            print(f"  Leg: {leg_start.strftime('%Y-%m-%d %H:%M:%S')} to {prev_time.strftime('%H:%M:%S')} "
                  f"({(prev_time - leg_start).total_seconds():.0f}s)")
            leg_start = current_time
        
        prev_time = current_time
    
    # Add final leg
    flight_legs.append((leg_start, img_finfo["ftimes"][-1]))
    print(f"  Leg: {leg_start.strftime('%Y-%m-%d %H:%M:%S')} to {prev_time.strftime('%H:%M:%S')} "
          f"({(prev_time - leg_start).total_seconds():.0f}s)")
    
    print(f"Total flight legs identified: {len(flight_legs)}")
    
    return flight_legs


def divide_into_granules(flight_legs, granule_duration_sec=300):
    """
    Divide flight legs into 5-minute granules
    
    :param flight_legs:          list of (start_time, end_time) tuples
    :param granule_duration_sec: duration of each granule in seconds (default: 300 = 5 minutes)
    :return: list of (granule_start, granule_end) tuples
    """
    
    print("\n" + "="*70)
    print("DIVIDING LEGS INTO 5-MINUTE GRANULES")
    print("="*70)
    
    granules = []
    total_granules = 0
    
    for leg_idx, (leg_start, leg_end) in enumerate(flight_legs):
        leg_duration = (leg_end - leg_start).total_seconds()
        
        # Calculate number of complete granules in this leg
        num_complete_granules = int(leg_duration // granule_duration_sec)
        
        current_time = leg_start
        
        for gran_idx in range(num_complete_granules):
            granule_start = current_time
            granule_end = granule_start + dt.timedelta(seconds=granule_duration_sec)
            granules.append((granule_start, granule_end))
            current_time = granule_end
            total_granules += 1
        
        # Handle remaining partial granule (if > 30 seconds)
        remaining_duration = (leg_end - current_time).total_seconds()
        if remaining_duration > 30:  # Threshold to avoid tiny granules
            granules.append((current_time, leg_end))
            total_granules += 1
            print(f"  Leg {leg_idx+1}: {num_complete_granules} complete 5-min + 1 partial ({remaining_duration:.0f}s)")
        else:
            print(f"  Leg {leg_idx+1}: {num_complete_granules} complete 5-min granules")
    
    print(f"Total granules created: {total_granules}")
    if granules:
        print(f"  First: {granules[0][0].strftime('%Y-%m-%d %H:%M:%S')} to {granules[0][1].strftime('%H:%M:%S')}")
        print(f"  Last:  {granules[-1][0].strftime('%H:%M:%S')} to {granules[-1][1].strftime('%H:%M:%S')}")
    
    return granules


def load_granules_from_file(granule_file):
    """
    Load pre-processed granule times from a text file
    
    File format (GRANULE.DATE.txt):
    # number of granules: N
    # Leg#, Granule#, Granule_start_time, Granule_end_time, Granule_duration
    leg, granule_num, start_time, end_time, duration
    
    :param granule_file: string path to granule file
    :return: list of granule dicts with start/end and optional leg/granule numbers
    """
    
    print("\n" + "="*70)
    print("LOADING GRANULES FROM FILE")
    print("="*70)
    print(f"Granule file: {granule_file}")
    
    if not os.path.exists(granule_file):
        print(f"ERROR: Granule file not found: {granule_file}")
        sys.exit(1)
    
    granules = []
    
    try:
        with open(granule_file, 'r') as f:
            lines = f.readlines()
        
        # Parse file, skipping comments and empty lines
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Parse data line: leg, granule_num, start_time, end_time, duration
            parts = [p.strip() for p in line.split(',')]
            
            if len(parts) < 4:
                print(f"WARNING: Skipping line {line_num} (insufficient columns): {line}")
                continue
            
            try:
                # Extract start and end times
                start_time_str = parts[2]
                end_time_str = parts[3]
                
                # Parse timestamps (format: YYYY-MM-DD HH:MM:SS)
                granule_start = dt.datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
                granule_end = dt.datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")

                leg_number = None
                granule_number = None
                try:
                    leg_number = int(parts[0])
                    granule_number = int(parts[1])
                except (ValueError, IndexError):
                    pass

                granules.append({
                    'start': granule_start,
                    'end': granule_end,
                    'leg_number': leg_number,
                    'granule_number': granule_number,
                })
                
            except (ValueError, IndexError) as e:
                print(f"WARNING: Failed to parse line {line_num}: {line}")
                print(f"         Error: {e}")
                continue
        
        if not granules:
            print("ERROR: No valid granules parsed from file")
            sys.exit(1)
        
        print(f"Total granules loaded: {len(granules)}")
        if granules:
            first_start, first_end, _ = parse_granule_entry(granules[0], 1)
            last_start, last_end, _ = parse_granule_entry(granules[-1], len(granules))
            print(f"  First: {first_start.strftime('%Y-%m-%d %H:%M:%S')} to {first_end.strftime('%H:%M:%S')}")
            print(f"  Last:  {last_start.strftime('%H:%M:%S')} to {last_end.strftime('%H:%M:%S')}")
        
        return granules
        
    except IOError as e:
        print(f"ERROR: Failed to read granule file: {e}")
        sys.exit(1)


def parse_granule_entry(granule_entry, granule_idx):
    """
    Parse a granule entry into start/end datetimes and a display label.

    Supports both tuple entries: (start, end) and dict entries from
    load_granules_from_file() with leg/granule numbering.

    :return: (granule_start, granule_end, granule_label)
    """

    if isinstance(granule_entry, dict):
        granule_start = granule_entry['start']
        granule_end = granule_entry['end']
        leg_number = granule_entry.get('leg_number')
        granule_number = granule_entry.get('granule_number')

        if leg_number is not None and granule_number is not None:
            granule_label = f"Leg {leg_number}, Granule {granule_number}"
        else:
            granule_label = f"Granule {granule_idx}"

        return granule_start, granule_end, granule_label

    granule_start, granule_end = granule_entry
    return granule_start, granule_end, f"Granule {granule_idx}"


def write_missing_cfc_capture_error(nc_output_dir, granule_label, granule_start, granule_end, matching_acq_ids):
    """
    Write a .err file describing missing CFC capture IDs for a granule.

    Writes to:
      1) nc_output_dir
      2) controller runs/log-<user> (mirror)

    :return: list of updated .err file paths
    """

    written_paths = []

    try:
        os.makedirs(nc_output_dir, exist_ok=True)

        err_fname = "missing_cfc_capture_ids.err"
        err_fpath = os.path.join(nc_output_dir, err_fname)

        target_paths = [err_fpath]

        controller_dir = os.path.dirname(os.path.abspath(__file__))
        user_name = os.getenv('USER') or 'unknown'
        run_date = dt.datetime.now().strftime('%Y-%m-%d')
        mirror_dir = os.path.join(controller_dir, f"runs/log-{user_name}", run_date)
        mirror_fpath = os.path.join(mirror_dir, err_fname)
        target_paths.append(mirror_fpath)

        error_content = (
            f"{'=' * 80}\n"
            f"{granule_label}: No images found for {len(matching_acq_ids)} ACQ IDs\n"
            f"Granule time range: {granule_start.strftime('%Y-%m-%d %H:%M:%S')} to "
            f"{granule_end.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Missing CFC capture IDs (from ACQ IDs):\n"
            f"{','.join(str(capture_id) for capture_id in sorted(set(matching_acq_ids)))}\n"
            f"{'=' * 80}\n\n"
        )

        for path in target_paths:
            parent_dir = os.path.dirname(path)
            os.makedirs(parent_dir, exist_ok=True)
            with open(path, 'a') as err_file:
                err_file.write(error_content)
            written_paths.append(path)

        return written_paths
    except Exception:
        return written_paths


def process_granule(granule_idx, granule_start, granule_end, granule_label, img_catalog, meta_catalog, swir_img_dir, swir_meta_dir, 
                    nc_output_dir, nc_processing_level, swir_nc_processor_script, ah2_imu_nc,
                    include_er2_nav=False, er2_imu_log=None, er2_imu_xml=None):
    """
    Process a single granule using SWIR_NC_Processor
    
    :param granule_idx: index of granule (for progress tracking)
    :param granule_start: start time of granule
    :param granule_end: end time of granule
    :param img_catalog: dict mapping CFC capture IDs to image info
    :param meta_catalog: dict mapping metadata filenames to their info
    :param swir_img_dir: SWIR image directory
    :param swir_meta_dir: SWIR metadata directory
    :param nc_output_dir: NetCDF4 output directory
    :param nc_processing_level: processing level (e.g., "1A")
    :param swir_nc_processor_script: path to SWIR_NC_Processor.py
    :param ah2_imu_nc: path to AH2 IMU NetCDF file
    :param include_er2_nav: whether to include ER2 navigation
    :param er2_imu_log: path to ER2 IMU log file
    :param er2_imu_xml: path to ER2 IMU XML file
    :return: tuple (granule_idx, success, message)
    """
    
    cmd, granule_images, error_msg = build_granule_command(
        granule_idx, granule_start, granule_end, img_catalog, meta_catalog,
        swir_img_dir, swir_meta_dir, nc_output_dir, nc_processing_level, granule_label,
        swir_nc_processor_script, ah2_imu_nc, include_er2_nav, er2_imu_log, er2_imu_xml
    )

    if cmd is None:
        return (granule_idx, False, error_msg)

    try:
        # Debug: print command for first few granules
        if granule_idx <= 2:
            print(f"[DEBUG] {granule_label} command: {' '.join(cmd)}")

        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=3600)
        return (granule_idx, True, f"{granule_label} ({len(granule_images)} images) processed successfully")
    except subprocess.TimeoutExpired:
        return (granule_idx, False, f"{granule_label}: Processing timeout (>1 hour)")
    except subprocess.CalledProcessError as e:
        return (granule_idx, False, f"{granule_label}: {e.stderr[:200]}")


def resolve_shell_paths(shell_output_dir, shell_log_output_dir, controller_dir):
    """
    Resolve shell output/log directories from defaults to absolute paths.

    :return: (resolved_shell_output_dir, resolved_shell_log_output_dir)
    """

    if shell_output_dir == "./runs":
        shell_output_dir = os.path.join(controller_dir, "runs")

    if shell_log_output_dir == "./runs/log-%u":
        user_name = os.getenv('USER') or 'unknown'
        run_date = dt.datetime.now().strftime('%Y-%m-%d')
        shell_log_output_dir = os.path.join(controller_dir, f"runs/log-{user_name}", run_date)

    return shell_output_dir, shell_log_output_dir


def get_environment_activation_line():
    """
    Get shell line used to activate the currently running Python environment.
    """

    if os.getenv("VIRTUAL_ENV") is not None:
        return f"source \"{os.path.join(os.getenv('VIRTUAL_ENV'), 'bin/activate')}\""

    return f"conda activate {os.path.basename(sys.prefix)}"


def collect_granule_command_lines(granules, img_catalog, meta_catalog, swir_img_dir, swir_meta_dir,
                                  nc_output_dir, nc_processing_level, swir_nc_processor_script,
                                  ah2_imu_nc, include_er2_nav, er2_imu_log, er2_imu_xml):
    """
    Build shell-safe processor command lines for all valid granules.

    :return: (command_lines, skipped_count)
    """

    command_lines = []
    skipped = 0

    for idx, granule_entry in enumerate(granules, 1):
        granule_start, granule_end, granule_label = parse_granule_entry(granule_entry, idx)
        cmd, _, error_msg = build_granule_command(
            idx, granule_start, granule_end, img_catalog, meta_catalog,
            swir_img_dir, swir_meta_dir, nc_output_dir, nc_processing_level, granule_label,
            swir_nc_processor_script, ah2_imu_nc, include_er2_nav, er2_imu_log, er2_imu_xml
        )

        if cmd is None:
            skipped += 1
            print(f"WARNING: {error_msg}")
            continue

        command_lines.append(shlex.join(cmd))

    return command_lines, skipped


def write_slurm_script(script_path, command_lines, shell_log_output_dir):
    """
    Write Slurm batch script for one-task-per-granule execution.
    """

    with open(script_path, "w") as output_sh:
        output_sh.write("#!/bin/bash\n\n")
        output_sh.write(f"""#SBATCH --job-name=L1AGen_SWIR
#SBATCH --ntasks={len(command_lines)}
#SBATCH --partition=zen4
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --output={shell_log_output_dir}/%j.out
#SBATCH --error={shell_log_output_dir}/%j.err
""")
        output_sh.write("\nSTART_TIME=$(date +\"%s\")\n")
        output_sh.write("echo JOB START TIME: $(date -d @$START_TIME)\n")
        output_sh.write("echo GRANULE COUNTER: enabled\n")
        output_sh.write("echo\n")
        output_sh.write(f"{get_environment_activation_line()}\n")
        output_sh.write("\n")
        output_sh.write(f"TOTAL_GRANULES={len(command_lines)}\n")
        output_sh.write("SUCCESSFUL_GRANULES=0\n")
        output_sh.write("FAILED_GRANULES=0\n")
        output_sh.write("PIDS=()\n")
        for cmd_line in command_lines:
            output_sh.write(f"srun --nodes=1 --ntasks=1 --mem-per-cpu=4G --exclusive {cmd_line} &\n")
            output_sh.write("PIDS+=(\"$!\")\n")
        output_sh.write("for pid in \"${PIDS[@]}\"; do\n")
        output_sh.write("  if wait \"$pid\"; then\n")
        output_sh.write("    SUCCESSFUL_GRANULES=$((SUCCESSFUL_GRANULES + 1))\n")
        output_sh.write("  else\n")
        output_sh.write("    FAILED_GRANULES=$((FAILED_GRANULES + 1))\n")
        output_sh.write("  fi\n")
        output_sh.write("done\n\n")
        output_sh.write("END_TIME=$(date +\"%s\")\n")
        output_sh.write("DURATION=$(( END_TIME - START_TIME ))\n")
        output_sh.write("echo JOB END TIME: $(date -d @$END_TIME)\n")
        output_sh.write("echo ELAPSED: $(date -d @$DURATION -u +%T)\n")
        output_sh.write("echo GRANULES: ${SUCCESSFUL_GRANULES}/${TOTAL_GRANULES} completed\n")
        output_sh.write("echo GRANULES FAILED: ${FAILED_GRANULES}\n")

    os.chmod(script_path, 0o755)


def write_gnu_parallel_script(script_path, command_lines, shell_log_output_dir, num_processes):
    """
    Write GNU parallel shell script for one-command-per-granule execution.
    """

    with open(script_path, "w") as output_sh:
        output_sh.write("#!/bin/bash\n\n")
        output_sh.write(f"exec 1>{shell_log_output_dir}/$$.out\n")
        output_sh.write(f"exec 2>{shell_log_output_dir}/$$.err\n\n")
        output_sh.write("START_TIME=$(date +\"%s\")\n")
        output_sh.write("echo JOB START TIME: $(date -d @$START_TIME)\n")
        output_sh.write("echo\n")
        output_sh.write(f"{get_environment_activation_line()}\n\n")
        output_sh.write("parallel -j {} <<'EOF'\n".format(num_processes))
        for cmd_line in command_lines:
            output_sh.write(f"{cmd_line}\n")
        output_sh.write("EOF\n\n")
        output_sh.write("END_TIME=$(date +\"%s\")\n")
        output_sh.write("DURATION=$(( END_TIME - START_TIME ))\n")
        output_sh.write("echo JOB END TIME: $(date -d @$END_TIME)\n")
        output_sh.write("echo ELAPSED: $(date -d @$DURATION -u +%T)\n")

    os.chmod(script_path, 0o755)


def build_granule_command(granule_idx, granule_start, granule_end, img_catalog, meta_catalog,
                          swir_img_dir, swir_meta_dir, nc_output_dir, nc_processing_level, granule_label,
                          swir_nc_processor_script, ah2_imu_nc,
                          include_er2_nav=False, er2_imu_log=None, er2_imu_xml=None):
    """
    Build per-granule command for l1agen_SWIR_NC_Processor.py.

    Uses the same matching logic as process_granule so all execution backends
    (multiprocessing, slurm, gnu-parallel) produce identical granule selections.

    :return: (cmd_list | None, granule_images_list, error_message | None)
    """

    if not granule_label:
        granule_label = f"Granule {granule_idx}"

    # Find metadata files whose adjusted ACQ times overlap with this granule time range
    matching_acq_ids = []
    for meta_file, meta_info in meta_catalog.items():
        # Check if the metadata file's time range overlaps with granule time range
        meta_start, meta_end = meta_info['time_range']
        
        # Overlap check: metadata overlaps granule if:
        # meta_start <= granule_end AND meta_end >= granule_start
        if meta_start <= granule_end and meta_end >= granule_start:
            # Find which specific ACQ IDs fall within the granule range
            # Use adjusted times for selection, but return original ACQ IDs for image matching
            for i, acq_time in enumerate(meta_info['acq_times']):
                if granule_start <= acq_time <= granule_end:
                    # Use original ACQ ID for image lookup
                    matching_acq_ids.append(meta_info['acq_ids'][i])
    
    if not matching_acq_ids:
        return None, [], f"{granule_label}: No metadata files found in time range"
    
    # Find images with matching CFC capture IDs
    granule_images = []
    for acq_id in matching_acq_ids:
        if acq_id in img_catalog:
            granule_images.append(img_catalog[acq_id]['fname'])
    
    if not granule_images:
        err_paths = write_missing_cfc_capture_error(
            nc_output_dir=nc_output_dir,
            granule_label=granule_label,
            granule_start=granule_start,
            granule_end=granule_end,
            matching_acq_ids=matching_acq_ids
        )

        error_msg = f"{granule_label}: No images found for {len(matching_acq_ids)} ACQ IDs\n"
        if err_paths:
            error_msg += f" (details: {err_paths[0]})\n"
            if len(err_paths) > 1:
                error_msg += f" (mirror: {err_paths[1]})\n"

        return None, [], error_msg
    
    # Sort images by filename to ensure correct order
    granule_images.sort()
    
    # Get first and last image for this granule
    start_img = granule_images[0]
    end_img = granule_images[-1]
    
    # Build Python command with required and optional arguments
    cmd = [
        "python",
        swir_nc_processor_script,
        "--img-dir", swir_img_dir,
        "--meta-dir", swir_meta_dir,
        "--output_dir", nc_output_dir,
        "--start-img", start_img,
        "--end-img", end_img,
        "--level", nc_processing_level,
        "--ah2_imu_ncfile", ah2_imu_nc,
    ]
    
    # Add ER2 navigation if requested
    if include_er2_nav and er2_imu_log and er2_imu_xml:
        cmd.extend(["--include-er2-nav"])
        cmd.extend(["--er2-infile", er2_imu_log])
        cmd.extend(["--er2-xmlfile", er2_imu_xml])

    return cmd, granule_images, None


def process_granules_multiprocessing(granules, img_catalog, meta_catalog, swir_img_dir, swir_meta_dir, 
                                     nc_output_dir, nc_processing_level, swir_nc_processor_script,
                                     ah2_imu_nc, include_er2_nav, er2_imu_log, er2_imu_xml, 
                                     num_processes=4):
    """
    Process granules using multiprocessing pool
    
    :param granules: list of (start_time, end_time) tuples
    :param img_catalog: dict mapping CFC capture IDs to image info
    :param meta_catalog: dict mapping metadata filenames to their info
    :param swir_img_dir: SWIR image directory
    :param swir_meta_dir: SWIR metadata directory
    :param nc_output_dir: NetCDF4 output directory
    :param nc_processing_level: processing level (e.g., "1A")
    :param swir_nc_processor_script: path to SWIR_NC_Processor.py
    :param ah2_imu_nc: path to AH2 IMU NetCDF file
    :param include_er2_nav: whether to include ER2 navigation
    :param er2_imu_log: path to ER2 IMU log file
    :param er2_imu_xml: path to ER2 IMU XML file
    :param num_processes: number of parallel processes
    """
    
    print("\n" + "="*70)
    print("PROCESSING GRANULES WITH MULTIPROCESSING")
    print("="*70)
    print(f"Total granules to process: {len(granules)}")
    print(f"Number of processes: {num_processes}\n")
    
    # Create partial function with constant arguments
    process_func = partial(
        process_granule,
        img_catalog=img_catalog,
        meta_catalog=meta_catalog,
        swir_img_dir=swir_img_dir,
        swir_meta_dir=swir_meta_dir,
        nc_output_dir=nc_output_dir,
        nc_processing_level=nc_processing_level,
        swir_nc_processor_script=swir_nc_processor_script,
        ah2_imu_nc=ah2_imu_nc,
        include_er2_nav=include_er2_nav,
        er2_imu_log=er2_imu_log if include_er2_nav else None,
        er2_imu_xml=er2_imu_xml if include_er2_nav else None,
    )
    
    # Prepare arguments for map
    args = []
    for idx, granule_entry in enumerate(granules, 1):
        granule_start, granule_end, granule_label = parse_granule_entry(granule_entry, idx)
        args.append((idx, granule_start, granule_end, granule_label))
    
    # Create output directory
    os.makedirs(nc_output_dir, exist_ok=True)
    
    # Process granules in parallel
    successful = 0
    failed = 0
    
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(process_func, args)
    
    # Print results
    print("\n" + "="*70)
    print("PROCESSING RESULTS")
    print("="*70)
    
    for granule_idx, success, message in results:
        if success:
            successful += 1
            print(f"✓ {message}")
        else:
            failed += 1
            print(f"✗ {message}")
    
    print(f"\nTotal completed: {successful}/{len(granules)}")
    if failed > 0:
        print(f"Total failed: {failed}")
    
    return successful, failed


def generate_slurm_job(granules, img_catalog, meta_catalog, swir_img_dir, swir_meta_dir,
                       nc_output_dir, nc_processing_level, swir_nc_processor_script,
                       ah2_imu_nc, include_er2_nav, er2_imu_log, er2_imu_xml,
                       shell_output_dir="./runs", shell_output_fname="l1agen_swir_batch.sh",
                       shell_log_output_dir="./runs/log-%u", disable_autorun=False):
    """
    Process granules using SLURM (one independent srun task per granule)
    
    :param granules: list of (start_time, end_time) tuples
    :param img_catalog: dict mapping CFC capture IDs to image info
    :param meta_catalog: dict mapping metadata filenames to their info
    :param swir_img_dir: SWIR image directory
    :param swir_meta_dir: SWIR metadata directory
    :param nc_output_dir: NetCDF4 output directory
    :param nc_processing_level: processing level (e.g., "1A")
    :param swir_nc_processor_script: path to SWIR_NC_Processor.py
    :param ah2_imu_nc: path to AH2 IMU NetCDF file
    :param include_er2_nav: whether to include ER2 navigation
    :param er2_imu_log: path to ER2 IMU log file
    :param er2_imu_xml: path to ER2 XML file
    """

    print("\n" + "="*70)
    print("GENERATING SLURM JOB")
    print("="*70)

    controller_dir = os.path.dirname(os.path.abspath(__file__))
    shell_output_dir, shell_log_output_dir = resolve_shell_paths(
        shell_output_dir, shell_log_output_dir, controller_dir
    )

    os.makedirs(shell_output_dir, exist_ok=True)
    os.makedirs(shell_log_output_dir, exist_ok=True)
    os.makedirs(nc_output_dir, exist_ok=True)

    command_lines, skipped = collect_granule_command_lines(
        granules, img_catalog, meta_catalog, swir_img_dir, swir_meta_dir,
        nc_output_dir, nc_processing_level, swir_nc_processor_script,
        ah2_imu_nc, include_er2_nav, er2_imu_log, er2_imu_xml
    )

    if not command_lines:
        print("ERROR: No valid granules to submit to Slurm")
        return 0, len(granules)

    script_path = os.path.join(shell_output_dir, shell_output_fname)
    write_slurm_script(script_path, command_lines, shell_log_output_dir)

    if disable_autorun:
        print(f"Slurm script generated at: {script_path}")
        print(f"Valid granules to run: {len(command_lines)} | skipped: {skipped}")
        print(f"Submit manually with: sbatch {script_path}")
    else:
        submit = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
        if submit.returncode != 0:
            print(f"ERROR: sbatch submission failed: {submit.stderr.strip()}")
            return 0, len(granules)
        print(f"Submitted Slurm job: {submit.stdout.strip()}")
        print(f"Granules queued: {len(command_lines)} | skipped: {skipped}")

    return len(command_lines), skipped


def generate_gnu_parallel(granules, img_catalog, meta_catalog, swir_img_dir, swir_meta_dir,
                          nc_output_dir, nc_processing_level, swir_nc_processor_script,
                          ah2_imu_nc, include_er2_nav, er2_imu_log, er2_imu_xml,
                          shell_output_dir="./runs", shell_output_fname="l1agen_swir_batch.sh",
                          shell_log_output_dir="./runs/log-%u", disable_autorun=False,
                          num_processes=4):
    """
    Process granules using GNU parallel
    
    :param granules: list of (start_time, end_time) tuples
    :param img_catalog: dict mapping CFC capture IDs to image info
    :param meta_catalog: dict mapping metadata filenames to their info
    :param swir_img_dir: SWIR image directory
    :param swir_meta_dir: SWIR metadata directory
    :param nc_output_dir: NetCDF4 output directory
    :param nc_processing_level: processing level (e.g., "1A")
    :param swir_nc_processor_script: path to SWIR_NC_Processor.py
    :param ah2_imu_nc: path to AH2 IMU NetCDF file
    :param include_er2_nav: whether to include ER2 navigation
    :param er2_imu_log: path to ER2 IMU log file
    :param er2_imu_xml: path to ER2 XML file
    """

    print("\n" + "="*70)
    print("GENERATING GNU PARALLEL JOB")
    print("="*70)

    controller_dir = os.path.dirname(os.path.abspath(__file__))
    shell_output_dir, shell_log_output_dir = resolve_shell_paths(
        shell_output_dir, shell_log_output_dir, controller_dir
    )

    os.makedirs(shell_output_dir, exist_ok=True)
    os.makedirs(shell_log_output_dir, exist_ok=True)
    os.makedirs(nc_output_dir, exist_ok=True)

    command_lines, skipped = collect_granule_command_lines(
        granules, img_catalog, meta_catalog, swir_img_dir, swir_meta_dir,
        nc_output_dir, nc_processing_level, swir_nc_processor_script,
        ah2_imu_nc, include_er2_nav, er2_imu_log, er2_imu_xml
    )

    if not command_lines:
        print("ERROR: No valid granules to run with GNU parallel")
        return 0, len(granules)

    script_path = os.path.join(shell_output_dir, shell_output_fname)
    write_gnu_parallel_script(script_path, command_lines, shell_log_output_dir, num_processes)

    if disable_autorun:
        print(f"GNU parallel script generated at: {script_path}")
        print(f"Valid granules to run: {len(command_lines)} | skipped: {skipped}")
        print(f"Run manually with: {script_path}")
    else:
        called_subprocess = subprocess.Popen([script_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        print(f"GNU parallel launched in background with PID: {called_subprocess.pid}")
        print(f"Granules queued: {len(command_lines)} | skipped: {skipped}")

    return len(command_lines), skipped


def main():
    """
    Main controller workflow:
    1. Parse command line arguments
    2. Enumerate SWIR images (build CFC capture ID catalog)
    3. Build metadata catalog (extract timestamps and ACQ IDs from metadata files)
    4. Load granules from pre-processed granule file
    5. Match metadata files to granule time ranges
    6. Match images to granules via ACQ ID/CFC capture ID
    7. Process granules with multiprocessing
    """
    
    # Parse command line arguments
    args = parse_input_args()
    
    # Validate ER2 navigation arguments
    if args.include_er2_nav and (not args.er2_infile or not args.er2_xmlfile):
        print("ERROR: --er2-infile and --er2-xmlfile are required when --include-er2-nav is set")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("SWIR GRANULE CONTROLLER")
    print("="*70)
    print(f"Image Directory: {args.img_dir}")
    
    if args.granule_file:
        print(f"Granule Source: Pre-processed file ({args.granule_file})")
    else:
        print(f"Granule Source: Automatic calculation")
        print(f"Granule Duration: {args.granule_duration} seconds")
        print(f"Time Tolerance: {args.time_tolerance} seconds")
    
    print(f"Processing Level: L{args.level}")
    print(f"Parallel Processes: {args.num_processes}")
    print(f"Execution Backend: {args.execution_backend}")
    
    # Step 1: Enumerate images
    img_catalog = enumerate_swir_images(args.img_dir)
    
    # Step 2: Build metadata catalog
    meta_catalog = build_metadata_catalog(args.meta_dir)
    
    # Step 3: Create granules from either file or automatic calculation
    if args.granule_file:
        # Use pre-processed granule file
        granules = load_granules_from_file(args.granule_file)
    else:
        # Use automatic flight leg identification and granule division
        # Note: For automatic mode, we still need time-based image info
        print("ERROR: Automatic granule calculation not yet updated for metadata-based matching")
        print("Please use --granule-file option")
        sys.exit(1)
    
    if not granules:
        print("ERROR: No granules created. Check input data or granule file.")
        sys.exit(1)
    
    # Step 4: Process granules using selected backend
    backend = args.execution_backend
    if backend == "auto":
        try:
            subprocess.run(["sinfo"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            backend = "slurm"
        except (FileNotFoundError, subprocess.CalledProcessError):
            backend = "multiprocessing"

    if backend == "multiprocessing":
        successful, failed = process_granules_multiprocessing(
            granules,
            img_catalog,
            meta_catalog,
            swir_img_dir=args.img_dir,
            swir_meta_dir=args.meta_dir,
            nc_output_dir=args.output_dir,
            nc_processing_level=args.level,
            swir_nc_processor_script=args.nc_processor_script,
            ah2_imu_nc=args.ah2_imu_ncfile,
            include_er2_nav=args.include_er2_nav,
            er2_imu_log=args.er2_infile,
            er2_imu_xml=args.er2_xmlfile,
            num_processes=args.num_processes
        )
    elif backend == "slurm":
        successful, failed = generate_slurm_job(
            granules,
            img_catalog,
            meta_catalog,
            swir_img_dir=args.img_dir,
            swir_meta_dir=args.meta_dir,
            nc_output_dir=args.output_dir,
            nc_processing_level=args.level,
            swir_nc_processor_script=args.nc_processor_script,
            ah2_imu_nc=args.ah2_imu_ncfile,
            include_er2_nav=args.include_er2_nav,
            er2_imu_log=args.er2_infile,
            er2_imu_xml=args.er2_xmlfile,
            shell_output_dir=args.shell_output_dir,
            shell_output_fname=args.shell_output_fname,
            shell_log_output_dir=args.shell_log_output_dir,
            disable_autorun=args.disable_autorun
        )
    else:
        successful, failed = generate_gnu_parallel(
            granules,
            img_catalog,
            meta_catalog,
            swir_img_dir=args.img_dir,
            swir_meta_dir=args.meta_dir,
            nc_output_dir=args.output_dir,
            nc_processing_level=args.level,
            swir_nc_processor_script=args.nc_processor_script,
            ah2_imu_nc=args.ah2_imu_ncfile,
            include_er2_nav=args.include_er2_nav,
            er2_imu_log=args.er2_infile,
            er2_imu_xml=args.er2_xmlfile,
            shell_output_dir=args.shell_output_dir,
            shell_output_fname=args.shell_output_fname,
            shell_log_output_dir=args.shell_log_output_dir,
            disable_autorun=args.disable_autorun,
            num_processes=args.num_processes
        )
    
    print("\n" + "="*70)
    print("CONTROLLER WORKFLOW COMPLETE")
    print("="*70)
    print(f"Total granules processed: {successful + failed}/{len(granules)}")
    print(f"Output directory: {args.output_dir}")
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
