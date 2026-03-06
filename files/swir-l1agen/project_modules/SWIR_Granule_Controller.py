"""
SWIR Granule Controller for AirHARP2 SWIR L1A Processing

This controller orchestrates SWIR image processing into 5-minute L1A granules.
It handles:
1. Flight leg detection from timestamp gaps
2. 5-minute granule division (300-second chunks)
3. Metadata matching (CFC capture IDs with TEC/integration time)
4. Parallel job generation and submission

Key Features:
- Automatic leg segmentation based on time gaps
- Standard 5-minute granule structure (matches polarimeter products)
- Robust error handling for incomplete segments
- Parallel execution support
- Detailed logging and progress tracking

Usage:
    python SWIR_Granule_Controller.py

Configuration:
    Edit the GLOBALS section below with your flight day paths and settings.

__name__ =      SWIR_Granule.Controller.py
__author__ =    "Charlemagne Marc"
__copyright__ = "Copyright 2025, ESI SWIR Project"
__credits__ =   ["Charlemagne Marc"]
__version__ =   "1.5.5"
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
from pathlib import Path
from multiprocessing import Pool
from functools import partial

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
    parser.add_argument('--nc-processor-script', type=str, required=True, help='Full path to SWIR_NC_Processor.py script')
    
    #-- optional arguments
    parser.add_argument('--level', type=str, default='1A', help='Processing level (default: 1A)')
    parser.add_argument('--time-tolerance', type=int, default=5, help='Gap threshold in seconds to identify flight leg breaks (default: 5)')
    parser.add_argument('--granule-duration', type=int, default=300, help='Granule duration in seconds (default: 300 = 5 minutes)')
    parser.add_argument('--num-processes', type=int, default=4, help='Number of parallel processes (default: 4)')
    parser.add_argument('--include-er2-nav', action='store_true', help='Include ER2 navigation data')
    parser.add_argument('--er2-infile', type=str, default=None, help='Full path to ER2 log file (required if --include-er2-nav is set)')
    parser.add_argument('--er2-xmlfile', type=str, default=None, help='Full path to ER2 XML file (required if --include-er2-nav is set)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    
    #-- return arguments
    return parser.parse_args()



def get_timestamp_from_filename(fname):
    """
    Extract timestamp from SWIR image filename
    
    Handles multiple formats:
    1. cfc_capture_<unix_ms>.tiff - Unix timestamp in milliseconds
    2. YYYYMMDD_HHMMSS format - Standard date-time format
    
    :param fname: string filename
    :return: datetime object, or None if cannot parse
    """
    try:
        # Remove file extension
        base_name = fname.rsplit('.', 1)[0]
        
        # Try format: cfc_capture_<unix_timestamp_ms>
        if base_name.startswith("cfc_capture_"):
            try:
                unix_ms = int(base_name.split("_")[-1])
                # Convert milliseconds to seconds
                unix_sec = unix_ms / 1000.0
                timestamp = dt.datetime.fromtimestamp(unix_sec)
                return timestamp
            except (ValueError, IndexError):
                pass
        
        # Try parsing standard format: YYYYMMDD_HHMMSS
        if "_" in base_name:
            parts = base_name.split("_")
            if len(parts) >= 2:
                time_part = parts[0] + parts[1]
                timestamp = dt.datetime.strptime(time_part, "%Y%m%d%H%M%S")
                return timestamp
        
        # If no underscore, try extracting first 14 chars
        if len(base_name) >= 14:
            timestamp = dt.datetime.strptime(base_name[:14], "%Y%m%d%H%M%S")
            return timestamp
    except (ValueError, IndexError, OSError):
        pass
    
    return None


def enumerate_swir_images(img_dir, start_img=None, end_img=None):
    """
    Enumerate all SWIR image files in chronological order
    
    :param img_dir:   string path to SWIR image directory
    :param start_img: optional start image filename
    :param end_img:   optional end image filename
    :return: dict with lists of filenames, paths, timestamps, and frame IDs
    """
    
    print("\n" + "="*70)
    print("ENUMERATING SWIR IMAGES")
    print("="*70)
    
    # List and sort all TIFF files
    all_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.tif', '.tiff'))]
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
    
    # Extract timestamps
    img_finfo = {
        "fnames": [],
        "fpaths": [],
        "ftimes": [],
        "frame_IDs": []
    }
    
    for idx, fname in enumerate(all_files):
        timestamp = get_timestamp_from_filename(fname)
        
        if timestamp is None:
            print(f"WARNING: Could not parse timestamp from {fname}, skipping")
            continue
        
        img_finfo["fnames"].append(fname)
        img_finfo["fpaths"].append(os.path.join(img_dir, fname))
        img_finfo["ftimes"].append(timestamp)
        img_finfo["frame_IDs"].append(idx)
    
    print(f"Total SWIR images enumerated: {len(img_finfo['fnames'])}")
    if img_finfo["fnames"]:
        print(f"  First image: {img_finfo['fnames'][0]} @ {img_finfo['ftimes'][0]}")
        print(f"  Last image:  {img_finfo['fnames'][-1]} @ {img_finfo['ftimes'][-1]}")
    
    return img_finfo


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


def process_granule(granule_idx, granule_start, granule_end, img_finfo, swir_img_dir, swir_meta_dir, 
                    nc_output_dir, nc_processing_level, swir_nc_processor_script, ah2_imu_nc,
                    include_er2_nav=False, er2_imu_log=None, er2_imu_xml=None):
    """
    Process a single granule using SWIR_NC_Processor
    
    :param granule_idx: index of granule (for progress tracking)
    :param granule_start: start time of granule
    :param granule_end: end time of granule
    :param img_finfo: image file info dict with fnames and ftimes
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
    
    # Find images that fall within this granule time range
    granule_images = []
    for fname, ftime in zip(img_finfo["fnames"], img_finfo["ftimes"]):
        if granule_start <= ftime <= granule_end:
            granule_images.append((fname, ftime))
    
    if not granule_images:
        return (granule_idx, False, f"Granule {granule_idx}: No images found in time range")
    
    # Get first and last image for this granule
    start_img = granule_images[0][0]
    end_img = granule_images[-1][0]
    
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
    
    try:
        # Debug: print command for first few granules
        if granule_idx <= 2:
            print(f"[DEBUG] Granule {granule_idx} command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=3600)
        return (granule_idx, True, f"Granule {granule_idx} ({len(granule_images)} images) processed successfully")
    except subprocess.TimeoutExpired:
        return (granule_idx, False, f"Granule {granule_idx}: Processing timeout (>1 hour)")
    except subprocess.CalledProcessError as e:
        return (granule_idx, False, f"Granule {granule_idx}: {e.stderr[:200]}")


def process_granules_multiprocessing(granules, img_finfo, swir_img_dir, swir_meta_dir, 
                                     nc_output_dir, nc_processing_level, swir_nc_processor_script,
                                     ah2_imu_nc, include_er2_nav, er2_imu_log, er2_imu_xml, 
                                     num_processes=4):
    """
    Process granules using multiprocessing pool
    
    :param granules: list of (start_time, end_time) tuples
    :param img_finfo: image file info dict with fnames and ftimes
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
        img_finfo=img_finfo,
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
    args = [
        (idx, granule_start, granule_end)
        for idx, (granule_start, granule_end) in enumerate(granules, 1)
    ]
    
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


def main():
    """
    Main controller workflow:
    1. Parse command line arguments
    2. Enumerate SWIR images
    3. Identify flight legs
    4. Divide into 5-minute granules
    5. Process granules with multiprocessing
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
    print(f"Granule Duration: {args.granule_duration} seconds")
    print(f"Time Tolerance: {args.time_tolerance} seconds")
    print(f"Processing Level: L{args.level}")
    print(f"Parallel Processes: {args.num_processes}")
    
    # Step 1: Enumerate images
    img_finfo = enumerate_swir_images(args.img_dir)
    
    # Step 2: Identify flight legs
    flight_legs = identify_flight_legs(img_finfo, args.time_tolerance)
    
    # Step 3: Divide into granules
    granules = divide_into_granules(flight_legs, args.granule_duration)
    
    if not granules:
        print("ERROR: No granules created. Check input data.")
        sys.exit(1)
    
    # Step 4: Process granules with multiprocessing
    successful, failed = process_granules_multiprocessing(
        granules, 
        img_finfo, 
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
    
    print("\n" + "="*70)
    print("CONTROLLER WORKFLOW COMPLETE")
    print("="*70)
    print(f"Total granules processed: {successful + failed}/{len(granules)}")
    print(f"Output directory: {args.output_dir}")
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
