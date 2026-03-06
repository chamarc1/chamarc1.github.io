"""
Test script for load_granules_from_file function

This script tests the granule loading function and diagnoses potential issues
with granule time ranges, metadata matching, and coverage gaps.

Usage:
    python test_load_granules.py --granule-file <path> --meta-dir <path>
"""

import os
import sys
import datetime as dt
import argparse

# Add parent directory to path to import the controller module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from l1agen_SWIR_Granule_Controller import (
    load_granules_from_file,
    build_metadata_catalog
)

# Import time offset functions from the matcher module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'modules_l1a_SWIR'))
from l1agen_SWIR_Image_Data_TEC_Metadata_Matcher import (
    CalculateTimeOffset,
    ApplyTimeOffsetToAcqIDs,
    ConvertTimestampToUTC
)


def analyze_granule_coverage(granules, meta_catalog):
    """
    Analyze granule time coverage against metadata catalog
    
    :param granules: list of (start_time, end_time) tuples
    :param meta_catalog: dict of metadata files with timestamps
    """
    print("\n" + "="*70)
    print("GRANULE COVERAGE ANALYSIS (Using Adjusted ACQ Times)")
    print("="*70)
    
    # Get metadata time range from adjusted ACQ times
    if meta_catalog:
        all_start_times = [info['time_range'][0] for info in meta_catalog.values()]
        all_end_times = [info['time_range'][1] for info in meta_catalog.values()]
        meta_start = min(all_start_times)
        meta_end = max(all_end_times)
        print(f"Metadata time range (adjusted ACQ times):")
        print(f"  Start: {meta_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  End:   {meta_end.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("WARNING: No metadata times found")
        meta_start = None
        meta_end = None
    
    # Analyze each granule
    print(f"\nAnalyzing {len(granules)} granules:")
    print("-" * 70)
    
    issues_found = []
    
    for idx, (gran_start, gran_end) in enumerate(granules, 1):
        duration = (gran_end - gran_start).total_seconds()
        
        # Check if granule is within metadata time range
        if meta_start and meta_end:
            before_meta = gran_end < meta_start
            after_meta = gran_start > meta_end
            partial_before = gran_start < meta_start <= gran_end
            partial_after = gran_start <= meta_end < gran_end
            
            status = "✓ OK"
            issue = None
            
            if before_meta:
                status = "✗ BEFORE metadata range"
                issue = f"Granule {idx} is completely before metadata time range"
            elif after_meta:
                status = "✗ AFTER metadata range"
                issue = f"Granule {idx} is completely after metadata time range"
            elif partial_before:
                status = "⚠ PARTIAL (starts before metadata)"
                issue = f"Granule {idx} starts before metadata coverage"
            elif partial_after:
                status = "⚠ PARTIAL (ends after metadata)"
                issue = f"Granule {idx} ends after metadata coverage"
            
            if issue:
                issues_found.append(issue)
        else:
            status = "? No metadata to compare"
        
        # Count metadata files and ACQ IDs in this granule's time range
        meta_count = 0
        acq_count = 0
        matching_files = []
        
        for meta_file, meta_info in meta_catalog.items():
            # Check if metadata time range overlaps with granule
            meta_start_time, meta_end_time = meta_info['time_range']
            
            if meta_start_time <= gran_end and meta_end_time >= gran_start:
                meta_count += 1
                # Count how many ACQ times fall within granule
                acqs_in_range = sum(1 for t in meta_info['acq_times'] if gran_start <= t <= gran_end)
                acq_count += acqs_in_range
                matching_files.append((meta_file, meta_start_time, meta_end_time, acqs_in_range))
        
        # Print granule info
        print(f"Granule {idx:2d}: {gran_start.strftime('%H:%M:%S')} - {gran_end.strftime('%H:%M:%S')} "
              f"({duration:5.0f}s) | {meta_count:3d} meta files | {acq_count:4d} ACQ IDs | {status}")
        
        # Show matching metadata files for problematic granules
        if meta_count == 0 or status.startswith("✗") or status.startswith("⚠"):
            if matching_files:
                for meta_file, m_start, m_end, acqs in matching_files[:3]:  # Show first 3
                    print(f"         -> {meta_file} ({m_start.strftime('%H:%M:%S')}-{m_end.strftime('%H:%M:%S')}, {acqs} ACQs)")
                if len(matching_files) > 3:
                    print(f"         -> ... and {len(matching_files) - 3} more")
            else:
                print(f"         -> No matching metadata files in time range")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total granules: {len(granules)}")
    print(f"Issues found: {len(issues_found)}")
    
    if issues_found:
        print("\nDetailed Issues:")
        for issue in issues_found:
            print(f"  - {issue}")
    else:
        print("\nNo issues found - all granules have metadata coverage")
    
    return issues_found


def analyze_acq_id_coverage(granules, meta_dir):
    """
    Analyze granule coverage using ACQ IDs from metadata files with time offset correction
    
    :param granules: list of (start_time, end_time) tuples
    :param meta_dir: directory containing metadata files
    """
    print("\n" + "="*70)
    print("ACQ ID TIME OFFSET ANALYSIS")
    print("="*70)
    print("Checking if ACQ IDs (after time offset) match granule ranges...")
    
    # Get all metadata files
    meta_files = sorted([f for f in os.listdir(meta_dir) if f.endswith('.meta')])
    
    if not meta_files:
        print("ERROR: No metadata files found")
        return
    
    # Analyze last few metadata files in detail (most likely to affect Granule 38)
    print(f"\nAnalyzing last 10 metadata files in detail:")
    print("-" * 70)
    
    for meta_file in meta_files[-10:]:
        meta_path = os.path.join(meta_dir, meta_file)
        
        # Read ACQ IDs from file
        acq_ids = []
        try:
            with open(meta_path, 'r') as f:
                for line in f:
                    if 'ACQ' in line:
                        line = line.replace(':', ';').replace(' ', '').rstrip('\n').split(';')
                        acq_id = int(line[1])
                        acq_ids.append(acq_id)
        except Exception as e:
            print(f"  ERROR reading {meta_file}: {e}")
            continue
        
        if not acq_ids:
            continue
        
        # Calculate time offset
        first_acq_id = acq_ids[0]
        last_acq_id = acq_ids[-1]
        offset_ms = CalculateTimeOffset(first_acq_id, meta_file)
        
        if offset_ms is None:
            print(f"\n{meta_file}:")
            print(f"  WARNING: Could not calculate time offset")
            continue
        
        # Apply offset to ACQ IDs
        import numpy as np
        adjusted_acq_ids = ApplyTimeOffsetToAcqIDs(np.array(acq_ids), offset_ms)
        
        # Convert to datetime for comparison
        first_adjusted_time_str = ConvertTimestampToUTC(adjusted_acq_ids[0])
        last_adjusted_time_str = ConvertTimestampToUTC(adjusted_acq_ids[-1])
        
        if first_adjusted_time_str and last_adjusted_time_str:
            first_adjusted_time = dt.datetime.strptime(first_adjusted_time_str, '%Y-%m-%d %H:%M:%S')
            last_adjusted_time = dt.datetime.strptime(last_adjusted_time_str, '%Y-%m-%d %H:%M:%S')
            
            print(f"\n{meta_file}:")
            print(f"  ACQ ID count: {len(acq_ids)}")
            print(f"  Time offset: {offset_ms/1000:.1f}s ({offset_ms}ms)")
            print(f"  First ACQ ID (raw): {first_acq_id} -> After offset: {adjusted_acq_ids[0]}")
            print(f"  Last ACQ ID (raw):  {last_acq_id} -> After offset: {adjusted_acq_ids[-1]}")
            print(f"  Time range (adjusted): {first_adjusted_time.strftime('%H:%M:%S')} to {last_adjusted_time.strftime('%H:%M:%S')}")
            
            # Check which granules this metadata file could match
            matching_granules = []
            for idx, (gran_start, gran_end) in enumerate(granules, 1):
                # Check if any adjusted ACQ ID falls within this granule
                if first_adjusted_time <= gran_end and last_adjusted_time >= gran_start:
                    matching_granules.append(idx)
            
            if matching_granules:
                print(f"  Matches granules: {matching_granules}")
            else:
                print(f"  Matches granules: None (outside all granule ranges)")
    
    print("\n" + "="*70)
    print("CHECKING GRANULE 38 SPECIFICALLY")
    print("="*70)
    
    # Find Granule 38
    if len(granules) >= 38:
        gran38_start, gran38_end = granules[37]  # 0-indexed
        print(f"Granule 38 time range: {gran38_start.strftime('%Y-%m-%d %H:%M:%S')} to {gran38_end.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check all metadata files to see if any adjusted ACQ IDs fall in Granule 38
        matching_meta_files = []
        
        for meta_file in meta_files:
            meta_path = os.path.join(meta_dir, meta_file)
            
            # Read ACQ IDs
            acq_ids = []
            try:
                with open(meta_path, 'r') as f:
                    for line in f:
                        if 'ACQ' in line:
                            line = line.replace(':', ';').replace(' ', '').rstrip('\n').split(';')
                            acq_id = int(line[1])
                            acq_ids.append(acq_id)
            except:
                continue
            
            if not acq_ids:
                continue
            
            # Calculate and apply offset
            offset_ms = CalculateTimeOffset(acq_ids[0], meta_file)
            if offset_ms is None:
                continue
            
            import numpy as np
            adjusted_acq_ids = ApplyTimeOffsetToAcqIDs(np.array(acq_ids), offset_ms)
            
            # Check if any adjusted ACQ ID falls in Granule 38
            for adj_acq_id in adjusted_acq_ids:
                time_str = ConvertTimestampToUTC(adj_acq_id)
                if time_str:
                    acq_time = dt.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                    if gran38_start <= acq_time <= gran38_end:
                        matching_meta_files.append((meta_file, acq_time))
                        break  # Found a match, no need to check other ACQ IDs
        
        if matching_meta_files:
            print(f"\nFound {len(matching_meta_files)} metadata files with ACQ IDs in Granule 38 range:")
            for meta_file, acq_time in matching_meta_files:
                print(f"  - {meta_file} has ACQ IDs @ {acq_time.strftime('%H:%M:%S')}")
            print("\n⚠ WARNING: The controller is using metadata FILENAME timestamps instead of")
            print("           adjusted ACQ ID timestamps for granule matching!")
            print("           This causes metadata files to be missed.")
        else:
            print("\nNo metadata files have ACQ IDs in Granule 38 range (even after offset correction)")
            print("Granule 38 may be legitimately without data.")
    
    return


def analyze_granule_gaps(granules):
    """
    Analyze gaps between consecutive granules
    
    :param granules: list of (start_time, end_time) tuples
    """
    print("\n" + "="*70)
    print("GRANULE GAP ANALYSIS")
    print("="*70)
    
    if len(granules) < 2:
        print("Need at least 2 granules for gap analysis")
        return
    
    gaps_found = []
    overlaps_found = []
    
    for idx in range(len(granules) - 1):
        curr_start, curr_end = granules[idx]
        next_start, next_end = granules[idx + 1]
        
        gap = (next_start - curr_end).total_seconds()
        
        if gap > 1:  # Gap larger than 1 second
            gaps_found.append((idx + 1, idx + 2, gap))
            print(f"Gap between Granule {idx + 1} and {idx + 2}: {gap:.1f}s")
            print(f"  Granule {idx + 1} ends:   {curr_end.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Granule {idx + 2} starts: {next_start.strftime('%Y-%m-%d %H:%M:%S')}")
        elif gap < -1:  # Overlap larger than 1 second
            overlaps_found.append((idx + 1, idx + 2, abs(gap)))
            print(f"Overlap between Granule {idx + 1} and {idx + 2}: {abs(gap):.1f}s")
            print(f"  Granule {idx + 1} ends:   {curr_end.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Granule {idx + 2} starts: {next_start.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\nTotal gaps > 1s: {len(gaps_found)}")
    print(f"Total overlaps > 1s: {len(overlaps_found)}")


def show_granule_details(granules, num_to_show=5):
    """
    Show detailed information about first and last granules
    
    :param granules: list of (start_time, end_time) tuples
    :param num_to_show: number of granules to show from start and end
    """
    print("\n" + "="*70)
    print("GRANULE DETAILS")
    print("="*70)
    
    print(f"\nFirst {num_to_show} granules:")
    for idx, (gran_start, gran_end) in enumerate(granules[:num_to_show], 1):
        duration = (gran_end - gran_start).total_seconds()
        print(f"  {idx:2d}. {gran_start.strftime('%Y-%m-%d %H:%M:%S')} -> "
              f"{gran_end.strftime('%H:%M:%S')} ({duration:.0f}s)")
    
    if len(granules) > num_to_show * 2:
        print(f"\n  ... ({len(granules) - num_to_show * 2} granules omitted) ...")
    
    print(f"\nLast {num_to_show} granules:")
    for idx, (gran_start, gran_end) in enumerate(granules[-num_to_show:], len(granules) - num_to_show + 1):
        duration = (gran_end - gran_start).total_seconds()
        print(f"  {idx:2d}. {gran_start.strftime('%Y-%m-%d %H:%M:%S')} -> "
              f"{gran_end.strftime('%H:%M:%S')} ({duration:.0f}s)")


def test_granule_file_format(granule_file):
    """
    Test the granule file format and show raw content
    
    :param granule_file: path to granule file
    """
    print("\n" + "="*70)
    print("GRANULE FILE FORMAT TEST")
    print("="*70)
    print(f"File: {granule_file}")
    
    if not os.path.exists(granule_file):
        print(f"ERROR: File not found")
        return
    
    try:
        with open(granule_file, 'r') as f:
            lines = f.readlines()
        
        print(f"\nTotal lines: {len(lines)}")
        print(f"\nFirst 10 lines:")
        print("-" * 70)
        for idx, line in enumerate(lines[:10], 1):
            print(f"{idx:3d}: {line.rstrip()}")
        
        print(f"\nLast 10 lines:")
        print("-" * 70)
        for idx, line in enumerate(lines[-10:], len(lines) - 9):
            print(f"{idx:3d}: {line.rstrip()}")
        
        # Count comment lines, empty lines, data lines
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        empty_lines = sum(1 for line in lines if not line.strip())
        data_lines = len(lines) - comment_lines - empty_lines
        
        print(f"\nLine breakdown:")
        print(f"  Comment lines: {comment_lines}")
        print(f"  Empty lines: {empty_lines}")
        print(f"  Data lines: {data_lines}")
        
    except Exception as e:
        print(f"ERROR reading file: {e}")


def main():
    parser = argparse.ArgumentParser(description='Test load_granules_from_file function')
    parser.add_argument('--granule-file', type=str, required=True, 
                       help='Path to granule file (GRANULE.DATE.txt)')
    parser.add_argument('--meta-dir', type=str, default=None,
                       help='Path to metadata directory (for coverage analysis)')
    parser.add_argument('--show-format', action='store_true',
                       help='Show raw granule file format')
    
    args = parser.parse_args()
    
    print("="*70)
    print("TEST: load_granules_from_file")
    print("="*70)
    
    # Test 1: Show file format (if requested)
    if args.show_format:
        test_granule_file_format(args.granule_file)
    
    # Test 2: Load granules
    print("\n" + "="*70)
    print("TEST: Loading Granules")
    print("="*70)
    
    try:
        granules = load_granules_from_file(args.granule_file)
        print(f"✓ Successfully loaded {len(granules)} granules")
    except Exception as e:
        print(f"✗ ERROR loading granules: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test 3: Show granule details
    if granules:
        show_granule_details(granules)
    
    # Test 4: Analyze gaps
    if granules:
        analyze_granule_gaps(granules)
    
    # Test 5: Compare with metadata catalog (if provided)
    if args.meta_dir:
        print("\n" + "="*70)
        print("Loading metadata catalog for comparison...")
        print("="*70)
        
        try:
            meta_catalog = build_metadata_catalog(args.meta_dir)
            if meta_catalog:
                issues = analyze_granule_coverage(granules, meta_catalog)
                
                # NEW: Analyze ACQ IDs with time offset correction
                analyze_acq_id_coverage(granules, args.meta_dir)
                
                if issues:
                    print("\n" + "="*70)
                    print("RECOMMENDATION")
                    print("="*70)
                    print("Issues were found with granule/metadata time coverage.")
                    print("Consider:")
                    print("  1. Check if metadata files are missing for certain time ranges")
                    print("  2. Verify granule file time ranges are correct")
                    print("  3. Adjust time tolerance if needed")
                    print("  4. Use ACQ ID times (with offset) instead of filename times")
                    sys.exit(1)
                else:
                    print("\n✓ All granules have proper metadata coverage")
                    sys.exit(0)
            else:
                print("WARNING: No metadata files found")
        except Exception as e:
            print(f"ERROR loading metadata catalog: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n" + "="*70)
        print("NOTE: Use --meta-dir to compare granules with metadata coverage")
        print("="*70)


if __name__ == "__main__":
    main()
