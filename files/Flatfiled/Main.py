"""
SWIR Flatfield Processing - Main Entry Point

This script processes Short-Wave Infrared (SWIR) images to create flatfield corrections.
A flatfield correction removes spatial variations in sensor response, making images
more uniform and improving data quality.

The script now automatically processes ALL filter wheel positions (pos1-pos4) and creates
combined visualizations showing all positions together.

Usage:
    python Main.py [--num_sigma <value>]
    
Example:
    python Main.py --num_sigma 1.0

__author__ =    "Charlemagne Marc"
__copyright__ = "Copyright 2025, ESI SWIR Project"
__version__ =   "2.0.0"
__email__ =     "chamrc1@oumbc.edu"
"""

#----------------------------------------------------------------------------
#-- IMPORT STATEMENTS
#----------------------------------------------------------------------------
from project_modules.CompositeProcessor import CompositeProcessor
from project_modules.FlatfieldProcessor import FlatfieldProcessor, plot_composite
from project_modules.Constants import directory_dict, crossTrack_dict, crossTrackDark_dict, alongTrack_dict, alongTrackDark_dict
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#----------------------------------------------------------------------------
#-- MAIN FUNCTION
#----------------------------------------------------------------------------
def main():
    """
    Main function that processes SWIR images for flatfield correction.
    
    This function:
    1. Parses command line arguments
    2. Creates flatfield processors for all filter positions
    3. Generates combined plots for all positions
    4. Creates both 2D composite and 3D envelope visualizations
    """
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Create flatfield corrections for SWIR images.\n\n"
                   "This tool processes multiple SWIR images for ALL filter positions\n"
                   "and creates combined visualizations showing all positions together.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "--num_sigma",
        type=float,
        default=1.0,
        help="Standard deviation for Gaussian smoothing (default: 1.0)\n"
             "Higher values = more smoothing\n"
             "Set to 0 to disable smoothing"
    )
    
    # Parse the command line arguments
    args = parser.parse_args()
    
    # Process all positions
    process_all_positions(args.num_sigma)

def process_all_positions(num_sigma):
    """
    Process all filter wheel positions and create combined visualizations.
    
    Args:
        num_sigma (float): Standard deviation for Gaussian smoothing
    """
    print("Starting SWIR flatfield processing for ALL filter positions...")
    print(f"Using smoothing sigma: {num_sigma}")
    
    # Create output directory for PowerPoint plots
    presentation_dir = r"/data/home/cmarc/SWIR_Projects/Flatfield/Images/PowerPoint_Plots"
    corrected_dir = r"/data/home/cmarc/SWIR_Projects/Flatfield/Images/Corrected_Images"
    os.makedirs(presentation_dir, exist_ok=True)
    os.makedirs(corrected_dir, exist_ok=True)

    try:
        # Create the flatfield processor (now handles all positions)
        pos1_processor = FlatfieldProcessor("pos1")
        pos2_processor = FlatfieldProcessor("pos2")
        pos3_processor = FlatfieldProcessor("pos3")
        pos4_processor = FlatfieldProcessor("pos4")
        
        # Store processors for summary plots
        processors = {
            "pos1": pos1_processor,
            "pos2": pos2_processor, 
            "pos3": pos3_processor,
            "pos4": pos4_processor
        }
        
        # Process all positions and create combined plots and Generate detailed analysis for each position
        print("\n" + "="*70)
        print("Generating detailed flatfield analysis...")
        print("="*70)
        
        # Process each position with enhanced plot saving
        position_results = {}
        
        print("\n" + "="*70)
        print("Filter Position 1:")
        print("="*70)
        position_results["pos1"] = pos1_processor.process_position_with_plots("pos1", num_sigma, corrected_dir)
        # Apply pos1 flatfield map to raw image
        raw_image_path_pos1 = "/data/archive/ESI/AirHARP2/PACE-PAX_Campaign/PACE-PAX_FlightData_20240927/SWIR/cfc_capture_1719352872456.tiff"
        if os.path.exists(raw_image_path_pos1):
            print(f"Applying pos1 flatfield correction to: {raw_image_path_pos1}")
            # Load the raw image first
            raw_image = np.array(Image.open(raw_image_path_pos1))
            corrected_path_pos1 = os.path.join(corrected_dir, "pos1_corrected_image.tiff")
            pos1_processor.apply_flatfield_correction(raw_image, save_corrected_path=corrected_path_pos1)
        else:
            print(f"Warning: Raw image not found at {raw_image_path_pos1}")
        
        print("\n" + "="*70)
        print("Filter Position 2:")
        print("="*70)
        position_results["pos2"] = pos2_processor.process_position_with_plots("pos2", num_sigma, corrected_dir)
        # Apply pos2 flatfield map to raw image
        raw_image_path_pos2 = "/data/archive/ESI/AirHARP2/PACE-PAX_Campaign/PACE-PAX_FlightData_20240927/SWIR/cfc_capture_1719352875732.tiff"
        if os.path.exists(raw_image_path_pos2):
            print(f"Applying pos2 flatfield correction to: {raw_image_path_pos2}")
            # Load the raw image first
            raw_image = np.array(Image.open(raw_image_path_pos2))
            corrected_path_pos2 = os.path.join(corrected_dir, "pos2_corrected_image.tiff")
            pos2_processor.apply_flatfield_correction(raw_image, save_corrected_path=corrected_path_pos2)
        else:
            print(f"Warning: Raw image not found at {raw_image_path_pos2}")
        
        print("\n" + "="*70)
        print("Filter Position 3:")
        print("="*70)
        position_results["pos3"] = pos3_processor.process_position_with_plots("pos3", num_sigma, corrected_dir)
        # Apply pos3 flatfield map to raw image
        raw_image_path_pos3 = "/data/archive/ESI/AirHARP2/PACE-PAX_Campaign/PACE-PAX_FlightData_20240927/SWIR/cfc_capture_1719352878987.tiff"
        if os.path.exists(raw_image_path_pos3):
            print(f"Applying pos3 flatfield correction to: {raw_image_path_pos3}")
            # Load the raw image first
            raw_image = np.array(Image.open(raw_image_path_pos3))
            corrected_path_pos3 = os.path.join(corrected_dir, "pos3_corrected_image.tiff")
            pos3_processor.apply_flatfield_correction(raw_image, save_corrected_path=corrected_path_pos3)
        else:
            print(f"Warning: Raw image not found at {raw_image_path_pos3}")
        
        print("\n" + "="*70)
        print("Filter Position 4:")
        print("="*70)
        position_results["pos4"] = pos4_processor.process_position_with_plots("pos4", num_sigma, corrected_dir)
        # Apply pos4 flatfield map to raw image
        raw_image_path_pos4 = "/data/archive/ESI/AirHARP2/PACE-PAX_Campaign/PACE-PAX_FlightData_20240927/SWIR/cfc_capture_1719352882312.tiff"
        if os.path.exists(raw_image_path_pos4):
            print(f"Applying pos4 flatfield correction to: {raw_image_path_pos4}")
            # Load the raw image first
            raw_image = np.array(Image.open(raw_image_path_pos4))
            corrected_path_pos4 = os.path.join(corrected_dir, "pos4_corrected_image.tiff")
            pos4_processor.apply_flatfield_correction(raw_image, save_corrected_path=corrected_path_pos4)
        else:
            print(f"Warning: Raw image not found at {raw_image_path_pos4}")
        
        # # Generate summary plots for PowerPoint
        # print("\n" + "="*70)
        # print("Generating PowerPoint Summary Plots...")
        # print("="*70)
        # FlatfieldProcessor.generate_summary_plots(position_results, presentation_dir)
        
        print("\n" + "="*70)
        print("PROCESSING COMPLETE!")
        print("="*70)
        print(f"PowerPoint plots saved to: {presentation_dir}")
        print("Generated visualizations:")
        print("• Individual position profile analysis plots")
        print("• Individual position 3D envelope plots")
        print("• Individual position 2D flatfield maps")
        print("• Flatfield corrections applied to raw images")
        print("• Before/after correction comparison plots")
        print("• Multi-position summary plots")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        print("Please check your input parameters and try again.")
        raise

if __name__ == "__main__":
    main()
