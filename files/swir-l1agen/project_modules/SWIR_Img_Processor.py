"""
__name__ =      SWIR_Img_Processor.py
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
import datetime as dt
import numpy as np
from PIL import Image

# Handle imports that work both when run directly and when imported as module
try:
    from .SWIR_Image_Data_TEC_Metadata_Matcher import LocateFiles
except ImportError:
    # If relative import fails, try absolute import
    from SWIR_Image_Data_TEC_Metadata_Matcher import LocateFiles

#----------------------------------------------------------------------------
#-- GLOBALS
#----------------------------------------------------------------------------
SWIR_FILE_SIZE_MIN = 1000000  # minimum SWIR TIFF file size in bytes to indicate capture is not corrupted
BIT_SHIFT = 2**14            # 14-bit sensor inversion reference (per requirement)

#----------------------------------------------------------------------------
#-- CLASS DEFINITIONS
#----------------------------------------------------------------------------
class SWIR_img_processor():
    """
    SWIR image processor class for AirHARP2 SWIR sensor data.
    Processes SWIR images and metadata files, matching them by CFC capture ID.
    """
    
    #-- constructor
    def __init__(self, img_dir, meta_dir, start_img, end_img, processing_level, verbose=False):
        """
        Initialize the SWIR_img_processor with necessary parameters
        
        :param img_dir:          string defining root directory containing SWIR images of interest
        :param meta_dir:         string defining directory containing SWIR metadata files  
        :param start_img:        string defining file to start processing with
        :param end_img:          string defining file to end processing with
        :param processing_level: string defining processing level
        :param verbose:          logical True or False for printing script progress to terminal
        """
        
        #-- initialize instance attributes
        self.img_dir          = img_dir
        self.meta_dir         = meta_dir
        self.start_img        = start_img
        self.end_img          = end_img
        self.processing_level = processing_level
        self.verbose          = verbose
        self.img_finfo        = self.enum_imgs()
        self.meta_finfo       = self.enum_meta()
        
        # Match images with metadata
        self.matched_data = self.match_imgs_with_metadata()
    
    
    def enum_imgs(self):
        """
        Enumerate all SWIR image files in the img_dir
        
        :return img_finfo: Dictionary containing valid SWIR images' filenames, full paths, and CFC capture IDs
        """
        
        if self.verbose:
            print(f"\nGetting SWIR image filenames and paths...")
        
        # Use LocateFiles to find all TIFF files
        img_finfo = LocateFiles(self.img_dir, ".tif")
        
        # Filter for files between start_img and end_img if specified
        if self.start_img and self.end_img:
            # Sort files to ensure proper ordering
            sorted_indices = np.argsort(img_finfo["fnames"])
            img_finfo["fnames"] = img_finfo["fnames"][sorted_indices]
            img_finfo["fpaths"] = img_finfo["fpaths"][sorted_indices]
            
            # Find start and end indices
            start_idx = None
            end_idx = None
            for i, fname in enumerate(img_finfo["fnames"]):
                if self.start_img in fname:
                    start_idx = i
                if self.end_img in fname:
                    end_idx = i
            
            if start_idx is not None and end_idx is not None:
                img_finfo["fnames"] = img_finfo["fnames"][start_idx:end_idx+1]
                img_finfo["fpaths"] = img_finfo["fpaths"][start_idx:end_idx+1]
        
        # Filter out corrupted files (too small)
        valid_fnames = []
        valid_fpaths = []
        
        for fname, fpath in zip(img_finfo["fnames"], img_finfo["fpaths"]):
            if os.path.getsize(fpath) >= SWIR_FILE_SIZE_MIN:
                valid_fnames.append(fname)
                valid_fpaths.append(fpath)
        
        img_finfo["fnames"] = np.array(valid_fnames, dtype=object)
        img_finfo["fpaths"] = np.array(valid_fpaths, dtype=object)
        
        # Extract CFC capture IDs from filenames
        img_finfo["cfc_ids"] = []
        for fname in img_finfo["fnames"]:
            # Extract CFC capture ID from filename (e.g., cfc_capture_1719352629638.tiff -> 1719352629638)
            try:
                cfc_id = int(fname.split("_")[-1].split(".")[0])
                img_finfo["cfc_ids"].append(cfc_id)
            except (ValueError, IndexError):
                if self.verbose:
                    print(f"Warning: Could not extract CFC ID from {fname}")
                img_finfo["cfc_ids"].append(None)
        
        img_finfo["cfc_ids"] = np.array(img_finfo["cfc_ids"], dtype=object)
        
        if self.verbose:
            print(f"Found {len(img_finfo['fnames'])} valid SWIR images")
        
        return img_finfo
    
    
    def enum_meta(self):
        """
        Enumerate all SWIR metadata files in the meta_dir
        
        :return meta_finfo: Dictionary containing metadata filenames, paths, and parsed content
        """
        
        if self.verbose:
            print(f"Getting SWIR metadata filenames and paths...")
        
        # Use LocateFiles to find all .meta files
        meta_finfo = LocateFiles(self.meta_dir, ".meta")
        
        # Parse metadata file contents
        meta_finfo["fcontent"] = {
            "cap_ID": [],
            "TEC": [],
            "filt": [],
            "int_time": []
        }
        
        for fpath in meta_finfo["fpaths"]:
            try:
                with open(fpath, "r") as f:
                    for line in f:
                        if "ACQ" in line:
                            # Parse line format: ACQ:capture_id:tec_temp:filter_pos:integration_time_ms
                            line = line.replace(":", ";").replace(" ", "").rstrip("\n").split(";")
                            meta_finfo["fcontent"]["cap_ID"].append(int(line[1]))
                            meta_finfo["fcontent"]["TEC"].append(int(line[2]))
                            meta_finfo["fcontent"]["filt"].append(int(line[3]))
                            meta_finfo["fcontent"]["int_time"].append(float(line[-1].replace("ms", "")))
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Error reading metadata file {fpath}: {e}")
        
        # Convert to numpy arrays
        for key, val in meta_finfo["fcontent"].items():
            meta_finfo["fcontent"][key] = np.array(val, dtype=object)
        
        if self.verbose:
            print(f"Found {len(meta_finfo['fcontent']['cap_ID'])} metadata entries")
        
        return meta_finfo
    
    
    def match_imgs_with_metadata(self):
        """
        Match SWIR images with their corresponding metadata based on CFC capture ID
        
        :return matched_data: Dictionary containing matched image and metadata information
        """
        
        if self.verbose:
            print(f"Matching images with metadata...")
        
        matched_data = {
            "img_fnames": [],
            "img_fpaths": [],
            "cfc_ids": [],
            "tec_temps": [],
            "filter_positions": [],
            "integration_times": []
        }
        
        # Match images with metadata by CFC capture ID
        for i, cfc_id in enumerate(self.img_finfo["cfc_ids"]):
            if cfc_id is None:
                continue
                
            # Find matching metadata entry
            meta_indices = np.where(self.meta_finfo["fcontent"]["cap_ID"] == cfc_id)[0]
            
            if len(meta_indices) > 0:
                meta_idx = meta_indices[0]  # Take first match if multiple found
                
                matched_data["img_fnames"].append(self.img_finfo["fnames"][i])
                matched_data["img_fpaths"].append(self.img_finfo["fpaths"][i])
                matched_data["cfc_ids"].append(cfc_id)
                matched_data["tec_temps"].append(self.meta_finfo["fcontent"]["TEC"][meta_idx])
                matched_data["filter_positions"].append(self.meta_finfo["fcontent"]["filt"][meta_idx])
                matched_data["integration_times"].append(self.meta_finfo["fcontent"]["int_time"][meta_idx])
            elif self.verbose:
                print(f"Warning: No metadata found for image {self.img_finfo['fnames'][i]} (CFC ID: {cfc_id})")
        
        # Convert to numpy arrays
        for key, val in matched_data.items():
            matched_data[key] = np.array(val, dtype=object)
        
        if self.verbose:
            print(f"Successfully matched {len(matched_data['img_fnames'])} images with metadata")
        
        return matched_data
    
    
    def get_images_by_filter_position(self, filter_pos):
        """
        Get all images for a specific filter position
        
        :param filter_pos: integer filter position (0-4)
        :return: Dictionary containing images and metadata for the specified filter position
        """
        
        indices = np.where(self.matched_data["filter_positions"] == filter_pos)[0]
        
        filter_data = {}
        for key, val in self.matched_data.items():
            filter_data[key] = val[indices]
        
        return filter_data

    def get_images_chronological(self):
        """
        Get all matched images and metadata in acquisition order (chronological).

        Sorts by `cfc_ids` (Unix timestamp in milliseconds). Entries with `None`
        `cfc_ids` are excluded to avoid undefined ordering.

        :return: Dictionary containing arrays ordered chronologically with keys:
                 'img_fnames', 'img_fpaths', 'cfc_ids', 'tec_temps',
                 'filter_positions', 'integration_times'.
        """

        cfc_ids = self.matched_data.get("cfc_ids", np.array([], dtype=object))
        valid_indices = [i for i, cid in enumerate(cfc_ids) if cid is not None]
        order = sorted(valid_indices, key=lambda i: int(cfc_ids[i]))

        chrono = {}
        for key, val in self.matched_data.items():
            chrono[key] = val[order]
        return chrono

    def iterate_chronological(self):
        """
        Iterate per-frame records in acquisition order for convenient consumption
        by the NetCDF writer. Yields dicts with canonical keys.

        Yielded dict keys:
        - 'img_fname'
        - 'img_fpath'
        - 'cfc_id'
        - 'tec_temp'
        - 'filter_pos'
        - 'integration_time_ms'
        """

        chrono = self.get_images_chronological()
        n = len(chrono.get("img_fnames", []))
        for i in range(n):
            yield {
                "img_fname": chrono["img_fnames"][i],
                "img_fpath": chrono["img_fpaths"][i],
                "cfc_id": int(chrono["cfc_ids"][i]),
                "tec_temp": float(chrono["tec_temps"][i]),
                "filter_pos": int(chrono["filter_positions"][i]),
                "integration_time_ms": float(chrono["integration_times"][i])
            }
    
    
    def load_img(self, img_fpath):
        """
        Load a single SWIR image
        
        :param img_fpath: string path to image file
        :return: 2D numpy array containing image data, or None if loading fails
        """
        
        try:
            # Load image using PIL and convert to numpy array
            img_obj = Image.open(img_fpath)
            img = np.asarray(img_obj)
            img_obj.close()

            # Apply 14-bit inversion processing: processed = 2**14 - raw
            # Cast to a wider integer to avoid unsigned underflow, then back to uint16
            processed = (BIT_SHIFT - img.astype(np.int32)).astype(np.uint16)
            return processed
        except Exception as e:
            if self.verbose:
                print(f"Error loading image {img_fpath}: {e}")
            return None
    
    
    def extract_time_from_cfc_id(self, cfc_id):
        """
        Extract timestamp from CFC capture ID (which is typically a Unix timestamp in milliseconds)
        
        :param cfc_id: CFC capture ID (Unix timestamp in milliseconds)
        :return: datetime object or None if conversion fails
        """
        
        try:
            # CFC capture ID is typically Unix timestamp in milliseconds
            timestamp_sec = cfc_id / 1000.0
            frame_time = dt.datetime.fromtimestamp(timestamp_sec, tz=dt.timezone.utc)
            return frame_time
        except Exception as e:
            if self.verbose:
                print(f"Error extracting time from CFC ID {cfc_id}: {e}")
            return None
    
    
    def get_summary(self):
        """
        Get a summary of the processed images and metadata
        
        :return: Dictionary containing summary statistics
        """
        
        summary = {
            "total_images": len(self.matched_data["img_fnames"]),
            "filter_position_counts": {},
            "tec_temp_range": None,
            "integration_time_range": None
        }
        
        if len(self.matched_data["filter_positions"]) > 0:
            # Count images per filter position
            for pos in range(5):  # SWIR has 5 filter positions (0-4)
                count = np.sum(self.matched_data["filter_positions"] == pos)
                summary["filter_position_counts"][pos] = count
            
            # TEC temperature range
            if len(self.matched_data["tec_temps"]) > 0:
                summary["tec_temp_range"] = (
                    np.min(self.matched_data["tec_temps"]),
                    np.max(self.matched_data["tec_temps"])
                )
            
            # Integration time range
            if len(self.matched_data["integration_times"]) > 0:
                summary["integration_time_range"] = (
                    np.min(self.matched_data["integration_times"]),
                    np.max(self.matched_data["integration_times"])
                )
        
        return summary


#----------------------------------------------------------------------------
#-- FUNCTION DEFINITIONS
#----------------------------------------------------------------------------
def parse_test_args():
    """
    Define input arguments for testing the SWIR_img_processor
    
    :return args: argparse.Namespace class instance, which holds arguments and their values
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Test SWIR image processor')
    
    # Required arguments
    parser.add_argument('--img-dir', type=str, required=True, 
                       help='Directory containing SWIR image files (.tiff)')
    parser.add_argument('--meta-dir', type=str, required=True,
                       help='Directory containing SWIR metadata files (.meta)')
    
    # Optional arguments
    parser.add_argument('--start-img', type=str, default=None,
                       help='First image filename to process (optional)')
    parser.add_argument('--end-img', type=str, default=None,
                       help='Last image filename to process (optional)')
    parser.add_argument('--processing-level', type=str, default='1A',
                       help='Processing level (default: 1A)')
    parser.add_argument('--test-load', action='store_true',
                       help='Test loading a few images to verify functionality')
    parser.add_argument('--filter-pos', type=int, choices=[0, 1, 2, 3, 4],
                       help='Show details for specific filter position')
    
    return parser.parse_args()


def main():
    """
    Main function to test the SWIR_img_processor class
    """
    print("=" * 60)
    print("SWIR Image Processor Test")
    print("=" * 60)
    
    # Parse command line arguments
    args = parse_test_args()
    
    try:
        # Initialize the SWIR image processor
        print(f"\nInitializing SWIR_img_processor...")
        print(f"  Image directory: {args.img_dir}")
        print(f"  Metadata directory: {args.meta_dir}")
        if args.start_img:
            print(f"  Start image: {args.start_img}")
        if args.end_img:
            print(f"  End image: {args.end_img}")
        
        swir_processor = SWIR_img_processor(
            img_dir=args.img_dir,
            meta_dir=args.meta_dir,
            start_img=args.start_img,
            end_img=args.end_img,
            processing_level=args.processing_level,
            verbose=True
        )
        
        # Get and display summary
        print("\n" + "=" * 40)
        print("PROCESSING SUMMARY")
        print("=" * 40)
        
        summary = swir_processor.get_summary()
        print(f"Total matched images: {summary['total_images']}")
        
        print(f"\nFilter position distribution:")
        for pos, count in summary['filter_position_counts'].items():
            filter_desc = ["dark", "1.57μm", "1.55μm", "1.38μm", "open"][pos]
            print(f"  Position {pos} ({filter_desc}): {count} images")
        
        if summary['tec_temp_range']:
            print(f"\nTEC temperature range: {summary['tec_temp_range'][0]}°C to {summary['tec_temp_range'][1]}°C")
        
        if summary['integration_time_range']:
            print(f"Integration time range: {summary['integration_time_range'][0]}ms to {summary['integration_time_range'][1]}ms")
        
        # Show details for specific filter position if requested
        if args.filter_pos is not None:
            print(f"\n" + "=" * 40)
            print(f"FILTER POSITION {args.filter_pos} DETAILS")
            print("=" * 40)
            
            filter_data = swir_processor.get_images_by_filter_position(args.filter_pos)
            print(f"Images for filter position {args.filter_pos}: {len(filter_data['img_fnames'])}")
            
            if len(filter_data['img_fnames']) > 0:
                print("\nFirst 5 images:")
                for i in range(min(5, len(filter_data['img_fnames']))):
                    fname = filter_data['img_fnames'][i]
                    cfc_id = filter_data['cfc_ids'][i]
                    tec_temp = filter_data['tec_temps'][i]
                    int_time = filter_data['integration_times'][i]
                    
                    # Extract timestamp from CFC ID
                    frame_time = swir_processor.extract_time_from_cfc_id(cfc_id)
                    time_str = frame_time.strftime('%Y-%m-%d %H:%M:%S UTC') if frame_time else 'Unknown'
                    
                    print(f"  {i+1:2d}. {fname}")
                    print(f"      CFC ID: {cfc_id}, Time: {time_str}")
                    print(f"      TEC: {tec_temp}°C, Integration: {int_time}ms")
        
        # Test image loading if requested
        if args.test_load and summary['total_images'] > 0:
            print(f"\n" + "=" * 40)
            print("IMAGE LOADING TEST")
            print("=" * 40)
            
            # Try to load the first image
            test_img_path = swir_processor.matched_data['img_fpaths'][0]
            test_img_name = swir_processor.matched_data['img_fnames'][0]
            
            print(f"Testing image loading with: {test_img_name}")
            img_data = swir_processor.load_img(test_img_path)
            
            if img_data is not None:
                print(f"✓ Successfully loaded image")
                print(f"  Image shape: {img_data.shape}")
                print(f"  Data type: {img_data.dtype}")
                print(f"  Value range: {img_data.min()} to {img_data.max()}")
                print(f"  File size: {os.path.getsize(test_img_path) / 1024 / 1024:.2f} MB")
            else:
                print("✗ Failed to load image")
        
        print(f"\n" + "=" * 60)
        print("SWIR Image Processor Test Complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())