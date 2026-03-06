"""
__name__ = SWIR_Image_Data_TEC_Metadata_Matcher.py
__author__ = Charlemagne Marc
__co_author__ =  cjescobar
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
import numpy as np
import argparse
import datetime as dt
import re


#----------------------------------------------------------------------------
#-- PATHS/DIRECTORIES OF INTEREST
#----------------------------------------------------------------------------
SWIR_IMG_DIR       = r"/stor/z101/Data/PACE-PAX/Shared/PACE-PAX_FlightData_20240904/SWIR"  # parent path to SWIR images
SWIR_METADATA_DIR  = r"/stor/z101/Data/PACE-PAX/Shared/PACE-PAX_FlightData_20240904/SWIR"  # parent path to metadata files
# SWIR Metadata stor/z101/Data/PACE-PAX/Shared
MATCHED_OUTPUT_DIR = r"/data/home/cmarc/SWIR_Projects/SWIR_L1A_Generation/metadata_matcher"  # path to outputed csv containing matched data

# SWIR_IMG_DIR       = r"/data/home/cjescobar/Projects/AirHARP2/SWIR/raw_data/2025_SWIR_Flatfield/"  # parent path to SWIR images
# SWIR_METADATA_DIR  = r"/data/home/cjescobar/Projects/AirHARP2/SWIR/raw_data/2025_SWIR_Flatfield/METADATA"  # parent path to metadata files
# MATCHED_OUTPUT_DIR = r"/data/home/cjescobar/Projects/AirHARP2/SWIR/gits/AH2_SWIR_metadata_matcher/"  # path to outputed csv containing matched data


#----------------------------------------------------------------------------
#-- FUNCTION DEFINITIONS
#----------------------------------------------------------------------------
def ExtractTimeFromFilename(fname):
    """
    Extract UNIX timestamp (milliseconds) from SWIR image filename.
    
    Expected filename format: cfc_capture_{UNIX_TIMESTAMP_MS}.tif or .tiff
    
    :param fname: string filename
    :return: UNIX timestamp in milliseconds (int), or None if extraction fails
    """
    try:
        # Extract the number between underscore and file extension
        # Split by underscore to get ["cfc", "capture", "timestamp.tif"]
        parts = fname.split("_")
        if len(parts) >= 3:
            # Get the last part (e.g., "1719355104855.tif" or "1719355104855.tiff")
            last_part = parts[-1]
            
            # Remove extension by splitting on '.' and taking the first part
            # This handles both .tif and .tiff (and any other extension)
            timestamp_str = last_part.split(".")[0]
            
            # Convert to integer
            timestamp_ms = int(timestamp_str)
            return timestamp_ms
    except (ValueError, IndexError, AttributeError):
        pass
    return None


def ConvertTimestampToUTC(timestamp_ms):
    """
    Convert UNIX timestamp (milliseconds) to UTC datetime string.
    
    :param timestamp_ms: UNIX timestamp in milliseconds (int or float)
    :return: ISO format UTC datetime string (YYYY-MM-DD HH:MM:SS), or None if conversion fails
    """
    try:
        timestamp_sec = timestamp_ms / 1000.0
        ftime_utc = dt.datetime.fromtimestamp(timestamp_sec, tz=dt.timezone.utc)
        return ftime_utc.strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, OSError, TypeError):
        return None


def ExtractMetadataFileTimestamp(meta_fname):
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


def CalculateTimeOffset(first_acq_id_ms, meta_fname):
    """
    Calculate time offset between metadata filename timestamp and first ACQ ID.
    
    This offset can be applied to adjust acquisition IDs from the metadata file to match
    the actual flight time indicated by the metadata filename.
    The offset = (metadata_filename_time converted to milliseconds) - first_acq_id_ms
    
    :param first_acq_id_ms: First ACQ ID from metadata file (UNIX timestamp in milliseconds)
    :param meta_fname: Metadata filename to extract reference time from
    :return: Time offset in milliseconds (int), or None if calculation fails
    """
    try:
        meta_time = ExtractMetadataFileTimestamp(meta_fname)
        if meta_time is None:
            return None
        
        # Convert metadata filename time to UNIX timestamp in milliseconds
        meta_time_ms = int(meta_time.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
        
        # Calculate offset: we want to shift FROM the ACQ ID time TO the metadata filename time
        offset_ms = int(meta_time_ms - first_acq_id_ms)
        return offset_ms
    except (ValueError, TypeError, AttributeError):
        return None


def ApplyTimeOffsetToAcqIDs(acq_ids, offset_ms):
    """
    Apply time offset to all acquisition IDs from a metadata file.
    
    This adjusts ACQ IDs based on the calculated offset from the metadata filename.
    
    :param acq_ids: numpy array of acquisition IDs (UNIX timestamps in milliseconds)
    :param offset_ms: Time offset in milliseconds to apply
    :return: numpy array of adjusted acquisition IDs
    """
    try:
        adjusted_ids = np.array([int(cid + offset_ms) for cid in acq_ids], dtype=object)
        return adjusted_ids
    except (ValueError, TypeError):
        return acq_ids


def LocateFiles(parent_dir, ftype):
    fdict = {
             "fnames": [], 
             "fpaths": []
            }

    for root, dirs, files in os.walk(parent_dir):
        for file in files:
            if (ftype.lower() in file.lower()) and (os.path.getsize(os.path.join(root, file)) > 0):
                fdict["fnames"].append(file)
                fdict["fpaths"].append(os.path.join(root, file))
            # end if
        # end for
    # end for
    
    fdict["fnames"], fdict["fpaths"] = np.asarray(fdict["fnames"], dtype = np.object_), np.asarray(fdict["fpaths"], dtype = np.object_); 

    return fdict


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Match SWIR image data with TEC metadata')
    parser.add_argument('--img-dir', type=str, default=SWIR_IMG_DIR,
                        help='Directory containing SWIR image files (.tif)')
    parser.add_argument('--meta-dir', type=str, default=SWIR_METADATA_DIR,
                        help='Directory containing SWIR metadata files (.meta)')
    parser.add_argument('--output-dir', type=str, default=MATCHED_OUTPUT_DIR,
                        help='Output directory for matched CSV file')
    
    args = parser.parse_args()
    
    # Use command line arguments or defaults
    swir_img_dir = args.img_dir
    swir_metadata_dir = args.meta_dir
    matched_output_dir = args.output_dir
    
    SWIR_img_fdict = LocateFiles(swir_img_dir, ".tif")
    
    # split image fnames to isolate cfc capture ID
    parsed_fnames = []
    for i in range(SWIR_img_fdict["fnames"].shape[0]):
        parsed_fname = SWIR_img_fdict["fnames"][i].split("/")[-1].replace(".", "_").split("_")
        parsed_fnames.append(parsed_fname)
    # end for
    
    # Convert to proper 2D numpy array
    SWIR_img_fdict["fnames"] = np.array(parsed_fnames, dtype=object)


    #-- open SWIR metadata files
    SWIR_metadata_fdict = LocateFiles(swir_metadata_dir, ".meta");
    SWIR_metadata_fdict["fcontent"] = {
                                       "cap_ID"  : [], 
                                       "TEC"     : [],
                                       "filt"    : [],
                                       "int_time": []
                                      }
    
    for fpath in SWIR_metadata_fdict["fpaths"]:
        with open(fpath, "r") as f:
            for line in f:
                if "ACQ" in line:
                    line = line.replace(":",";").replace(" ","").rstrip("\n").split(";")
                    SWIR_metadata_fdict["fcontent"]["cap_ID"].append(int(line[1]))
                    SWIR_metadata_fdict["fcontent"]["TEC"].append(int(line[2]))
                    SWIR_metadata_fdict["fcontent"]["filt"].append(int(line[3]))
                    SWIR_metadata_fdict["fcontent"]["int_time"].append(np.float64(line[-1].replace("ms","")))
                # end if
            # end for
            f.close()
        # end with
    # end for
    
    for key, val in SWIR_metadata_fdict["fcontent"].items():
        SWIR_metadata_fdict["fcontent"][key] = np.asarray(SWIR_metadata_fdict["fcontent"][key], dtype = np.object_)  # specify dtype = np.object_ to ensure that datatype of each item is retained during conversion to NumPy array
    # end for
    
    
    #-- sort and match
    if SWIR_img_fdict["fnames"].shape[0] <= SWIR_metadata_fdict["fcontent"]["cap_ID"].shape[0]:
        
        #-- sort images according to capture ID
        # Extract capture IDs from parsed filenames (assuming capture ID is at index 2)
        img_capture_ids = [int(fname[2]) for fname in SWIR_img_fdict["fnames"]]
        sort_ind = np.argsort(img_capture_ids)
        
        for key, val in SWIR_img_fdict.items():
            SWIR_img_fdict[key] = val[sort_ind]
        # end for
    
    
        #-- match metadata to images
        match_ind = []
        for fname in SWIR_img_fdict["fnames"]:
            # print(f"int(fname[2]) = {int(fname[2])}");
            # print(f'from SWIR_metadata_fdict: {SWIR_metadata_fdict["fcontent"]["cap_ID"][np.argwhere(SWIR_metadata_fdict["fcontent"]["cap_ID"] == int(fname[2])).flatten()[0]]}')
            match_ind.append(np.argwhere(SWIR_metadata_fdict["fcontent"]["cap_ID"] == int(fname[2])).flatten()[0])
        # end for
    
        for key, val in SWIR_metadata_fdict["fcontent"].items():
            SWIR_metadata_fdict["fcontent"][key] = val[match_ind]
        # end for
    
    elif SWIR_img_fdict["fnames"].shape[0] > SWIR_metadata_fdict["fcontent"]["cap_ID"].shape[0]:
        
        #-- sort metadata according to capture ID
        sort_ind = np.argsort(SWIR_metadata_fdict["fcontent"]["cap_ID"])
    
        for key, val in SWIR_metadata_fdict["fcontent"].items():
            SWIR_metadata_fdict["fcontent"][key] = val[sort_ind]
        # end for


        #-- match images to metadata
        match_ind = []
        for cap_ID in SWIR_metadata_fdict["fcontent"]["cap_ID"]:
            # Find the index of the image with matching capture ID
            img_capture_ids = [int(fname[2]) for fname in SWIR_img_fdict["fnames"]]
            match_idx = np.argwhere(np.array(img_capture_ids) == cap_ID).flatten()[0]
            match_ind.append(match_idx)
        # end for
    
        for key, val in SWIR_img_fdict.items():
            SWIR_img_fdict[key] = val[match_ind]
        # end for

    # Create output directory if it doesn't exist
    os.makedirs(matched_output_dir, exist_ok=True)

    np.savetxt(os.path.join(matched_output_dir, "202502_SWIR_Flatfield_Matched_Metadata_v2.csv"), np.transpose([SWIR_img_fdict["fpaths"], ["_".join(fname) for fname in SWIR_img_fdict["fnames"]], SWIR_metadata_fdict["fcontent"]["cap_ID"], SWIR_metadata_fdict["fcontent"]["TEC"], SWIR_metadata_fdict["fcontent"]["filt"], SWIR_metadata_fdict["fcontent"]["int_time"]]), fmt = ['%s','%s','%d','%d','%d','%.1f'], delimiter = ',', header = 'FILEPATH,FILENAME,CFC_CAPTURE_ID,TEC_READING(CELCIUS),FILTER_POSITION,INTEGRATION_TIME(MS)', comments = '')
    


if __name__ == "__main__":
    main()
# end if