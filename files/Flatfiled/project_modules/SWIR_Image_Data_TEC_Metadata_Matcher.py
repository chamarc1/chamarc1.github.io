"""

__author__ = cjescobar
__updated_by__ = "Charlemagne Marc"
__copyright__ = "Copyright 2024, ESI SWIR Project"

"""

#----------------------------------------------------------------------------
#-- IMPORT STATEMENTS
#----------------------------------------------------------------------------
import os;
import numpy as np;


#----------------------------------------------------------------------------
#-- PATHS/DIRECTORIES OF INTEREST
#----------------------------------------------------------------------------
SWIR_IMG_DIR       = r"/data/archive/ESI/AirHARP2/PACE-PAX_Campaign/PACE-PAX_FlightData_20240927/SWIR";  # parent path to SWIR images
SWIR_METADATA_DIR  = r"/data/archive/ESI/AirHARP2/PACE-PAX_Campaign/PACE-PAX_FlightData_20240927/SWIR/MetadataGroups";  # parent path to metadata files
MATCHED_OUTPUT_DIR = r"/data/home/cmarc/SWIR_Projects/Flatfield/metadata_matcher";  # path to outputed csv containing matched data

# SWIR_IMG_DIR       = r"/data/home/cjescobar/Projects/AirHARP2/SWIR/raw_data/2025_SWIR_Flatfield/";  # parent path to SWIR images
# SWIR_METADATA_DIR  = r"/data/home/cjescobar/Projects/AirHARP2/SWIR/raw_data/2025_SWIR_Flatfield/METADATA";  # parent path to metadata files
# MATCHED_OUTPUT_DIR = r"/data/home/cjescobar/Projects/AirHARP2/SWIR/gits/AH2_SWIR_metadata_matcher/";  # path to outputed csv containing matched data


#----------------------------------------------------------------------------
#-- FUNCTION DEFINITIONS
#----------------------------------------------------------------------------
def LocateFiles(parent_dir, ftype):
    fdict = {
             "fnames": [], 
             "fpaths": []
            };

    for root, dirs, files in os.walk(parent_dir):
        for file in files:
            if (ftype.lower() in file.lower()) and (os.path.getsize(os.path.join(root, file)) > 0):
                fdict["fnames"].append(file);
                fdict["fpaths"].append(os.path.join(root, file));
            # end if
        # end for
    # end for
    
    fdict["fnames"], fdict["fpaths"] = np.asarray(fdict["fnames"], dtype = np.object_), np.asarray(fdict["fpaths"], dtype = np.object_); 

    return fdict;


def main():
    SWIR_img_fdict = LocateFiles(SWIR_IMG_DIR, ".tif");
    
    # split image fnames to isolate cfc capture ID
    parsed_fnames = []
    for i in range(SWIR_img_fdict["fnames"].shape[0]):
        parsed_fname = SWIR_img_fdict["fnames"][i].split("/")[-1].replace(".", "_").split("_");
        parsed_fnames.append(parsed_fname);
    # end for
    
    # Convert to proper 2D numpy array
    SWIR_img_fdict["fnames"] = np.array(parsed_fnames, dtype=object)


    #-- open SWIR metadata files
    SWIR_metadata_fdict = LocateFiles(SWIR_METADATA_DIR, ".meta");
    SWIR_metadata_fdict["fcontent"] = {
                                       "cap_ID"  : [], 
                                       "TEC"     : [],
                                       "filt"    : [],
                                       "int_time": []
                                      };
    
    for fpath in SWIR_metadata_fdict["fpaths"]:
        with open(fpath, "r") as f:
            for line in f:
                if "ACQ" in line:
                    line = line.replace(":",";").replace(" ","").rstrip("\n").split(";");
                    SWIR_metadata_fdict["fcontent"]["cap_ID"].append(int(line[1]));
                    SWIR_metadata_fdict["fcontent"]["TEC"].append(int(line[2]));
                    SWIR_metadata_fdict["fcontent"]["filt"].append(int(line[3]));
                    SWIR_metadata_fdict["fcontent"]["int_time"].append(np.float64(line[-1].replace("ms","")));
                # end if
            # end for
            f.close();
        # end with
    # end for
    
    for key, val in SWIR_metadata_fdict["fcontent"].items():
        SWIR_metadata_fdict["fcontent"][key] = np.asarray(SWIR_metadata_fdict["fcontent"][key], dtype = np.object_);  # specify dtype = np.object_ to ensure that datatype of each item is retained during conversion to NumPy array
    # end for
    
    
    #-- sort and match
    if SWIR_img_fdict["fnames"].shape[0] <= SWIR_metadata_fdict["fcontent"]["cap_ID"].shape[0]:
        
        #-- sort images according to capture ID
        # Extract capture IDs from parsed filenames (assuming capture ID is at index 2)
        img_capture_ids = [int(fname[2]) for fname in SWIR_img_fdict["fnames"]]
        sort_ind = np.argsort(img_capture_ids);
        
        for key, val in SWIR_img_fdict.items():
            SWIR_img_fdict[key] = val[sort_ind];
        # end for
    
    
        #-- match metadata to images
        match_ind = [];
        for fname in SWIR_img_fdict["fnames"]:
            # print(f"int(fname[2]) = {int(fname[2])}");
            # print(f'from SWIR_metadata_fdict: {SWIR_metadata_fdict["fcontent"]["cap_ID"][np.argwhere(SWIR_metadata_fdict["fcontent"]["cap_ID"] == int(fname[2])).flatten()[0]]}')
            match_ind.append(np.argwhere(SWIR_metadata_fdict["fcontent"]["cap_ID"] == int(fname[2])).flatten()[0]);
        # end for
    
        for key, val in SWIR_metadata_fdict["fcontent"].items():
            SWIR_metadata_fdict["fcontent"][key] = val[match_ind];
        # end for
    
    elif SWIR_img_fdict["fnames"].shape[0] > SWIR_metadata_fdict["fcontent"]["cap_ID"].shape[0]:
        
        #-- sort metadata according to capture ID
        sort_ind = np.argsort(SWIR_metadata_fdict["fcontent"]["cap_ID"]);
    
        for key, val in SWIR_metadata_fdict["fcontent"].items():
            SWIR_metadata_fdict["fcontent"][key] = val[sort_ind];
        # end for


        #-- match images to metadata
        match_ind = [];
        for cap_ID in SWIR_metadata_fdict["fcontent"]["cap_ID"]:
            # Find the index of the image with matching capture ID
            img_capture_ids = [int(fname[2]) for fname in SWIR_img_fdict["fnames"]]
            match_idx = np.argwhere(np.array(img_capture_ids) == cap_ID).flatten()[0]
            match_ind.append(match_idx);
        # end for
    
        for key, val in SWIR_img_fdict.items():
            SWIR_img_fdict[key] = val[match_ind];
        # end for

    # Create output directory if it doesn't exist
    os.makedirs(MATCHED_OUTPUT_DIR, exist_ok=True);

    np.savetxt(os.path.join(MATCHED_OUTPUT_DIR, "202502_SWIR_Flatfield_Matched_Metadata_v2.csv"), np.transpose([SWIR_img_fdict["fpaths"], ["_".join(fname) for fname in SWIR_img_fdict["fnames"]], SWIR_metadata_fdict["fcontent"]["cap_ID"], SWIR_metadata_fdict["fcontent"]["TEC"], SWIR_metadata_fdict["fcontent"]["filt"]]), fmt = ['%s','%s','%d','%d','%d'], delimiter = ',', header = 'FILEPATH,FILENAME,CFC_CAPTURE_ID,TEC_READING(CELCIUS),FILTER_POSITION', comments = '');


if __name__ == "__main__":
    main();
# end if