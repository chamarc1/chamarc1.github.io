"""
SWIR NetCDF4 Processor for AirHARP2 SWIR sensor data

This module integrates with SWIR_Img_Processor to create NetCDF4 files from SWIR
images and metadata. The integration provides:

1. Automatic image-metadata matching via CFC capture IDs using SWIR_Img_Processor
2. Filter-position-specific NetCDF4 organization (positions 0-4)
3. Time extraction from CFC capture IDs (Unix timestamps)
4. TEC temperature and integration time from matched metadata
5. Comprehensive processing summaries and progress reporting
6. Automatic time range detection from processed images

Key Features:
- Uses SWIR_Img_Processor class for robust image/metadata matching
- Supports filter-specific dimensions and variables in NetCDF4
- Includes representative metadata values in global attributes
- Provides detailed processing progress and error handling
- Automatically determines time coverage from CFC capture IDs
- Uses 64-bit integers for CFC capture IDs (Unix timestamps in milliseconds)

Integration Flow:
1. SWIR_Img_Processor matches images with metadata by CFC capture ID
2. NC_Processor creates NetCDF4 structure with filter-position dimensions
3. Images are processed by filter position using matched metadata
4. Each image is written with its specific TEC temp and integration time

Usage Example:
python SWIR_NC_Processor.py \
    --img-dir /path/to/swir/images \
    --meta-dir /path/to/metadata \
    --output_dir /path/to/output \
    --start-img first_image.tiff \
    --end-img last_image.tiff \
    --level 1A

__name__ =      SWIR_NC_Processor.py
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
# PYTHON STANDARD LIBRARIES
import sys
import os
import datetime as dt
import numpy as np
import pandas as pd
import argparse
import tempfile
import shutil
import tempfile
import shutil
import xml.etree.ElementTree as ET

# REQUIRED ADDITIONAL LIBRARIES/MODULES
from netCDF4 import Dataset
from PIL import Image

# CUSTOM MODULES
# Add the parent directory to Python path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Handle imports that work both when run directly and when imported as module
try:
    from .SWIR_Image_Data_TEC_Metadata_Matcher import LocateFiles
    from .SWIR_Img_Processor import SWIR_img_processor
    from . import Constants as SWIR_constants
except ImportError:
    # If relative import fails, try absolute import
    from SWIR_Image_Data_TEC_Metadata_Matcher import LocateFiles
    from SWIR_Img_Processor import SWIR_img_processor
    import Constants as SWIR_constants

#----------------------------------------------------------------------------
#-- CLASS DEFINITIONS
#----------------------------------------------------------------------------
class NC_Processor():
    """
    NetCDF4 processor class for AirHARP2 SWIR sensor data.
    Creates and manages NetCDF4 files that store SWIR instrument data in L1A format.
    """
    
    #-- Constructor
    def __init__(self, fname, out_dir, start_time, end_time, filt_pos, int_time, cfc_capture_id, tec_reading, processing_level, include_er2_nav=False):
        """
        Initialize the NC_Processor with necessary parameters
        
        :param fname:           string defining NetCDF4 filename
        :param out_dir:         string defining output directory path
        :param start_time:      string defining time coverage start
        :param end_time:        string defining time coverage end
        :param filt_pos:        array of filter positions of SWIR
        :param int_time:        integration time in MICROSECONDS
        :param cfc_capture_id:  CFC Capture ID
        :param tec_reading:     TEC temperature reading
        :param processing_level: string defining processing level (e.g., "1A")
        :param include_er2_nav: logical True or False determining if ER2 IMU data should be included in output NETCDF4
        """

        #-- initialize instance attributes
        self.fname              = fname                         # NetCDF4 file name
        self.out_dir            = out_dir                       # directory to which NetCDF4 should be saved
        self.fpath              = os.path.join(out_dir, fname)  # full file path to NetCDF4
        self.start_time         = start_time
        self.end_time           = end_time
        self.filt_pos           = filt_pos                      # filter position of SWIR
        self.int_time           = int_time / 1000               # convert to milliseconds ONLY FOR USE IN NetCDF4 FILE
        self.cfc_capture_id     = cfc_capture_id
        self.tec_reading        = tec_reading
        self.processing_level   = processing_level
        self.include_er2_nav    = include_er2_nav
        self.nc_obj             = self.init_nc()                # initialize NetCDF4 file on instance initialization
        # Global frame counter for unified frames dimension
        self.frame_counter      = 0


    def init_nc(self):
        """
        Initialize NetCDF4 file for SWIR data
        
        :return nc: initialized NetCDF4 file object
        """
        
        #-- initialize NC file object
        nc = Dataset(self.fpath, mode = 'w', format = 'NETCDF4')
        
        #-- write NC file global attributes and return modified NC file object
        nc = self.set_global_attrs(nc)
        
        #-- create NC file groups and return modified NC file object
        nc = self.init_img_data_grp(nc)
        nc = self.init_nav_data_grp(nc)
        if self.include_er2_nav:
            nc = self.init_er2_nav_data_grp(nc)
        # end if
        
        return nc


    def set_global_attrs(self, nc):
        """
        Initialize NETCDF4 global attributes
        
        :param nc: NETCDF4 object
        
        :return nc: NETCDF4 object with global attributes
        """
    
        nc.title                = f"AirHARP2 SWIR Level {self.processing_level} data"
        nc.source               = "Generated by SWIR_NC_Processor.py"
        nc.HARP2_L1AGen_version = SWIR_constants.VERSION  # Version adapted from HARP2_L1AGen
        nc.SWIR_L1AGen_version  = "1.0"
        nc.processing_level     = self.processing_level
        nc.date_created         = dt.datetime.utcnow().isoformat(sep = 'T', timespec = 'milliseconds') + 'Z'
        nc.time_coverage_start  = self.start_time
        nc.time_coverage_end    = self.end_time
        nc.epoch_tai_year       = SWIR_constants.EPOCH_TAI.year
        nc.epoch_tai_month      = SWIR_constants.EPOCH_TAI.month
        nc.epoch_tai_day        = SWIR_constants.EPOCH_TAI.day
        nc.tai2utc_leapsec      = SWIR_constants.TAI2UTC
        nc.cfc_capture_id       = self.cfc_capture_id
        nc.tec_reading          = self.tec_reading
        
        return nc

    def init_img_data_grp(self, nc):
        """
        Initialize NETCDF4's image_data group and its corresponding variables and dimensions
        
        :param nc: NETCDF4 object in which image_data should be created
        
        :return nc: NETCDF4 object with image_data group initialized
        """
        img_grp = nc.createGroup("image_data")
        img_grp.long_name = "Group for AirHARP2 SWIR detector image data and status"
        
        #-- create dimensions for image_data group
        img_grp.createDimension("frames", None)          # unified frame dimension across all filter positions
        img_grp.createDimension("sensors", 1)
        img_grp.createDimension("lines", 1024)            # rows
        img_grp.createDimension("pixels", 1280)           # columns for SWIR
        img_grp.createDimension("number_filters", 5)

        #-- create unified variables with filter position dimensions
        # Image data with unified frames: sensor1[frame, line, pixel]
        img_grp.createVariable("sensor1", datatype = "u2", dimensions = ("frames", "lines", "pixels"), fill_value = 65535)
        
        # Frame and timestamps with filter position dimension
        img_grp.createVariable("frame_ID", datatype = "i4", dimensions = ("frames", "sensors"), fill_value = -999999)
        img_grp.createVariable("time_stamp", datatype = "i8", dimensions = ("frames", "sensors"), fill_value = -999999)
        img_grp.createVariable("JD", datatype = "f8", dimensions = ("frames", "sensors"), fill_value = -999999)
        img_grp.createVariable("seconds_of_day", datatype = "f8", dimensions = ("frames", "sensors"), fill_value = -999999)
        img_grp.createVariable("acquisition_type", datatype = "i4", dimensions = ("frames", "sensors"), fill_value = -999999)
        img_grp.createVariable("image_type", datatype = "i4", dimensions = ("frames", "sensors"), fill_value = -999999)
        img_grp.createVariable("binning_cross_track", datatype = "i4", dimensions = ("frames", "sensors"), fill_value = -999999)
        
        # Sensor information with filter position dimension
        img_grp.createVariable("sensor_temperature", datatype = "f4", dimensions = ("frames", "sensors"), fill_value = -999999)
        img_grp.createVariable("integration_time", datatype = "f4", dimensions = ("frames", "sensors"), fill_value = -999999)
        

        #-- define units, long_name, valid_min, and valid_max for each variable as applicable
        filter_descriptions = ["dark", "1.57 um filter", "1.55 um filter", "1.38 um filter", "open"]
        filter_wavelengths = [0., 1.57, 1.55, 1.38, 999.]  # wavelengths in micrometers
        
        # Unified sensor variable
        img_grp["sensor1"].units = "counts"
        img_grp["sensor1"].long_name = "Image read from SWIR Sensor counts"
        img_grp["sensor1"].sensor_bitdepth = 14
        img_grp["sensor1"].valid_min = 0
        img_grp["sensor1"].valid_max = 16383
        # CF-convention friendly scaling metadata (no-op unless changed later)
        img_grp["sensor1"].scale_factor = 1.0
        img_grp["sensor1"].add_offset = 0.0
        img_grp["sensor1"].comment = "Access data by frame: sensor1[frame, line, pixel]"
        img_grp["sensor1"].dimensions_info = "frames(unlimited), lines(1024), pixels(1280)"
        
        # Frame ID variable
        img_grp["frame_ID"].long_name = "Image ID number read from image header"
        img_grp["frame_ID"].comment = "Access data by frame: frame_ID[frame, sensor]"
        
        # Unified time stamp variable
        img_grp["time_stamp"].long_name = "CFC capture time stamp (Unix milliseconds)"
        img_grp["time_stamp"].comment = "Access data by frame: time_stamp[frame, sensor]"
        img_grp["time_stamp"].dimensions_info = "frames(unlimited), sensors(1)"
        
        # Unified time variables
        img_grp["JD"].units = "days"
        img_grp["JD"].long_name = "Calculated Julian Day for each frame"
        img_grp["JD"].comment = "Access data by frame: JD[frame, sensor]"
        
        img_grp["seconds_of_day"].units = "seconds"
        img_grp["seconds_of_day"].long_name = "Calculated elapsed seconds of the day"
        img_grp["seconds_of_day"].comment = "Access data by frame: seconds_of_day[frame, sensor]"

        # Acquisition type variable
        img_grp["acquisition_type"].long_name = "HARP2 acquisition mode flag: 0 = Indirect (Flash), 1 = Direct (SPW)"
        img_grp["acquisition_type"].comment = "Access data by frame: acquisition_type[frame, sensor]"

        # Image type variable
        img_grp["image_type"].long_name = "HARP2 operational mode flag: 0 = Science, 1 = Calibration"
        img_grp["image_type"].comment = "Access data by frame: image_type[frame, sensor]"

        # Cross-track binning variable
        img_grp["binning_cross_track"].long_name = "Cross-track binning mode: 0 = no binning, 2 = binning by 2, 4 = custom SCI binning"
        
        # Unified sensor information variables
        img_grp["sensor_temperature"].units = "degrees C"
        img_grp["sensor_temperature"].long_name = "Detector temperature"
        img_grp["sensor_temperature"].valid_min = -50.0
        img_grp["sensor_temperature"].valid_max = 60.0
        img_grp["sensor_temperature"].comment = "Access data by frame: sensor_temperature[frame, sensor]"
        
        img_grp["integration_time"].units = "milliseconds"
        img_grp["integration_time"].long_name = "Detector integration time"
        img_grp["integration_time"].valid_min = 1.0
        img_grp["integration_time"].valid_max = 1000.0
        img_grp["integration_time"].comment = "Access data by frame: integration_time[frame, sensor]"
        
        # Note: filter position is reported per frame via 'filter_wheel_position' variable

        # Number of filters listing variable
        img_grp.createVariable("number_filters", datatype = "i4", dimensions = ("number_filters",), fill_value = -999999)
        img_grp["number_filters"][:] = [0, 1, 2, 3, 4]
        img_grp["number_filters"].long_name = "Filter numbers (0-4)"

        # Filter wheel position per frame
        img_grp.createVariable("filter_wheel_position", datatype = "i4", dimensions = ("frames",), fill_value = -999999)
        img_grp["filter_wheel_position"].long_name = "Filter wheel position (0-4)"

        return nc


    def init_nav_data_grp(self, nc):
        """
        Initialize NETCDF4's navigation_data group and its corresponding variables and dimensions
        
        :param nc: NETCDF4 object in which navigation_data should be created
        
        :return nc: NETCDF4 object with navigation_data group initialized
        """
        
        #-- create navigation_data group
        nav_grp = nc.createGroup("navigation_data")
        nav_grp.long_name = "Group for spacecraft position and attitude from AirHARP2 IMU"
        
        #-- create dimensions for navigation_data group
        nav_grp.createDimension("quaternion_elements", 4)
        nav_grp.createDimension("vector_elements", 3)
        nav_grp.createDimension("att_records", None)
        nav_grp.createDimension("orb_records", None)
    
        #-- create group variables
        nav_grp.createVariable("att_time", datatype = "f8", dimensions = ("att_records"), fill_value = SWIR_constants.FILL_39)
        nav_grp.createVariable("att_quat", datatype = "f8", dimensions = ("att_records", "quaternion_elements"), fill_value = SWIR_constants.FILL_R8)
        nav_grp.createVariable("att_rate", datatype = "f8", dimensions = ("att_records", "vector_elements"), fill_value = SWIR_constants.FILL_R8)
        nav_grp.createVariable("att_euler", datatype = "f8", dimensions = ("att_records", "vector_elements"), fill_value = SWIR_constants.FILL_R8)
        nav_grp.createVariable("orb_time", datatype = "f8", dimensions = ("orb_records"), fill_value = SWIR_constants.FILL_39)
        nav_grp.createVariable("orb_pos", datatype = "f4", dimensions = ("orb_records", "vector_elements"), fill_value = SWIR_constants.FILL_R4)
        nav_grp.createVariable("orb_vel", datatype = "f4", dimensions = ("orb_records", "vector_elements"), fill_value = SWIR_constants.FILL_R4)
        nav_grp.createVariable("orb_lon", datatype = "f8", dimensions = ("orb_records"), fill_value = SWIR_constants.FILL_39)
        nav_grp.createVariable("orb_lat", datatype = "f8", dimensions = ("orb_records"), fill_value = SWIR_constants.FILL_39)
        nav_grp.createVariable("orb_alt", datatype = "f8", dimensions = ("orb_records"), fill_value = SWIR_constants.FILL_39)
        
        
        #-- define units, long_name, valid_min, and valid_max for each variable as applicable
        nav_grp["att_time"].units      = "seconds"
        nav_grp["att_time"].long_name  = "Attitude sample time (seconds of day)"
        nav_grp["att_time"].valid_min  = 0.0
        nav_grp["att_time"].valid_max  = 86400.999999
        nav_grp["att_quat"].units      = "unitless"
        nav_grp["att_quat"].long_name  = "Attitude quaternions (J2000 to space craft), in the order of q1, q2, q3, and q0"
        nav_grp["att_quat"].valid_min  = -1.0
        nav_grp["att_quat"].valid_max  = 1.0
        nav_grp["att_rate"].units      = "radians/second"
        nav_grp["att_rate"].long_name  = "Attitude angular rates in spacecraft frame"
        nav_grp["att_rate"].valid_min  = -0.004
        nav_grp["att_rate"].valid_max  = 0.004
        nav_grp["att_euler"].units     = "degrees"
        nav_grp["att_euler"].long_name = "Euler angles (roll, pitch, yaw) in spacecraft frame"
        nav_grp["att_euler"].valid_min = -360.0
        nav_grp["att_euler"].valid_max = 360.0
        nav_grp["orb_time"].units      = "Day"
        nav_grp["orb_time"].long_name  = "Julian Day for Spacecraft Orbit Records"
        nav_grp["orb_pos"].units       = "meters"
        nav_grp["orb_pos"].long_name   = "Spacecraft position vector in ECEF (ECR) frame"
        nav_grp["orb_pos"].valid_min   = -7200000.
        nav_grp["orb_pos"].valid_max   = 7200000.
        nav_grp["orb_vel"].units       = "meters/second"
        nav_grp["orb_vel"].long_name   = "Spacecraft velocity vector in ECEF (ECR) frame"
        nav_grp["orb_vel"].valid_min   = -7600.
        nav_grp["orb_vel"].valid_max   = 7600.
        nav_grp["orb_lon"].units       = "degrees"
        nav_grp["orb_lon"].long_name   = "Orbit longitude (degrees East)"
        nav_grp["orb_lon"].valid_min   = -180.
        nav_grp["orb_lon"].valid_max   = 180.
        nav_grp["orb_lat"].units       = "degrees"
        nav_grp["orb_lat"].long_name   = "Orbit latitude (degrees North)"
        nav_grp["orb_lat"].valid_min   = -90.
        nav_grp["orb_lat"].valid_max   = 90.
        nav_grp["orb_alt"].units       = "meters"
        nav_grp["orb_alt"].long_name   = "Orbit altitutde"
        nav_grp["orb_alt"].valid_min   = -1000
        nav_grp["orb_alt"].valid_max   = 50000.
        
        return nc


    def init_er2_nav_data_grp(self, nc):
        """
        Initialize NETCDF4's er2_navigation_data group and its corresponding variables and dimensions
        
        :param nc: NETCDF4 object in which er2_navigation_data should be created
        
        :return nc: NETCDF4 object with er2_navigation_data group initialized
        """
        
        #-- create er2_navigation_data group
        er2_grp = nc.createGroup("er2_navigation_data")
        er2_grp.long_name = "Group for spacecraft position and attitude from ER2"
        
        #-- create dimensions for er2_navigation_data group
        nc.createDimension("er2_vector_elements", 3)
        nc.createDimension("er2_orb_records", None)
        
        #-- create group variables
        er2_grp.createVariable("orb_time", datatype = "f8", dimensions = ("er2_orb_records"), fill_value = SWIR_constants.FILL_39)
        er2_grp.createVariable("orb_lon", datatype = "f8", dimensions = ("er2_orb_records"), fill_value = SWIR_constants.FILL_39)
        er2_grp.createVariable("orb_lat", datatype = "f8", dimensions = ("er2_orb_records"), fill_value = SWIR_constants.FILL_39)
        er2_grp.createVariable("orb_alt", datatype = "f8", dimensions = ("er2_orb_records"), fill_value = SWIR_constants.FILL_39)
        er2_grp.createVariable("att_euler", datatype = "f8", dimensions = ("er2_orb_records", "er2_vector_elements"), fill_value = SWIR_constants.FILL_R8)
        
        
        #-- define units, long_name, valid_min, and valid_max for each variable as applicable
        er2_grp["orb_time"].units       = "seconds"
        er2_grp["orb_time"].long_name   = "ER2 sample time (seconds of day)"
        er2_grp["orb_time"].valid_min   = 0.0
        er2_grp["orb_time"].valid_max   = 86400.999999
        er2_grp["orb_lon"].units        = "degrees"
        er2_grp["orb_lon"].long_name    = "ER2 longitude (degrees East)"
        er2_grp["orb_lon"].valid_min    = -180.
        er2_grp["orb_lon"].valid_max    = 180.
        er2_grp["orb_lat"].units        = "degrees"
        er2_grp["orb_lat"].long_name    = "ER2 latitude (degrees North)"
        er2_grp["orb_lat"].valid_min    = -90.
        er2_grp["orb_lat"].valid_max    = 90.
        er2_grp["orb_alt"].units        = "meters"
        er2_grp["orb_alt"].long_name    = "ER2 altitude"
        er2_grp["orb_alt"].valid_min    = -1000
        er2_grp["orb_alt"].valid_max    = 50000.
        er2_grp["att_euler"].units      = "degrees"
        er2_grp["att_euler"].long_name  = "ER2 Euler angles (roll, pitch, yaw) in spacecraft frame"
        er2_grp["att_euler"].valid_min  = -360.0
        er2_grp["att_euler"].valid_max  = 360.0
        
        return nc


    def write_img_to_l1a(self, img, filter_idx, iframe, cfc_capture_id, tec_temp=None, int_time_ms=None):
        """
        Write raw SWIR images to L1A NETCDF4
        
        :param img:        2D NumPy array of raw SWIR image. format: [rows][columns]
        :param filter_idx: integer determining filter index (0-4)
        :param iframe:     integer determining frame index for the specific filter position. NOTE: *list* index, not actual frame ID derived from filename
        :param cfc_capture_id: CFC capture ID (Unix timestamp in milliseconds)
        :param tec_temp:   optional TEC temperature reading for this frame
        :param int_time_ms: optional integration time in milliseconds for this frame
        """
    
        #-- write data to NETCDF4 variables using unified frames dimension
        gframe = self.frame_counter
        self.nc_obj["image_data"]["sensor1"][gframe, :, :] = img
        self.nc_obj["image_data"]["frame_ID"][gframe, 0] = gframe
        self.nc_obj["image_data"]["time_stamp"][gframe, 0] = cfc_capture_id
        self.nc_obj["image_data"]["filter_wheel_position"][gframe] = filter_idx
        self.nc_obj["image_data"]["integration_time"][gframe, 0] = int_time_ms
        self.nc_obj["image_data"]["sensor_temperature"][gframe, 0] = tec_temp
                
        # Set operational mode flags with default values
        self.nc_obj["image_data"]["binning_cross_track"][gframe, 0] = 0  # no binning
        self.nc_obj["image_data"]["acquisition_type"][gframe, 0] = 1  # Direct (SPW)
        self.nc_obj["image_data"]["image_type"][gframe, 0] = 0  # Science

        # advance global frame counter
        self.frame_counter += 1

        return


    def write_time_to_nc(self, frame_time, filter_idx, iframe):
        """
        Write time/frame-based data to L1A NETCDF4
        
        :param frame_time:  datetime object extracted from current frame's image name
        :param filter_idx:  integer determining filter index (0-4)
        :param iframe:      integer determining frame index for the specific filter position. NOTE: *list* index, not actual frame ID derived from filename
        """

        #-- convert to seconds and Julian Day
        frame_secs, frame_JD = SWIR_constants.calculate_seconds(frame_time)

        #-- write data to NETCDF4 variables using unified frames dimension
        gframe = max(self.frame_counter - 1, 0)
        self.nc_obj["image_data"]["JD"][gframe, 0] = frame_JD
        self.nc_obj["image_data"]["seconds_of_day"][gframe, 0] = frame_secs
        # Also ensure filter_wheel_position is set if not already
        if "filter_wheel_position" in self.nc_obj["image_data"].variables:
            # No change to existing value; this block is a placeholder in case time writes happen before image writes in some flows
            pass

        return


    def load_img(self, img_fpath):
        """
        Load SINGLE SWIR image
        """
        
        try:
            img = Image.open(img_fpath)
            img = np.array(img)
            return img
        except Exception as e:
            print(f"Error loading image {img_fpath}: {e}")
            return None
    
    def extract_time_from_filename(self, img_fname):
        """
        Extract timestamp from SWIR image filename
        Assumes filename format includes timestamp that can be parsed
        """
        try:
            # Extract timestamp from filename - adapt based on your filename format
            # This assumes format similar to UV: YYYYMMDDHHMMSS_...
            if isinstance(img_fname, (list, np.ndarray)):
                # If img_fname is the parsed filename array, reconstruct the original name
                timestamp_str = img_fname[0] if len(img_fname) > 0 else ""
            else:
                # If it's a string filename, extract timestamp
                timestamp_str = img_fname.split("_")[0]
            
            frame_time = dt.datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
            return frame_time
        except Exception as e:
            print(f"Error extracting time from filename {img_fname}: {e}")
            return None

    def process_swir_images_to_nc(self, swir_processor):
        """
        Process SWIR images using SWIR_img_processor and write to NetCDF4
        
        :param swir_processor: SWIR_img_processor instance with matched images and metadata
        """
        
        print("Processing SWIR images to NetCDF4 using SWIR_img_processor...")
        
        # Get summary of processed data
        summary = swir_processor.get_summary()
        print(f"Processing summary:")
        print(f"  Total images: {summary['total_images']}")
        print(f"  Filter position distribution: {summary['filter_position_counts']}")
        if summary['tec_temp_range']:
            print(f"  TEC temperature range: {summary['tec_temp_range'][0]}°C to {summary['tec_temp_range'][1]}°C")
        if summary['integration_time_range']:
            print(f"  Integration time range: {summary['integration_time_range'][0]}ms to {summary['integration_time_range'][1]}ms")
        
        # Process all frames in chronological order provided by the image processor
        if summary['total_images'] == 0:
            print("  No images to process.")
            return

        print(f"  Processing {summary['total_images']} images in chronological order")
        for i, frame in enumerate(swir_processor.iterate_chronological()):
            img_path = frame['img_fpath']
            img_name = frame['img_fname']
            filter_pos = frame['filter_pos']
            cfc_id = frame['cfc_id']
            tec_temp = frame['tec_temp']
            int_time_ms = frame['integration_time_ms']

            # Load image using SWIR_img_processor
            img_data = swir_processor.load_img(img_path)
            
            if img_data is None:
                print(f"    Warning: Failed to load image {img_name}")
                continue
            
            # Extract time from CFC capture ID
            frame_time = swir_processor.extract_time_from_cfc_id(cfc_id)
            
            # Write image data to NetCDF4 using the matched metadata
            self.write_img_to_l1a(
                img=img_data,
                filter_idx=filter_pos,
                iframe=i,
                cfc_capture_id=cfc_id,
                tec_temp=tec_temp,
                int_time_ms=int_time_ms
            )
            
            # Write time data to NetCDF4 if time extraction was successful
            if frame_time is not None:
                self.write_time_to_nc(frame_time, filter_pos, i)
            else:
                print(f"    Warning: Could not extract time from CFC ID {cfc_id}")
        
        print("All SWIR images processed and written to NetCDF4")
        return

    def close_nc(self):
        """
        Close the NetCDF4 file to save data to disk
        """
        self.nc_obj.close()
        return


#----------------------------------------------------------------------------
#-- FUNCTION DEFINITIONS
#----------------------------------------------------------------------------
def parse_input_args():
    """
    Define input argument parser for this script
    
    :return args: argparse. Namespace class instance, which holds arguments and their values
    """
    #-- create argument parser
    parser = argparse.ArgumentParser(description='Process SWIR images with TEC metadata into NetCDF4 format')
    
    #-- required arguments
    parser.add_argument('--img-dir',    type = str, required = True, help = "Directory containing SWIR image files (.tif)")
    parser.add_argument('--meta-dir',   type = str, required = True, help = "Directory containing SWIR metadata files (.meta)")
    parser.add_argument('--output_dir', type = str, required = True, help = "Output directory for NetCDF4 files")
    parser.add_argument('--start-img', type = str, required = True, help = "filename of first image in desired selection for processing. DO NOT INCLUDE FULL FILE PATH")
    parser.add_argument('--end-img',   type = str, required = True, help = "filename of last image in desired selection for processing. DO NOT INCLUDE FULL FILE PATH")
    parser.add_argument('--level',      type = str, required = True, help = "valid inputs: 1A")
    parser.add_argument('--ah2_imu_ncfile', type=str, required = True, help='full path to AH2 IMU netcdf file for navigation copy')
    
    #-- optional arguments
    parser.add_argument('-v', '--verbose', required = False, action='store_true', help='enable verbose output')
    parser.add_argument('--include-er2-nav', required = False, action='store_true', help='option to include ER2 navigation data in output NETCDF4 file')
    parser.add_argument('--er2-infile', type=str, required=False, default=None, help='full path to ER2 log file. required if --include-er2-nav is set')
    parser.add_argument('--er2-xmlfile', type=str, required=False, default=None, help='full path to ER2 XML file. required if --include-er2-nav is set')
    
    #-- return arguments
    return parser.parse_args()


def build_obpg_filename(mission, instrument, timestamp, level, version="V001", resolution=None, product=None, revision=None):
    """
    Construct OBPG-style filename for AirHARP2 products.

    Example: PACEPAX_AH2SWIRR.20230115T123456.L1A.V001.nc
    """
    parts = [f"{mission}_{instrument}", timestamp, f"L{str(level).upper()}"]
    if resolution:
        parts.append(resolution)
    if product:
        parts.append(product)
    if revision:
        parts.append(revision)
    parts.append(version)
    return ".".join(parts) + ".nc"


def copy_imu_to_l1a(nc4, ah2_imu_file, start_sec, end_sec, verbose=False):
    """
    (Function imported from HARP2_L1_Processor and updated to work with SWIR NC Processor)
    
    Extract navigation data from the AH2 IMU netcdf for the corresponding 
    AirHARP2 level 1A grandule time period.
    And, copy those data into the AirHARP2 level 1A netcdf file.
    (xxu, 9/04/2024)

    INPUTS:
        -          nc4: HARP2 L1A netcdf object
        - ah2_imu_file: numpy array of PACE level 1A HK netcdf filename with path
        -    start_sec: start time (seconds of day) of HARP2 granule
        -      end_sec: end start time (seconds of day) of HARP2 granule

    RETURNS:
        - nc4: save as input nc4
    """
    grpname = "navigation_data"
    pad = 60  # include 60-second padding for navigation data

    st_att = 0

    # open the IMU netcdf file
    nchk = Dataset(ah2_imu_file, "r", format="NETCDF4")

    # attitude data for time period
    hk_att_sec = np.array(nchk[grpname]["att_time"][:])
    aidx = np.where((hk_att_sec >= start_sec - pad) & (hk_att_sec < end_sec + pad))[0]
    natt = len(aidx)
    if verbose:
        print("Found att time records:", natt)

    # copy attitude/orbit data into target file
    if natt > 0:
        for varname in ("att_time", "orb_time", "orb_lon", "orb_lat", "orb_alt"):
            nc4[grpname][varname][st_att : st_att + natt] = nchk[grpname][varname][aidx]
        for varname in ("att_quat", "att_rate", "att_euler", "orb_pos", "orb_vel"):
            nc4[grpname][varname][st_att : st_att + natt, :] = nchk[grpname][varname][aidx, :]
    else:
        print("Warning: no att records found in PACE HK!!!!!!!!!")

    if verbose:
        print("Done copy HK data to HARP2 L1A!")

    nchk.close()

    return nc4


def parse_xml_ids_xml_id(xml_file_path):
    """
    (Helper function for write_er2_imu_given_time)
    Extract all id and xml:id attributes from an XML file.
    """
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        ids = {}

        for element in root.findall('.//*[@id]'):
            ids[element.get('id')] = 1

        for i, element in enumerate(root.findall('.//*[@{http://www.w3.org/XML/1998/namespace}id]')):
            ids[element.get('{http://www.w3.org/XML/1998/namespace}id')] = i + 1

        return ids
    except FileNotFoundError:
        print(f"Error: File not found: {xml_file_path}")
        return None
    except ET.ParseError:
        print(f"Error: Invalid XML format in file: {xml_file_path}")
        return None
    except Exception as exc:
        print(f"An unexpected error occurred: {exc}")
        return None


def write_er2_imu_given_time(nc4, csv_file, xml_file, start_sec, end_sec, pad_sec=60, verbose=False):
    """
    Copy ER2 telemetry (CSV + XML header) into er2_navigation_data group for the requested time window.
    """

    assert os.path.exists(csv_file), "write_er2_imu_given_time(): csv_file doesn't exist"
    assert os.path.exists(xml_file), "write_er2_imu_given_time(): xml_file doesn't exist"

    ids_with_xml_id = parse_xml_ids_xml_id(xml_file)
    if ids_with_xml_id and verbose:
        print("IDs and xml:ids found:", ids_with_xml_id)

    df = pd.read_csv(csv_file, comment='#')

    utc_time_str = df.iloc[:, 1]
    utc_time = [dt.datetime.strptime(t_str, '%Y-%m-%dT%H:%M:%S.%f') for t_str in utc_time_str]
    t0 = dt.datetime(utc_time[0].year, utc_time[0].month, utc_time[0].day)
    time_sec = np.array([(t - t0).total_seconds() for t in utc_time])

    aidx = np.where((time_sec >= start_sec - pad_sec) & (time_sec <= end_sec + pad_sec))[0]
    norb = len(aidx)
    if verbose:
        print(f'Number of records within the time period: {norb}')

    lat_lon_alt = np.array(df.iloc[aidx, 2:5])
    roll = np.array(df.iloc[aidx, 17])
    pitch = np.array(df.iloc[aidx, 16])
    yaw = np.array(df.iloc[aidx, 13])
    euler_angles = np.column_stack((roll, pitch, yaw))
    if verbose:
        print(lat_lon_alt.shape, euler_angles.shape)

    grpname = 'er2_navigation_data'
    st_att = 0
    if norb > 0:
        nc4[grpname]['orb_time'][st_att : st_att + norb] = time_sec[aidx]
        nc4[grpname]['orb_lat'][st_att : st_att + norb] = lat_lon_alt[:, 0]
        nc4[grpname]['orb_lon'][st_att : st_att + norb] = lat_lon_alt[:, 1]
        nc4[grpname]['orb_alt'][st_att : st_att + norb] = lat_lon_alt[:, 2]
        nc4[grpname]['att_euler'][st_att : st_att + norb, :] = euler_angles

    if verbose:
        print('Done copy ER2 data to HARP2 L1A!')

    return nc4


def main():
    """
    Main function to process SWIR images with metadata matching using integrated approach.
    
    Processing flow:
    1. Initialize SWIR_img_processor to match images with metadata by CFC capture ID
    2. Extract representative metadata values for NetCDF4 global attributes  
    3. Initialize NC_Processor with proper NetCDF4 structure
    4. Process all images using matched metadata from SWIR_img_processor
    5. Save completed NetCDF4 file to output directory
    """
    # Parse command line arguments
    args = parse_input_args()
    
    # Use command line arguments
    img_dir = args.img_dir
    metadata_dir = args.meta_dir
    output_dir = args.output_dir
    start_img = args.start_img
    end_img = args.end_img
    processing_level = args.level
    ah2_imu_ncfile = args.ah2_imu_ncfile
    include_er2_nav = args.include_er2_nav
    er2_infile = args.er2_infile if hasattr(args, 'er2_infile') else None
    er2_xmlfile = args.er2_xmlfile if hasattr(args, 'er2_xmlfile') else None
    verbose = args.verbose if hasattr(args, 'verbose') else False
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process images with metadata matching using SWIR_img_processor first to get actual metadata
    print("Initializing SWIR image processor...")
    swir_processor = SWIR_img_processor(
        img_dir=img_dir,
        meta_dir=metadata_dir, 
        start_img=start_img,
        end_img=end_img,
        processing_level=processing_level,
        verbose=True
    )
    
    # Get summary for better filename generation
    summary = swir_processor.get_summary()
    
    # Generate start_time and end_time from processed images
    if summary['total_images'] > 0:
        # Get time range from CFC capture IDs
        cfc_ids = swir_processor.matched_data['cfc_ids']
        earliest_time = swir_processor.extract_time_from_cfc_id(np.min(cfc_ids))
        latest_time = swir_processor.extract_time_from_cfc_id(np.max(cfc_ids))
        
        start_time = earliest_time.isoformat() if earliest_time else "unknown"
        end_time = latest_time.isoformat() if latest_time else "unknown"
        
        # Generate filename with actual time range
        if earliest_time:
            time_str = earliest_time.strftime("%Y%m%dT%H%M%S")
        else:
            time_str = "unknown"
    else:
        start_time = "unknown"
        end_time = "unknown"
        time_str = "unknown"
    
    # Generate filename with actual metadata
    tmp_out_dir = tempfile.mkdtemp()
    fname = build_obpg_filename(
        mission="PACEPAX",
        instrument="AH2SWIRR",
        timestamp=time_str,
        level=processing_level,
        version="V001"
    )
    print(f"Initializing {fname} NetCDF4...")
    
    # Get representative metadata values for NetCDF initialization
    if summary['total_images'] > 0:
        representative_tec = int(np.mean(swir_processor.matched_data['tec_temps'])) if len(swir_processor.matched_data['tec_temps']) > 0 else 0
        representative_int_time = int(np.mean(swir_processor.matched_data['integration_times']) * 1000) if len(swir_processor.matched_data['integration_times']) > 0 else 5000  # convert to microseconds
        representative_cfc_id = int(swir_processor.matched_data['cfc_ids'][0]) if len(swir_processor.matched_data['cfc_ids']) > 0 else 0
    else:
        representative_tec = 0
        representative_int_time = 5000
        representative_cfc_id = 0
    
    # Initialize NC_Processor with actual metadata
    nc_processor = NC_Processor(
        fname=fname,
        out_dir=tmp_out_dir,
        start_time=start_time,
        end_time=end_time,
        filt_pos=0,  # placeholder - will use filter-specific variables anyway
        int_time=representative_int_time,  # representative value in microseconds
        cfc_capture_id=representative_cfc_id,  # representative value
        tec_reading=representative_tec,  # representative value
        processing_level=processing_level,
        include_er2_nav=include_er2_nav
    )
    print(f"{nc_processor.fpath} NetCDF4 initialization successful")
    
    # Process all images using the integrated method
    nc_processor.process_swir_images_to_nc(swir_processor)
    
    # Add navigation data before closing if IMU file is provided
    if ah2_imu_ncfile and os.path.exists(ah2_imu_ncfile):
        print("\nCopying AH2 IMU navigation data...")
        # Calculate seconds of day for start and end times
        if earliest_time and latest_time:
            t0_start = dt.datetime(earliest_time.year, earliest_time.month, earliest_time.day, tzinfo=dt.timezone.utc)
            start_sec = (earliest_time - t0_start).total_seconds()
            end_sec = (latest_time - t0_start).total_seconds()
            
            # Reopen NetCDF in append mode to add navigation data
            nc_for_nav = Dataset(nc_processor.fpath, mode='r+', format='NETCDF4')
            nc_for_nav = copy_imu_to_l1a(nc_for_nav, ah2_imu_ncfile, start_sec, end_sec, verbose=verbose)
            nc_for_nav.close()
            print("AH2 IMU navigation data copied successfully")
        else:
            print("Warning: Could not compute time window for IMU data copy")
    
    # Add ER2 navigation data if requested
    if include_er2_nav and er2_infile and er2_xmlfile:
        if os.path.exists(er2_infile) and os.path.exists(er2_xmlfile):
            print("\nCopying ER2 navigation data...")
            if earliest_time and latest_time:
                t0_start = dt.datetime(earliest_time.year, earliest_time.month, earliest_time.day, tzinfo=dt.timezone.utc)
                start_sec = (earliest_time - t0_start).total_seconds()
                end_sec = (latest_time - t0_start).total_seconds()
                
                # Reopen NetCDF in append mode to add ER2 navigation data
                nc_for_er2 = Dataset(nc_processor.fpath, mode='r+', format='NETCDF4')
                nc_for_er2 = write_er2_imu_given_time(nc_for_er2, er2_infile, er2_xmlfile, 
                                                       start_sec, end_sec, pad_sec=60, verbose=verbose)
                nc_for_er2.close()
                print("ER2 navigation data copied successfully")
            else:
                print("Warning: Could not compute time window for ER2 data copy")
        else:
            print(f"Warning: ER2 files not found: {er2_infile}, {er2_xmlfile}")
    
    # Close the NetCDF file
    nc_processor.close_nc()
    
    # Get final summary from SWIR processor
    final_summary = swir_processor.get_summary()
    
    print(f"\n" + "=" * 60)
    print("SWIR L1A PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total images processed: {final_summary['total_images']}")
    print(f"Processing level: L{processing_level}")
    print(f"Filter position distribution:")
    filter_descriptions = ["dark", "1.57μm", "1.55μm", "1.38μm", "open"]
    for pos, count in final_summary['filter_position_counts'].items():
        if count > 0:
            print(f"  Position {pos} ({filter_descriptions[pos]}): {count} images")
    
    if final_summary['tec_temp_range']:
        print(f"TEC temperature range: {final_summary['tec_temp_range'][0]}°C to {final_summary['tec_temp_range'][1]}°C")
    if final_summary['integration_time_range']:
        print(f"Integration time range: {final_summary['integration_time_range'][0]}ms to {final_summary['integration_time_range'][1]}ms")
    
    # Copy completed NetCDF4 from temporary directory to true output directory
    os.makedirs(output_dir, exist_ok=True)
    final_path = os.path.join(output_dir, fname)
    shutil.copy2(nc_processor.fpath, final_path)
    print(f"\nNetCDF4 file saved: {final_path}")
    
    # Remove temporary directory
    shutil.rmtree(tmp_out_dir)
    print("Temporary directory cleaned up.")
    print("=" * 60)


if __name__ == "__main__":
    main()