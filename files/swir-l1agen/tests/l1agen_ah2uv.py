"""

__author__ = cjescobar

Copied from same author's "UV_Pushbroom_Generator.py" code from cjescobar/AH2_UV_pushbroom_gen repo, commit 2befa4a749 (most recent commit relevant to that script prior to copying)
Modifications made after copying aforementioned script as necessary; previously served as both L1A and pushbroom quicklook script, with input arguments toggling processing level (*STILL IN PROGRESS)

"""

#----------------------------------------------------------------------------
#-- IMPORT STATEMENTS
#----------------------------------------------------------------------------
# PYTHON STANDARD LIBRARIES
import os;
import sys;
import tempfile;
import shutil;
import datetime as dt;

# REQUIRED ADDITIONAL LIBRARIES/MODULES
import numpy as np;
import pandas as pd;
from netCDF4 import Dataset;
from PIL import Image;

# CUSTOM MODULES
import modules_l1a_uv as uv1a;


#----------------------------------------------------------------------------
#-- CLASS DEFINITIONS
#----------------------------------------------------------------------------
class UV_img_processor:
    """
    [insert desc here]
    """
    
    #-- constructor
    def __init__(self, img_dir, start_img, end_img, int_time, verbose):
        """
        [insert desc here]
        
        :param img_dir:          string defining root directory containing UV images of interest. in flight mode, root_dir should only contain image files, NO subdirectories
        :param start_img:        string defining file to start pushbroom with
        :param end_img:          string defining file to end pushbroom with
        :param int_time:         integer defining integration time of images in MICROSECONDS, as is indicated by UV camera controller. all images for given flight leg should have same integration time. NOTE: conversion to milliseconds to match AH2MAP only occurs in NC_processor
        :param verbose:          logical True or False for printing script progress to terminal. determined by script command-line argument
        """
        
        #-- initialize instance attributes
        self.img_dir          = img_dir;
        self.start_img        = start_img;
        self.end_img          = end_img;
        self.int_time         = int_time;
        self.verbose          = verbose;
        self.img_finfo        = self.enum_imgs();

    
    def enum_imgs(self):
        """
        [insert desc here]
        Modified from previous versions of UV_Capture_Pushbroom_Generator.py's FetchUVImages function
        
        :return img_finfo: Dictionary containing valid UV images' filenames, full paths, timestamps (from filenames) and chronological frame IDs as lists. Keys: fnames, fpaths, , ftimes, frame_IDs
        """

        if self.verbose:
            print(f"\nGetting image filenames and paths...");
        # end if
        
        
        #-- list files in img_dir
        fnames = os.listdir(self.img_dir);
        fnames.sort();  # ensure that files are parsed in filename alphabetical order. sort() sorts list in place
        idx_start_file = fnames.index(self.start_img);  # find index corresponding to desired starting file
        idx_end_file   = fnames.index(self.end_img);    # find index corresponding to desired ending file
        fnames = fnames[idx_start_file:idx_end_file + 1];  # get slice of num_imgs of files starting from start_img. since list slicing is end-exclusive, setting stop point to idx_end_file + 1 results in final value actually being idx_end_file


        #-- get full paths to uncorrupted TIFF files
        # initialize dictionary containing lists of desired image filenames, full paths, and chronological frame IDs
        img_finfo = {
                    "fnames":    [],
                    "fpaths":    [],
                    "ftimes":    [],
                    "frame_IDs": []
                    };
        
        for fname in fnames:
            fpath = os.path.join(self.img_dir, fname);
            ftime = dt.datetime.strptime(fname.split("_")[0], "%Y%m%d%H%M%S");  # first part of filename is datetime
            frame_ID = int(fname.split("_")[1]);  # second part of filename is frame ID
            
            # only include uncorrupted TIFF files
            if (fname.lower().endswith((".tif", ".tiff"))) and (os.path.getsize(fpath) >= uv1a.constants.UV_FILE_SIZE_MIN):
                img_finfo["fnames"].append(fname);
                img_finfo["fpaths"].append(fpath);
                img_finfo["ftimes"].append(ftime);
                img_finfo["frame_IDs"].append(frame_ID);
            # end if
        # end for
        

        return img_finfo;

    
    def load_img(self, img_fpath):
        """
        Load SINGLE image
        """
        
        try:
            # specifying in two steps instead of np.asarray(Image.open(img_fpath)) so that original image object can be closed. code runs slower if image object is not closed
            img_obj = Image.open(img_fpath);
            img = np.asarray(img_obj, dtype = np.uint16);
            
            img_obj.close();
        except Exception as e:
            print(f"Error loading image {img_fpath}: {e}");
            img = None;
        
        return img;


class NC_processor:
    
    #-- constructor
    def __init__(self, fname, out_dir, start_time, start_time_sod, end_time, end_time_sod, int_time):
        """
        [insert desc here]
        """
        
        #-- initialize instance attributes
        self.fname            = fname;  # NETCDF4 file name
        self.tmp_dir          = tempfile.mkdtemp();  # directory for temporary file to handle reading/writing to NetCDF4; read/write via temp file speeds up code
        self.tmp_fpath        = os.path.join(self.tmp_dir, fname);  # full file path to NETCDF4 temporary file
        self.out_dir          = out_dir;  # directory to which NETCDF4 should be saved after all reading/writing is complete
        self.fpath            = os.path.join(out_dir, fname);  # full file path to NETCDF4 final destination (after all reading/writing is complete)
        self.start_time       = start_time;  # start time as string
        self.start_time_sod   = start_time_sod;  # start time as float representing seconds of day
        self.end_time         = end_time;  # end time as string
        self.end_time_sod     = end_time_sod;  # end time as float representing seconds of day
        self.int_time         = int_time / 1000;  # convert to milliseconds ONLY FOR USE IN NETCDF4 FILE
        self.nc_dset          = self.init_nc();  # initialize L1A NETCDF4 Dataset object on instance initialization
    
    
    def init_nc(self):
        """
        Based on Richard Xu's "init_l1a_nc4" function from "l1agen_harp2.py", "nc4_util.py", and "constants.py"
        Initialize Level 1A NETCDF4 file for UV data
        
        :return nc: initialized L1A NETCDF4 file object
        """
        
        #-- initialize NC file object at temporary location for fast reading/writing
        nc = Dataset(self.tmp_fpath, mode = 'w', format = 'NETCDF4');
        
        
        #-- write NC file global attributes and dimensions and return modified NC file object
        nc = self.set_global_attrs(nc);
        nc = self.set_dimensions(nc);
        
        #-- create NC file groups and return modified NC file object
        nc = self.init_img_data_grp(nc);
        nc = self.init_nav_data_grp(nc);
        nc = self.init_er2_nav_data_grp(nc);
        # end if
        
        return nc;

    
    def set_global_attrs(self, nc):
        """
        Initialize NETCDF4 global attributes
        
        :param nc: NETCDF4 object
        
        :return nc: NETCDF4 object with global attributes
        """
    
        nc.title                = f"AirHARP2 UV Level 1A data";
        nc.source               = "Generated by UV_Capture_Pushbroom_Generator.py";
        nc.HARP2_L1AGen_version = uv1a.constants.VERSION;  # specified as HARP2_L1AGen_version since script uses some of HARP2_L1 modules
        nc.UV_L1AGen_version    = "None";
        nc.date_created         = dt.datetime.utcnow().isoformat(sep = 'T', timespec = 'milliseconds') + 'Z'
        # nc.acquisition_scheme   = "None";  # !!! LEAVE AS None OR REMOVE? !!!
        nc.time_coverage_start  = self.start_time;
        nc.time_coverage_end    = self.end_time;
        nc.epoch_tai_year       = uv1a.constants.EPOCH_UNIX.year;
        nc.epoch_tai_month      = uv1a.constants.EPOCH_UNIX.month;
        nc.epoch_tai_day        = uv1a.constants.EPOCH_UNIX.day;
        nc.tai2utc_leapsec      = uv1a.constants.TAI2UTC;
        
        
        return nc;
    
    
    def set_dimensions(self, nc):
        """
        Initialize dimensions for NETCDF4 variables
        
        :param nc: NETCDF4 object
        
        :return nc: NETCDF4 object with defined dimensions
        """
    
        #-- create dimensions for image_data group
        nc.createDimension("wavelengths", None);
        nc.createDimension("frames", None);
        nc.createDimension("sensors", 1);
        nc.createDimension("lines", 1024);  # rows
        nc.createDimension("pixels", 1024);  # columns
        
        
        #-- create dimensions for navigation_data group
        nc.createDimension("quaternion_elements", 4);
        nc.createDimension("vector_elements", 3);
        nc.createDimension("att_records", None);
        nc.createDimension("orb_records", None);
        
        
        #-- create dimensions for er2_navigation_data group
        nc.createDimension("er2_vector_elements", 3);
        nc.createDimension("er2_orb_records", None);
        # end if
        
        return nc;


    def init_img_data_grp(self, nc):
        """
        Based on Richard Xu's "init_l1a_nc4" function from "l1agen_harp2.py", "nc4_util.py", and "constants.py" with modifications as necessary
        Initialize NETCDF4's image_data group and its corresponding variables and dimensions
        
        :param nc: NETCDF4 object in which image_data should be created
        
        :return nc: NETCDF4 object with image_data group initialized
        """
        
        #-- create image_data group
        img_grp = nc.createGroup("image_data");
        img_grp.long_name = "Group for AirHARP2 UV detector image data and status";
        

        #-- create group variables
        # image data
        img_grp.createVariable("sensor", datatype = "u2", dimensions = ("frames", "lines", "pixels"), fill_value = uv1a.constants.FILL_U2);
  
        
        # frame and timestamps
        img_grp.createVariable("frame_ID", datatype = "i4", dimensions = ("frames", "sensors"), fill_value = uv1a.constants.FILL_I4);
        img_grp.createVariable("time_in_seconds",    datatype = "i4", dimensions = ("frames", "sensors"), fill_value = uv1a.constants.FILL_I4);
        img_grp.createVariable("time_in_subseconds", datatype = "i4", dimensions = ("frames", "sensors"), fill_value = uv1a.constants.FILL_I4);
        img_grp.createVariable("JD",                 datatype = "f8", dimensions = ("frames", "sensors"), fill_value = uv1a.constants.FILL_R8);
        img_grp.createVariable("seconds_of_day",     datatype = "f8", dimensions = ("frames", "sensors"), fill_value = uv1a.constants.FILL_R8);
    
        
        # sensor information
        img_grp.createVariable("sensor_temperature", datatype = "f4", dimensions = ("frames", "lines", "pixels"), fill_value = uv1a.constants.FILL_R4);  # !! SEE ABOVE NOTE. ALSO, RENAMED FROM "ccd_temperature". ALSO, TEMP INFORMATION IS UNAVAILABLE FOR UV !!!
        img_grp.createVariable("integration_time",   datatype = "f4", dimensions = ("frames", "sensors"),         fill_value = uv1a.constants.FILL_R4);  # !!! KEPT SAME DIMENSIONS AS HARP2/AIRHARP2 L1A, AS INTEGRATION TIME IS CONSTANT FOR UV DURING SCIENCE FLIGHTS !!!
        # img_grp.createVariable("acquisition_scheme_ID");  # !!! WHAT TO DO WITH THIS FOR UV? !!!
        # img_grp.createVariable("number_filters");  # !!! WHAT TO DO WITH THIS FOR UV? !!!
        # img_grp.createVariable("pixel_clock");  # !!! WHAT TO DO WITH THIS FOR UV? !!!
        # img_grp.createVariable("acquisition_type");  # !!! WHAT TO DO WITH THIS FOR UV? !!!
        img_grp.createVariable("image_type",         datatype = "i4", dimensions = ("frames", "sensors"), fill_value = uv1a.constants.FILL_I4);  # !!! WHAT TO DO WITH THIS FOR UV? !!!
        # img_grp.createVariable("shutter_status");  # !!! WHAT TO DO WITH THIS FOR UV? !!!
        # img_grp.createVariable("binning_cross_track");  # !!! WHAT TO DO WITH THIS FOR UV? !!!
    
    
        #-- define units, long_name, valid_min, and valid_max for each variable as applicable !!! THESE ARE ALSO VERY CHANGED FROM HARP2/AIRHARP2 L1A !!!
        # image data
        img_grp["sensor"].units     = "DN";
        img_grp["sensor"].long_name = "Raw Image Captured with AirHARP2 UV Camera";
        # img_grp["sensor1aux"].units
        # img_grp["sensor1aux"].long_name
        
        
        # frame and timestamps
        img_grp["frame_ID"].long_name           = "Image ID in chronological order";  # !!! REWORD? !!!
        img_grp["time_in_seconds"].long_name    = "Acquisition Time in Seconds ... ";
        img_grp["time_in_subseconds"].units     = "microseconds";
        img_grp["time_in_subseconds"].long_name = "Elapsed microseconds of the second";
        img_grp["JD"].units                     = "days";
        img_grp["JD"].long_name                 = "Calculated Julian Day for each frame after checking times for sensor";
        img_grp["seconds_of_day"].units         = "seconds";
        img_grp["seconds_of_day"].long_name     = "Calculated elapsed seconds of the day";
        
        
        # sensor information
        # img_grp["sensor_temperature"].units = "degrees C";
        # img_grp["sensor_temperature"].long_name = "Detector temperature";
        # img_grp["sensor_temperature"].valid_min = ;
        # img_grp["sensor_temperature"].valid_max = ;
        img_grp["integration_time"].units = "milliseconds";  # !!! KEPT AS MILLISECONDS TO MATCH HARP2/AIRHARP2 L1AGEN, BUT UV INTEGRATION TIME IS ACTUALLY SET IN MICROSECONDS. DO CONVERSION FOR NC FILE OR KEEP AS MICROSECONDS INSTEAD? !!!
        img_grp["integration_time"].long_name = "Detector integration time";
        img_grp["integration_time"].valid_min = 0.024;  # THIS IS IN MILLISECONDS; == 24 MICROSECONDS. FROM MJ042MR-GP-P11-BSI SPECIFICATIONS
        img_grp["integration_time"].valid_max = 30000;  # THIS IS IN MILLISECONDS; == 30 SECONDS. FROM MJ042MR-GP-P11-BSI SPECIFICATIONS
        # img_grp["acquisition_scheme_ID"].long_name = ;
        # img_grp["number_filters"].long_name = ;
        # img_grp["pixel_clock"].long_name = ;
        # img_grp["acquisition_type"].long_name = ;
        img_grp["image_type"].long_name = "Operational mode flag: 0 = Science, 1 = Calibration";
        # img_grp["shutter_status"].long_name = ;
        # img_grp["binning_cross_track"].long_name = ;


        return nc;


    def init_nav_data_grp(self, nc):
        """
        Based on Richard Xu's "init_l1a_nc4" function from "l1agen_harp2.py", "nc4_util.py", and "constants.py" with modifications as necessary
        Initialize NETCDF4's navigation_data group and its corresponding variables and dimensions
        
        :param nc: NETCDF4 object in which navigation_data should be created
        
        :return nc: NETCDF4 object with navigation_data group initialized
        """
        
        #-- create navigation_data group
        nav_grp = nc.createGroup("navigation_data");
        nav_grp.long_name = "Group for spacecraft position and attitude from AirHARP2 IMU";
        
    
        #-- create group variables
        nav_grp.createVariable("att_time", datatype = "f8", dimensions = ("att_records"), fill_value = uv1a.constants.FILL_39);
        nav_grp.createVariable("att_quat", datatype = "f8", dimensions = ("att_records", "quaternion_elements"), fill_value = uv1a.constants.FILL_R8);
        nav_grp.createVariable("att_rate", datatype = "f8", dimensions = ("att_records", "vector_elements"), fill_value = uv1a.constants.FILL_R8);
        nav_grp.createVariable("att_euler", datatype = "f8", dimensions = ("att_records", "vector_elements"), fill_value = uv1a.constants.FILL_R8);
        nav_grp.createVariable("orb_time", datatype = "f8", dimensions = ("orb_records"), fill_value = uv1a.constants.FILL_39);
        nav_grp.createVariable("orb_pos", datatype = "f4", dimensions = ("orb_records", "vector_elements"), fill_value = uv1a.constants.FILL_R4);
        nav_grp.createVariable("orb_vel", datatype = "f4", dimensions = ("orb_records", "vector_elements"), fill_value = uv1a.constants.FILL_R4);
        nav_grp.createVariable("orb_lon", datatype = "f8", dimensions = ("orb_records"), fill_value = uv1a.constants.FILL_39);
        nav_grp.createVariable("orb_lat", datatype = "f8", dimensions = ("orb_records"), fill_value = uv1a.constants.FILL_39);
        nav_grp.createVariable("orb_alt", datatype = "f8", dimensions = ("orb_records"), fill_value = uv1a.constants.FILL_39);
        
        
        #-- define units, long_name, valid_min, and valid_max for each variable as applicable
        nav_grp["att_time"].units      = "seconds";
        nav_grp["att_time"].long_name  = "Attitude sample time (seconds of day)";
        nav_grp["att_time"].valid_min  = 0.0;
        nav_grp["att_time"].valid_max  = 86400.999999;
        nav_grp["att_quat"].units      = "unitless";
        nav_grp["att_quat"].long_name  = "Attitude quaternions (J2000 to space craft), in the order of q1, q2, q3, and q0";
        nav_grp["att_quat"].valid_min  = -1.0;
        nav_grp["att_quat"].valid_max  = 1.0;
        nav_grp["att_rate"].units      = "radians/second";
        nav_grp["att_rate"].long_name  = "Attitude angular rates in spacecraft frame";
        nav_grp["att_rate"].valid_min  = -0.004;
        nav_grp["att_rate"].valid_max  = 0.004;
        nav_grp["att_euler"].units     = "degrees";
        nav_grp["att_euler"].long_name = "Euler angles (roll, pitch, yaw) in spacecraft frame";
        nav_grp["att_euler"].valid_min = -360.0;
        nav_grp["att_euler"].valid_max = 360.0;
        nav_grp["orb_time"].units      = "Day";
        nav_grp["orb_time"].long_name  = "Julian Day for Spacecraft Orbit Records"
        nav_grp["orb_pos"].units       = "meters";
        nav_grp["orb_pos"].long_name   = "Spacecraft position vector in ECEF (ECR) frame";
        nav_grp["orb_pos"].valid_min   = -7200000.;
        nav_grp["orb_pos"].valid_max   = 7200000.;
        nav_grp["orb_vel"].units       = "meters/second";
        nav_grp["orb_vel"].long_name   = "Spacecraft velocity vector in ECEF (ECR) frame";
        nav_grp["orb_vel"].valid_min   = -7600.;
        nav_grp["orb_vel"].valid_max   = 7600.;
        nav_grp["orb_lon"].units       = "degrees";
        nav_grp["orb_lon"].long_name   = "Orbit longitude (degrees East)";
        nav_grp["orb_lon"].valid_min   = -180.;
        nav_grp["orb_lon"].valid_max   = 180.;
        nav_grp["orb_lat"].units       = "degrees";
        nav_grp["orb_lat"].long_name   = "Orbit latitude (degrees North)";
        nav_grp["orb_lat"].valid_min   = -90.;
        nav_grp["orb_lat"].valid_max   = 90.;
        nav_grp["orb_alt"].units       = "meters";
        nav_grp["orb_alt"].long_name   = "Orbit altitutde";
        nav_grp["orb_alt"].valid_min   = -1000;
        nav_grp["orb_alt"].valid_max   = 50000.;
        
        
        return nc;

    
    def init_er2_nav_data_grp(self, nc):
        """
        Based on Richard Xu's "init_l1a_nc4" function from "l1agen_harp2.py", "nc4_util.py", and "constants.py" with modifications as necessary
        Initialize NETCDF4's er2_navigation_data group and its corresponding variables and dimensions
        
        :param nc: NETCDF4 object in which er2_navigation_data should be created
        
        :return nc: NETCDF4 object with er2_navigation_data group initialized
        """
        
        #-- create er2_navigation_data group
        er2_grp = nc.createGroup("er2_navigation_data");
        er2_grp.long_name = "Group for spacecraft position and attitude from ER2";
        
        
        #-- create group variables
        er2_grp.createVariable("orb_time", datatype = "f8", dimensions = ("er2_orb_records"), fill_value = uv1a.constants.FILL_39);
        er2_grp.createVariable("orb_lon", datatype = "f8", dimensions = ("er2_orb_records"), fill_value = uv1a.constants.FILL_39);
        er2_grp.createVariable("orb_lat", datatype = "f8", dimensions = ("er2_orb_records"), fill_value = uv1a.constants.FILL_39);
        er2_grp.createVariable("orb_alt", datatype = "f8", dimensions = ("er2_orb_records"), fill_value = uv1a.constants.FILL_39);
        er2_grp.createVariable("att_euler", datatype = "f8", dimensions = ("er2_orb_records", "er2_vector_elements"), fill_value = uv1a.constants.FILL_R8);
        
        
        #-- define units, long_name, valid_min, and valid_max for each variable as applicable
        er2_grp["orb_time"].units       = "seconds";
        er2_grp["orb_time"].long_name   = "ER2 sample time (seconds of day)";
        er2_grp["orb_time"].valid_min   = 0.0;
        er2_grp["orb_time"].valid_max   = 86400.999999;
        er2_grp["orb_lon"].units        = "degrees";
        er2_grp["orb_lon"].long_name    = "ER2 longitude (degrees East)";
        er2_grp["orb_lon"].valid_min    = -180.;
        er2_grp["orb_lon"].valid_max    = 180.;
        er2_grp["orb_lat"].units        = "degrees";
        er2_grp["orb_lat"].long_name    = "ER2 latitude (degrees North)";
        er2_grp["orb_lat"].valid_min    = -90.;
        er2_grp["orb_lat"].valid_max    = 90.;
        er2_grp["orb_alt"].units        = "meters";
        er2_grp["orb_alt"].long_name    = "ER2 altitude";
        er2_grp["orb_alt"].valid_min    = -1000;
        er2_grp["orb_alt"].valid_max    = 50000.;
        er2_grp["att_euler"].units      = "degrees";
        er2_grp["att_euler"].long_name  = "ER2 Euler angles (roll, pitch, yaw) in spacecraft frame";
        er2_grp["att_euler"].valid_min  = -360.0;
        er2_grp["att_euler"].valid_max  = 360.0;
        
        
        return nc;


    def copy_ah2_imu_to_l1a(self, ah2_imu_file, verbose = False):
        """
        Copied from function "copy_imu_to_l1a" in HARP2-L1-Processor's ah2_imu.py, with modifications as needed for class method
        """
        '''
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
        '''

        if ah2_imu_file is not None:

            #-- group name is a constant
            grpname = 'navigation_data'

            #-- include 60-second padding for navigation data
            pad  = 60

            #-- initialize start record nums
            st_att = 0

            #-- open the IMU netcdf file
            nchk = Dataset( ah2_imu_file, 'r', format='NETCDF4' )

            #-- attitude data for time period
            hk_att_sec = np.array(nchk[grpname]['att_time'][:])
            aidx = np.where( (hk_att_sec >= self.start_time_sod-pad) & (hk_att_sec <self.end_time_sod+pad) )[0]
            natt = len(aidx)
            if verbose:
                print('Found att time records:', natt)
            # end if

             #-- now copy corresponding att data
            if natt > 0:

                #-- copy data
                for varname in ('att_time','orb_time','orb_lon','orb_lat','orb_alt'):
                    self.nc_dset[grpname][varname][st_att:st_att+natt] = nchk[grpname][varname][aidx]
                # end for
                for varname in ('att_quat','att_rate','att_euler','orb_pos','orb_vel'):
                    self.nc_dset[grpname][varname][st_att:st_att+natt,:] = nchk[grpname][varname][aidx,:]
                # end for
            else:
                #-- give a warning if no records found
                print('Warning: no att records found in PACE HK!!!!!!!!!')
            # end if

            if verbose:
                print('Done copy HK data to HARP2 L1A!')
            # end if
        # end if
    
    
    def copy_er2_imu_to_l1a(self, er2_imu_csv_file, pad_sec = 60, verbose = False):
        """
        Copied from function "write_er2_imu_given_time" in HARP2-L1-Processor's er2_imu_util.py, with modifications as needed for class method; removed references to ER2 XML file as it is not used/source code had its use commented out
        """
    
        # inputs:
        #   - er2_imu_csv_file: the text file with ER2 telemetry data

        # verify input files
        assert os.path.exists(er2_imu_csv_file), "write_er2_imu_given_time(): csv_file doesn't exist"


        # Read the CSV file into a DataFrame
        df = pd.read_csv(er2_imu_csv_file, comment='#')
        
        # convert time string into utc time and seconds of day
        utc_time_str = df.iloc[:,1]
        print(utc_time_str.shape)
        print(utc_time_str)
        utc_time = []
        for t_str in utc_time_str:
            utc_time.append( dt.datetime.strptime(t_str,'%Y-%m-%dT%H:%M:%S.%f') )
        t0 = dt.datetime( utc_time[0].year, utc_time[0].month, utc_time[0].day )
        time_sec = np.array([(t-t0).total_seconds() for t in utc_time])
        print(time_sec)

        # find indices for the interested time period
        aidx = np.where( (time_sec >= self.start_time_sod - pad_sec) & (time_sec <= self.end_time_sod + pad_sec) )[0]
        norb = len(aidx)
        print(f'Number of records within the time period: {len(aidx)}')
        lat_lon_alt = np.array(df.iloc[aidx,2:5])
        roll = np.array(df.iloc[aidx,17])
        pitch = np.array(df.iloc[aidx,16])
        yaw = np.array(df.iloc[aidx,13])
        euler_angles = np.column_stack((roll,pitch,yaw))
        print(lat_lon_alt.shape,euler_angles.shape) if verbose else None
        
        # now copy to nc file
        grpname = 'er2_navigation_data'
        st_att = 0
        if (norb > 0):
            self.nc_dset[grpname]['orb_time'][st_att:st_att+norb] = time_sec[aidx]
            self.nc_dset[grpname]['orb_lat'][st_att:st_att+norb] = lat_lon_alt[:,0]
            self.nc_dset[grpname]['orb_lon'][st_att:st_att+norb] = lat_lon_alt[:,1]
            self.nc_dset[grpname]['orb_alt'][st_att:st_att+norb] = lat_lon_alt[:,2]
            self.nc_dset[grpname]['att_euler'][st_att:st_att+norb,:] = euler_angles 

        if verbose:
            print('Done copy ER2 data to HARP2 L1A!')
    
    
    def write_img_to_l1a(self, img, img_type, iframe, frameID):
        """
        Write raw UV images to L1A NETCDF4; NO DARK-SUBTRACTION (dark_subtraction occurs in HIPP)
        
        :param img:      2D NumPy array of raw UV image. format: [rows][columns]
        :param img_type: integer (either 0 or 1) indicating if image was taken in science mode (0) or calibration mode (1)
        :param iframe:   integer determining frame index. NOTE: *list* index, not actual frame ID derived from filename
        :param frameID:  integer from image filename determining frame ID
        """
    
        #-- write data to NETCDF4 variables
        self.nc_dset["image_data"]["sensor"][iframe, :, :]        = img;
        self.nc_dset["image_data"]["frame_ID"][iframe, 0]         = frameID;
        self.nc_dset["image_data"]["image_type"][iframe, 0]       = img_type;
        self.nc_dset["image_data"]["integration_time"][iframe, 0] = self.int_time;  # recall that integration time is converted to milliseconds in NC_processor class instance, to match units with polarimeter/for consistency with input to HIPP

        return;


    def write_time_to_nc(self, frame_time, iframe):
        """
        !!! IN PROGRESS !!!
        Write time/frame-based data to L1A NETCDF4
        
        :param frame_time:  datetime object extracted from current frame's image name
        :param iframe:      integer determining frame index. NOTE: *list* index, not actual frame ID derived from filename
        """

        #-- convert to Julian date and seconds of day
        frame_JD   = uv1a.time_util.calc_jd(frame_time);
        frame_secs = uv1a.time_util.calc_sec_of_day(frame_time);

        #-- write data to NETCDF4 variables
        # self.nc_dset["image_data"]["time_in_seconds"][iframe, 0]    = 
        # self.nc_dset["image_data"]["time_in_subseconds"][iframe, 0] = 
        self.nc_dset["image_data"]["JD"][iframe, 0]                 = frame_JD;
        self.nc_dset["image_data"]["seconds_of_day"][iframe, 0]     = frame_secs;


    def move_completed_nc(self):
        """
        Copy completed L1A NetCDF4 file from temporary location to desired location, and remove temporary directory
        """
        
        #-- close NetCDF4 object; otherwise, copied file will NOT contain any data
        self.nc_dset.close();
        
        
        #-- make output directory if it doesn't already exist and copy NetCDF4 from temporary path to output path
        os.makedirs(self.out_dir, exist_ok = True);
        shutil.copy2(self.tmp_fpath, self.fpath);
        
        #-- remove temporary directory
        shutil.rmtree(self.tmp_dir);


#----------------------------------------------------------------------------
#-- FUNCTION DEFINITIONS
#----------------------------------------------------------------------------
def main():
    #-- retrieve script's command-line input arguments and assign to relevant variables
    args            = uv1a.argparse_util.parse_l1agen_ah2uv_args();
    uv_img_dir      = args.uv_input_dir;
    nc_out_dir      = args.nc_output_dir;
    ah2_imu_ncfile  = args.ah2_imu_ncfile;
    start_img       = args.start_img;
    end_img         = args.end_img;
    int_time        = args.int_time;
    er2_infile      = args.er2_infile;
    verbose         = args.verbose;
    

    #-- enumerate UV images
    UVimgs = UV_img_processor(uv_img_dir, start_img, end_img, int_time, verbose);
    
    
    #-- get coverage start and end times. also get as seconds of day as needed for certain functions
    start_time     = dt.datetime.strptime(start_img.split("_")[0], "%Y%m%d%H%M%S");
    end_time       = dt.datetime.strptime(end_img.split("_")[0], "%Y%m%d%H%M%S");
    start_time_sod = uv1a.time_util.calc_sec_of_day(start_time);
    end_time_sod   = uv1a.time_util.calc_sec_of_day(end_time);
    start_time     = start_time.strftime("%Y%m%dT%H%M%S");  # convert start and end times to strings as they are only used as strings after seconds-of-day calculations
    end_time       = end_time.strftime("%Y%m%dT%H%M%S");


    #-- initialize NetCDF4 file object
    fname = f"PACEPAX-AH2UV-L1A_ER2_{start_time}_RA.nc";
    if verbose:
        print();
        print(f"Initializing {fname} NetCDF4...");
    # end if
    
    nc = NC_processor(fname, nc_out_dir, start_time, start_time_sod, end_time, end_time_sod, UVimgs.int_time);
    if verbose:
        print(f"{nc.fname} NetCDF4 initialization successful at temporary location {nc.tmp_fpath}");
    # end if
    
    
    #-- write AH2 IMU and ER2 IMU data to NetCDF4
    if (ah2_imu_ncfile.lower() != "none") and (os.path.exists(ah2_imu_ncfile)):
        nc.copy_ah2_imu_to_l1a(ah2_imu_ncfile, verbose = verbose);
    # end if
    
    if (er2_infile.lower() != "none") and (os.path.exists(er2_infile)):
        nc.copy_er2_imu_to_l1a(er2_infile, verbose = verbose);
    # end if
    
    
    #-- write raw image data to L1A NetCDF4
    # NOTE: using indexing for frame_IDs instead of pulling value directly as actual list index is needed + frameID may be duplicated due to potential acquisition errors during flight
    for img_fpath, iframe in zip(UVimgs.img_finfo["fpaths"], range(len(UVimgs.img_finfo["frame_IDs"]))):
        img      = UVimgs.load_img(img_fpath);
        img_type = 0 if "Group" not in img_fpath else 1;  # 0 = science acquisition, 1 = calibration acquisition
        frameID  = UVimgs.img_finfo["frame_IDs"][iframe];
        nc.write_img_to_l1a(img, img_type, iframe, frameID);
    # end for


    #-- write time/frame-based data to NetCDF4
    if verbose:
        print(f"Writing time/frame-based data to L1A NETCDF4...");
    # end if
    
    for img_fname, iframe in zip(UVimgs.img_finfo["fnames"], range(len(UVimgs.img_finfo["frame_IDs"]))):
        frame_time = dt.datetime.strptime(img_fname.split("_")[0], "%Y%m%d%H%M%S");
        nc.write_time_to_nc(frame_time, iframe);
    # end for
    
    
    #-- copy completed NETCDF4 from temporary directory to true output directory
    nc.move_completed_nc();
    if verbose:
        print("\nAll images in given granule written to L1A NETCDF4.");
        print(f"FIRST IMAGE TIMESTAMP: {start_time}");
        print(f"LAST IMAGE TIMESTAMP: {end_time}");
        print();
        print(f"NETCDF4 successfully copied to output destination {nc.fpath}");
    # end if


if __name__ == "__main__":
    main();
    