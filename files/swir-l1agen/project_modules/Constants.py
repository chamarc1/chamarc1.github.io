"""
__name__ =      Constants.py
__author__ =    "Charlemagne Marc"
__copyright__ = "Copyright 2025, ESI SWIR Project"
__credits__ =   ["Charlemagne Marc"]
__version__ =   "1.5.5"
__maintainer__ ="Charlemagne Marc"
__email__ =     "chamrc1@umbc.edu"
__status__ =    "Production"
"""

import datetime as dt

#----------------------------------------------------------------------------
#-- CONSTANTS
#----------------------------------------------------------------------------

#-- Version of SWIR_L1AGen package (adapted from HARP2_L1AGen)
VERSION = '1.5.5'  # Updated 2026-01-22

#-- Time epoch constants
EPOCH_TAI = dt.datetime(year=1970, month=1, day=1)  # For AirHARP2, use UNIX epoch
EPOCH_UNIX = dt.datetime(year=1970, month=1, day=1)
TAI2UTC = 0  # For AirHARP2, no leap second offset

#-- Fill values for NetCDF4 variables
FILL_39 = -999.
FILL_R8 = -9999999.
FILL_R4 = -9999999.
FILL_I4 = -9999999
FILL_I2 = 32767
FILL_U2 = 65535
FILL_STR = ''

#----------------------------------------------------------------------------
#-- TIME UTILITY FUNCTIONS
#----------------------------------------------------------------------------

def GetJD(dt_obj):
    """
    Gets the proper julian date since the start of the julian calendar down to
    microsecond resolution.
    
    :param dt_obj: Python datetime object
    :return: Julian date
    """
    t = dt_obj
    jd_j2000 = 2451545.0  # Julian for 1/1/2000
    t -= dt.timedelta(hours=12)  # Julian time is 12 hours lagging behind UTC time
    year = t.year
    nleap = int((year-1-2000)/4+1)
    numdays = nleap*366+365*(year-2000-nleap)
    myjul = jd_j2000+numdays+int(dt.datetime.strftime(t-dt.timedelta(days=1), '%j'))
    myjul += t.hour/24+t.minute/60/24+t.second/3600/24+t.microsecond*1e-6/3600/24
    return myjul


def calculate_seconds(time):
    """
    Convert a Python time object into seconds of day and Julian Day
    
    :param time: Python datetime object
    :return: Tuple of (seconds_of_day, julian_date)
    """
    seconds_of_day = time.hour*3600 + time.minute*60 + time.second + time.microsecond*1e-6
    jdate = GetJD(time)
    return (seconds_of_day, jdate)

#----------------------------------------------------------------------------
#-- FLIGHT DATA PATHS
#----------------------------------------------------------------------------
# flight data parent path to SWIR images
flightData_20240904_SWIR_IMG_DIR = r"/data/archive/ESI/AirHARP2/PACE-PAX_Campaign/PACE-PAX_FlightData_20240904/SWIR"
flightData_20240906_SWIR_IMG_DIR = r"/data/archive/ESI/AirHARP2/PACE-PAX_Campaign/PACE-PAX_FlightData_20240906/SWIR"

# no metadata file
flightData_20240915_SWIR_IMG_DIR = r"/data/archive/ESI/AirHARP2/PACE-PAX_Campaign/PACE-PAX_FlightData_20240915/SWIR"

flightData_20240917_SWIR_IMG_DIR = r"/data/archive/ESI/AirHARP2/PACE-PAX_Campaign/PACE-PAX_FlightData_20240917/SWIR"
flightData_20240919_SWIR_IMG_DIR = r"/data/archive/ESI/AirHARP2/PACE-PAX_Campaign/PACE-PAX_FlightData_20240919/SWIR"
flightData_20240922_SWIR_IMG_DIR = r"/data/archive/ESI/AirHARP2/PACE-PAX_Campaign/PACE-PAX_FlightData_20240922/SWIR"
flightData_20240923_SWIR_IMG_DIR = r"/data/archive/ESI/AirHARP2/PACE-PAX_Campaign/PACE-PAX_FlightData_20240923/SWIR"
flightData_20240926_SWIR_IMG_DIR = r"/data/archive/ESI/AirHARP2/PACE-PAX_Campaign/PACE-PAX_FlightData_20240926/SWIR"
flightData_20240927_SWIR_IMG_DIR = r"/data/archive/ESI/AirHARP2/PACE-PAX_Campaign/PACE-PAX_FlightData_20240927/SWIR"


SWIR_METADATA_DIR  = r"/data/archive/ESI/AirHARP2/PACE-PAX_Campaign/PACE-PAX_FlightData_20240904/SWIR";  # parent path to metadata files
MATCHED_OUTPUT_DIR = r"/data/home/cmarc/SWIR_Projects/SWIR_L1A_Generation/metadata_matcher";  # path to outputed csv containing matched data

# save paths for netCDF4 nc files
nc_files_save_path = r"/data/home/cmarc/SWIR_Projects/SWIR_L1A_Generation/nc_files" # path to save path containing nc files 