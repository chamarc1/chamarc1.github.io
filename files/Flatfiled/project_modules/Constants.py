"""
__name__ =      Constants.py
__author__ =    "Charlemagne Marc"
__copyright__ = "Copyright 2025, ESI SWIR Project"
__credits__ =   ["Charlemagne Marc"]
__version__ =   "1.0.1"
__maintainer__ ="Charlemagne Marc"
__email__ =     "chamrc1@oumbc.edu"
__status__ =    "Production"
"""

#----------------------------------------------------------------------------
#-- CONSTANTS
#----------------------------------------------------------------------------
composite_save_path = r"/data/home/cmarc/SWIR_Projects/Flatfield/Images/composite.png"
parabola_save_path = r"/data/home/cmarc/SWIR_Projects/Flatfield/Images/along_track.png"
flatfield_save_path = r"/data/home/cmarc/SWIR_Projects/Flatfield/Images/flatfield.png"
flatfield_plot_save_path = r"/data/home/cmarc/SWIR_Projects/Flatfield/Images/flatfield_plot.png"
corrected_composite_save_path = r"/data/home/cmarc/SWIR_Projects/Flatfield/Images/corrected_composite.png"

# Additional save paths for combined plots
composite_all_positions_crosstrack_save_path = r"/data/home/cmarc/SWIR_Projects/Flatfield/Images/composite_all_positions_crosstrack.png"
composite_all_positions_alongtrack_save_path = r"/data/home/cmarc/SWIR_Projects/Flatfield/Images/composite_all_positions_alongtrack.png"
combined_3d_envelopes_save_path = r"/data/home/cmarc/SWIR_Projects/Flatfield/Images/combined_3d_envelopes_all_positions.png"

directory_dict = {
    "crossTrack":  r"/data/home/cjescobar/Projects/AirHARP2/SWIR/raw_data/2025_SWIR_Flatfield/LEFT_RIGHT/",
    "alongTrack" : r"/data/home/cjescobar/Projects/AirHARP2/SWIR/raw_data/2025_SWIR_Flatfield/UP_DOWN/",
    "metadata" : r"/data/ESI/User/cjescobar/Projects/AirHARP2/SWIR/gits/AH2_SWIR_metadata_matcher/202502_SWIR_Flatfield_Matched_Metadata_v2.csv"
}

crossTrack_dict = {
    "pos1" : "20250206_FILTPOS_050_INTTIME_02p0", # 1.57um
    "pos2" : "20250206_FILTPOS_090_INTTIME_01p5", # 1.55um
    "pos3" : "20250206_FILTPOS_130_INTTIME_02p0", # 1.38um
    "pos4" : "20250206_FILTPOS_169_INTTIME_00p2"  # OPEN
}

crossTrackDark_dict = {
    "pos1" : "20250206_FILTPOS_010_INTTIME_02p0",
    "pos2" : "20250206_FILTPOS_010_INTTIME_01p5",
    "pos3" : "20250206_FILTPOS_130_INTTIME_02p0",
    "pos4" : "20250206_FILTPOS_010_INTTIME_00p2"
}

alongTrack_dict = {
    "pos1" : "20250205_FILTPOS_050_INTTIME_02p0",  # 1.57um
    "pos2" : "20250205_FILTPOS_090_INTTIME_01p5",  # 1.55um
    "pos3" : "20250205_FILTPOS_130_INTTIME_02p0",  # 1.38um
    "pos4" : "20250205_FILTPOS_169_INTTIME_00p2",  # OPEN
}

alongTrackDark_dict = {
    "pos1" : "20250205_FILTPOS_010_INTTIME_02p0",
    "pos2" : "20250205_FILTPOS_010_INTTIME_01p5",
    "pos3" : "20250205_FILTPOS_010_INTTIME_02p0",
    "pos4" : "20250205_FILTPOS_010_INTTIME_00p2"
}

def parabola_func(x, constant, linear, quadratic):
    """
    Parabolic function: y = constant + linear*x + quadratic*x^2
    Used for curve fitting.
    """
    return constant + linear * x + quadratic * (x**2)