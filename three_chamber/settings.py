"""
Settings specific to the analysis of the 3-chamber experiments
"""

# used in additions to the general EEG filtering settings (see settings_general.py at root of project)
# used while creating Neurodata Without Border (NWB) files (one of the first steps)
filtering = {
    'resample_freq': 500  # set to desired sampling frequency or to None if you do not wish to down-sample the EEG data
}

paths_3c_sociability = {
    "metadata": "/Users/olledejong/Documents/MSc_Biology/ResearchProject2/rp2_data/3C_sociability/output/3c_sociability_metadata.xlsx",
    "edf_folder": "/Users/olledejong/Documents/MSc_Biology/ResearchProject2/rp2_data/3C_sociability/input/edf_files",
    "behaviour_data_dir": "/Users/olledejong/Documents/MSc_Biology/ResearchProject2/rp2_data/3C_sociability/input/behavioural_data",
    "plots_folder": "/Users/olledejong/Documents/MSc_Biology/ResearchProject2/rp2_data/3C_sociability/output/plots",
    "recordings_folder": "/Users/olledejong/Documents/MSc_Biology/ResearchProject2/rp2_data/3C_sociability/input/videos",
    "video_analysis_output": "/Users/olledejong/Documents/MSc_Biology/ResearchProject2/rp2_data/3C_sociability/output/videos",
    "nwb_files_folder": "/Users/olledejong/Documents/MSc_Biology/ResearchProject2/rp2_data/3C_sociability/output/nwb"
}

paths_3c_preference = {
    "metadata": "/Users/olledejong/Documents/MSc_Biology/ResearchProject2/rp2_data/3C_preference/output/3c_preference_metadata.xlsx",
    "edf_folder": "/Users/olledejong/Documents/MSc_Biology/ResearchProject2/rp2_data/3C_preference/input/edf_files",
    "behaviour_data_dir": "/Users/olledejong/Documents/MSc_Biology/ResearchProject2/rp2_data/3C_preference/input/behavioural_data",
    "plots_folder": "/Users/olledejong/Documents/MSc_Biology/ResearchProject2/rp2_data/3C_preference/output/plots",
    "nwb_files_folder": "/Users/olledejong/Documents/MSc_Biology/ResearchProject2/rp2_data/3C_preference/output/nwb"
}

