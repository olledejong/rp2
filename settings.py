"""
This file holds all settings for the project, and can be imported and used in every script/notebook
"""
general = {
    "lab": "Kas_Lab",
    "experimenter": "Vasilis Siozos & Mirthe Ronde",
    "institution": "University of Groningen"
}

paths = {
 "edf_folder": "/Users/olledejong/Documents/MSc_Biology/ResearchProject2/rp2_data/resting_state/input/EEG",
 "coordinate_data_folder": "/Users/olledejong/Documents/MSc_Biology/ResearchProject2/rp2_data/resting_state/input/coordinate_data",
 "nwb_files_folder": "/Users/olledejong/Documents/MSc_Biology/ResearchProject2/rp2_data/resting_state/output/nwb",
 "plots_folder": "/Users/olledejong/Documents/MSc_Biology/ResearchProject2/rp2_data/resting_state/output/plots",
 "epochs_folder": "/Users/olledejong/Documents/MSc_Biology/ResearchProject2/rp2_data/resting_state/output/epochs",
 "subject_metadata": "/Users/olledejong/Documents/MSc_Biology/ResearchProject2/rp2_data/DRD2_EEG_all_animal_info.xlsx",
 "metadata": "/Users/olledejong/Documents/MSc_Biology/ResearchProject2/rp2_data/resting_state/output/metadata/resting-state_metadata.xlsx",
 "psd_data_folder": "/Users/olledejong/Documents/MSc_Biology/ResearchProject2/rp2_data/resting_state/output/psd",
 "video_folder": "/Users/olledejong/Documents/MSc_Biology/ResearchProject2/rp2_data/resting_state/output/videos"
}

filtering = {
    "lcut": 0.5,
    "hcut": 200,
    "art": None,
    "low_val": 0.006,
    "high_val": 0.013,
    "electrode_info": {
        "EEG 2": ["OFC_R", 2.7, -0.75, 2.4, "depth"],
        "EEG 3": ["OFC_L", 2.7, 0.75, 2.4, "depth"],
        "EEG 4": ["CG", 1.3, 0.7, 2, "depth"],
        "EEG 13": ["STR_R", 1.3, -1.5, 3, "depth"],
        "EEG 6": ["S1_L", -0.5, 3, 0, "skull"],
        "EEG 11": ["S1_R", -0.5, -3, 0, "skull"],
        "EEG 12": ["V1_R", -4, -2.5, 0, "skull"],
        "EEG 7": ["EMG_L", 0, 0, 0, "emg"],
        "EEG 10": ["EMG_R", 0, 0, 0, "emg"]
    }
}
