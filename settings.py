"""
This file holds all settings for the project, and can be imported and used in every script/notebook
"""
general = {
    "lab": "Kas_Lab",
    "experimenter": "Vasilis Siozos & Mirthe Ronde",
    "institution": "University of Groningen"
}

paths = {
 "edf_folder": "",
 "coordinate_data_folder": "",
 "nwb_files_folder": "",
 "plots_folder": "",
 "epochs_folder": "",
 "subject_metadata": "",
 "metadata": "",
 "psd_data_folder": "",
 "video_folder": ""
}

# variables used for raw EEG filtering while creating Neurodata Without Border (NWB) files (one of the first steps)
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

# EMG frequency bands used for resting-state epoch classification (interpretation not comparable to EEG bands)
freq_bands_emg = {
    'band1': (0, 10), 'band2': (10, 20),
    'band3': (20, 30), 'band4': (30, 40),
    'band5': (40, 50), 'band6': (50, 60),
    'band7': (60, 70), 'band8': (70, 80),
    'band9': (80, 90), 'band10': (90, 100)
}

# EEG frequency bands used throughout the whole project for Power Spectral Density analysis
freq_bands_eeg = {
    r'$\delta$': (1, 4),  # Delta
    r'$\theta$': (4, 8),  # Theta
    r'$\alpha$': (8, 13),  # Alpha
    r'$\beta$': (13, 30),  # Beta
    r'$\gamma$': (30, 100)  # Gamma
}

# dictionary holding the channel names of the EMGs for all subject of which the quality is high enough for analysis
quality_emg = {
    81217: "EMG_L", 81175: "EMG_R", 79592: "EMG_R", 79593: "EMG_L",
    81207: "EMG_R", 80625: "EMG_R", 80630: "EMG_R", 78211: "EMG_R",
    39489: "EMG_L", 80620: "EMG_R", 78227: "EMG_L", 78233: "EMG_R",
    39508: "EMG_R", 79604: "EMG_L", 81218: "EMG_R", 79602: "EMG_L",
    78244: "EMG_R", 81193: "EMG_L",
}