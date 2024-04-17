"""
This file holds all general settings for the project, and can be imported and used in every script/notebook.
Each experiment has its own folder and accompanying settings file.
"""
general = {
    "lab": "Kas_Lab",
    "experimenter": "Vasilis Siozos & Mirthe Ronde",
    "institution": "University of Groningen"
}

paths_general = {
    "all_animal_metadata": "/Users/olledejong/Documents/MSc_Biology/ResearchProject2/rp2_data/DRD2_EEG_all_animal_info.xlsx",
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

# EEG frequency bands used throughout the whole project for Power Spectral Density analysis
freq_bands_eeg = {
    r'$\delta$': (1, 4),  # Delta
    r'$\theta$': (4, 8),  # Theta
    r'$\alpha$': (8, 13),  # Alpha
    r'$\beta$': (13, 30),  # Beta
    r'$\gamma$': (30, 100)  # Gamma
}

# after inspecting the power traces of all subjects, these channels were noted as having bad quality
low_qual_chans = {
    39489: ["OFC_R"],
    80625: ["OFC_L"],
    81193: ["OFC_R", "OFC_L"]
}
