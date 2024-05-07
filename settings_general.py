"""
This file holds all general settings for the project, and can be imported and used in every script/notebook.
Each experiment has its own folder and accompanying settings file.
"""
general = {
    "lab": "Kas_Lab",
    "experimenter": "Vasilis Siozos & Mirthe Ronde",
    "institution": "University of Groningen"
}

# dictionary holding the subject ids and their batch-cage combination identifiers
subject_id_batch_cage_dict = {
    78211: 'batch1_cage1', 79593: 'batch2_cage1', 81167: 'batch5_cage1', 80620: 'batch4_cage1', 81217: 'batch5b_cage1', 39489: 'batch6_cage1', 80108: 'batch3_cage3',
    78233: 'batch1_cage2', 79592: 'batch2_cage2', 81175: 'batch5_cage2', 80625: 'batch4_cage2', 81218: 'batch5b_cage2', 39508: 'batch6_cage2',
    78227: 'batch1_cage3', 79604: 'batch2_cage3', 81207: 'batch5_cage3', 80630: 'batch4_cage3',
    78244: 'batch1_cage4', 79602: 'batch2_cage4', 81193: 'batch5_cage4'
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

#######################################
## FILTERING / NWB CREATION SETTINGS ##
#######################################

# set to desired sampling frequency or to None if you do not wish to down-sample the EEG data
resample_freq = 1000

# variables used for raw EEG filtering while creating Neurodata Without Border (NWB) files (one of the first steps)
filtering = {
    "lcut": 0.5,  # lower limit of desired band / filter (to be normalized)
    "hcut": 200,  # upper limit of desired band / filter (to be normalized)
    "art": None,  # std of the signal is multiplied by this value to filter out additional artifacts
    "low_val": 0.006,  # lower value of artifact removal (caused by package loss)
    "high_val": 0.013,  # upper value of artifact removal (caused by package loss)
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

###################################
## FRAME BASED EPOCHING SETTINGS ##
###################################

# minimum duration of the interaction between mouse and cup/mouse (in seconds)
min_event_duration = 1.0

# the desired epoch length the events will be divided into
desired_epoch_length = 1.0

# whether to overlap epochs at all
overlap_epochs = False

# the maximum percentage (as a fraction) of information duplication between epochs we allow
# this would mean that with a cutoff of 50% (epoch_overlap_cutoff=0.5), an interaction of duration 1.4 would yield one
# epoch of 1 second (60.0% duplication), i.e. the 0.4 seconds of data is deleted.
epoch_overlap_cutoff = 0.5

####################
## OTHER SETTINGS ##
####################
