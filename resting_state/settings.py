"""
Settings specific to the resting-state experiment
"""
paths_resting_state = {
    "edf_folder": "",
    "coordinate_data_folder": "",
    "nwb_files_folder": "",
    "plots_folder": "",
    "epochs_folder": "",
    "metadata": "",
    "psd_data_folder": "",
    "recordings_folder": "",
    "video_analysis_output": ""
}

movie_subjects_dict = {
    78211: 'drd2_batch1_resting-state Camera 1 7-7-2023 09_40_22 1.mp4',
    78233: 'drd2_batch1_resting-state Camera 1 7-7-2023 09_40_22 1.mp4',
    78227: 'drd2_batch1_resting-state Camera 1 7-7-2023 09_40_22 1.mp4',
    78244: 'drd2_batch1_resting-state Camera 1 7-7-2023 09_40_22 1.mp4',
    79593: 'drd2_batch2_resting-state Camera 1 18-8-2023 10_00_03 1.mp4',
    79592: 'drd2_batch2_resting-state Camera 1 18-8-2023 10_00_03 1.mp4',
    79604: 'drd2_batch2_resting-state Camera 1 18-8-2023 10_00_03 1.mp4',
    79602: 'drd2_batch2_resting-state Camera 1 18-8-2023 10_00_03 1.mp4',
    81167: 'drd2_batch5_resting-state Camera 1 10-11-2023 09_44_15 1.mp4',
    81175: 'drd2_batch5_resting-state Camera 1 10-11-2023 09_44_15 1.mp4',
    81207: 'drd2_batch5_resting-state Camera 1 10-11-2023 09_44_15 1.mp4',
    81193: 'drd2_batch5_resting-state Camera 1 10-11-2023 09_44_15 1.mp4',
    80620: 'drd2_batch4_resting-state Camera 1 13-10-2023 09_29_44 1.mp4',
    80625: 'drd2_batch4_resting-state Camera 1 13-10-2023 09_29_44 1.mp4',
    80630: 'drd2_batch4_resting-state Camera 1 13-10-2023 09_29_44 1.mp4',
    81217: 'drd2_batch5b_resting-state Camera 1 8-12-2023 09_42_26 1.mp4',
    81218: 'drd2_batch5b_resting-state Camera 1 8-12-2023 09_42_26 1.mp4',
    39489: 'drd2_batch6_resting-state Camera 1 23-12-2023 09_46_45 1.mp4',
    39508: 'drd2_batch6_resting-state Camera 1 23-12-2023 09_46_45 1.mp4'
}

# EMG frequency bands used for resting-state epoch classification (interpretation not comparable to EEG bands)
freq_bands_emg = {
    'band1': (0, 10), 'band2': (10, 20),
    'band3': (20, 30), 'band4': (30, 40),
    'band5': (40, 50), 'band6': (50, 60),
    'band7': (60, 70), 'band8': (70, 80),
    'band9': (80, 90), 'band10': (90, 100)
}

# dictionary holding the channel names of the EMGs for all subject of which the quality is high enough for analysis
quality_emg = {
    81217: "EMG_L", 81175: "EMG_R", 79592: "EMG_R", 79593: "EMG_L",
    81207: "EMG_R", 80625: "EMG_R", 80630: "EMG_R", 78211: "EMG_R",
    39489: "EMG_L", 80620: "EMG_R", 78227: "EMG_L", 78233: "EMG_R",
    39508: "EMG_R", 79604: "EMG_L", 81218: "EMG_R", 79602: "EMG_L",
    78244: "EMG_R", 81193: "EMG_L",
}

# this dictionary holds the labels that belong to the cluster names (0, 1, 2) for each subject
# these were determined by visually inspecting the clustering output of the 'identify_rest_epochs' script
cluster_annotations = {
    39489: {'rest': 2, 'sleep': 1, 'active': 0},
    39508: {'rest': 1, 'sleep': 0, 'active': 2},
    78211: {'rest': 0, 'sleep': 1, 'active': 2},
    78227: {'rest': 1, 'sleep': 0, 'active': 2},
    79592: {'rest': 1, 'sleep': 2, 'active': 0},
    79593: {'rest': 0, 'sleep': 2, 'active': 1},
    79602: {'rest': 0, 'sleep': 2, 'active': 1},
    79604: {'rest': 1, 'sleep': 2, 'active': 0},
    80620: {'rest': 0, 'sleep': 1, 'active': 2},
    80625: {'rest': 2, 'sleep': 0, 'active': 1},
    80630: {'rest': 2, 'sleep': 1, 'active': 0},
    81175: {'rest': 0, 'sleep': 2, 'active': 1},
    81193: {'rest': 2, 'sleep': 0, 'active': 1},
    81207: {'rest': 2, 'sleep': 0, 'active': 1},
    81217: {'rest': 1, 'sleep': 2, 'active': 0},
    81218: {'rest': 1, 'sleep': 2, 'active': 0},
}

# for two animals the clustering results were inconclusive, interestingly both from batch 1
# they are omitted from power/connectivity/other downstream analysis
omitted_after_clustering = [78233, 78244]
# list of animal ids that are excluded from the resting-state analysis
omitted_other = [80108]  # way to high sampling frequency
