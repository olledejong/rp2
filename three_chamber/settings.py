"""
Settings specific to the analysis of the 3-chamber experiments
"""

resample_freq = 500  # set to desired sampling frequency or to None if you do not wish to down-sample the EEG data

# dictionary holding the subject id that belongs to the batch-cage combinations
subject_id_dict = {
    'batch1_cage1': 78211, 'batch2_cage1': 79593, 'batch5_cage1': 81167, 'batch4_cage1': 80620, 'batch5b_cage1': 81217, 'batch6_cage1': 39489,
    'batch1_cage2': 78233, 'batch2_cage2': 79592, 'batch5_cage2': 81175, 'batch4_cage2': 80625, 'batch5b_cage2': 81218, 'batch6_cage2': 39508,
    'batch1_cage3': 78227, 'batch2_cage3': 79604, 'batch5_cage3': 81207, 'batch4_cage3': 80630,
    'batch1_cage4': 78244, 'batch2_cage4': 79602, 'batch5_cage4': 81193,
}

# minimum duration of the interaction between mouse and cup (in seconds)
min_interaction_duration = 1.0

# the desired epoch length the interactions will be divided into
desired_epoch_length = 1.0

# the maximum percentage (as a fraction) of information duplication between epochs we allow
# this would mean that with a cutoff of 50% (epoch_overlap_cutoff=0.5), an interaction of duration 1.4 would yield one
# epoch of 1 second (60.0% duplication), i.e. the 0.4 seconds of data is deleted.
epoch_overlap_cutoff = 0.5
