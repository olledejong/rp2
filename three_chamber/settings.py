"""
Settings specific to the analysis of the 3-chamber experiments
"""

# minimum duration of the interaction between mouse and cup (in seconds)
min_interaction_duration = 1.0

# the desired epoch length the interactions will be divided into
desired_epoch_length = 1.0

# the maximum percentage (as a fraction) of information duplication between epochs we allow
# this would mean that with a cutoff of 50% (epoch_overlap_cutoff=0.5), an interaction of duration 1.4 would yield one
# epoch of 1 second (60.0% duplication), i.e. the 0.4 seconds of data is deleted.
epoch_overlap_cutoff = 0.5
