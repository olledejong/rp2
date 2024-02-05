"""
File that holds functions that can be used to retrieve data from a NWB file given its filename
"""
import re
import numpy as np
from pynwb import NWBHDF5IO


def get_filtered_eeg(nwb_file_path, segment=(0, -1), channel_names=True):
    """
    Retrieves filtered EEG data from a nwb_file (either complete or based on segment)
    Returns array (1D-array with filtered EEG samples)
    or
    dict:
        keys: electrode brain locations
        values: 1D-array with filtered EEG samples,
    """
    with NWBHDF5IO(nwb_file_path, "r") as io:
        nwb = io.read()
        filtered_eeg = nwb.acquisition['filtered_EEG'].data[segment[0]: segment[1]].T
        if not channel_names:
            return filtered_eeg

        # if channel names (e.g. OFC_R, etc) are requested
        channel_data = {}
        locations = nwb.electrodes.location.data[:]

        for signal, location in zip(filtered_eeg, locations):
            channel_data[location] = signal

        return channel_data


def get_package_loss(nwb_filepath, segment, locations, filtering):
    """
    Retrieves the raw EEG from the NWB file, searches and returns package loss information
    in two forms:
        - ploss_signal : raw signal containing np.nan values (without interpolating)
        - ploss_samples : sample indexes where there is package loss (more useful)
    """
    with NWBHDF5IO(nwb_filepath, "r") as io:
        nwb = io.read()

        # Parse filtering info
        f_info = re.search('low_val:(.+),.+high_val:(.+),.+art:(.+)', filtering)
        low_val, high_val, art = float(f_info[1]), float(f_info[2]), f_info[3]

        # take the data from the raw eeg that corresponds to this epoch
        raw_eeg_seg = nwb.acquisition['raw_EEG'].data[segment[0]: segment[1]].T

        ploss_signal, ploss_samples = {}, {}  # to store data in
        for signal, location in zip(raw_eeg_seg, locations):
            rej = np.where(signal > low_val, signal, np.nan)
            rej = np.where(signal < high_val, rej, np.nan)
            if art != 'None':
                art = float(f_info[3])
                rej = np.where((rej > np.mean(rej) + art*np.std(rej)) | (rej < np.mean(rej) - art*np.std(rej)), np.nan, rej)
            ploss_signal[location] = rej
            ploss_samples[location] = np.where(np.isnan(rej))[0]

    return ploss_signal, ploss_samples