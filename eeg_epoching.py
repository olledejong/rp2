"""
This script epochs the EEG data in various ways. For now only fixed length epochs are supported.

author: Olle, based on work by Vasilis
"""
import os
import mne
import json
import ndx_events
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

from nwb_retrieval_functions import get_filtered_eeg, get_package_loss


def epoch_eeg_fixed(nwb_file, epoch_length=5.0, relative_start=0, ploss_threshold=10):
    """
    Creates epochs of a fixed length for EEG data of all channels and omits bad epochs
    based on a package-loss cutoff value (get_package_loss function). Returns both unfiltered
    and filtered epocharrays.

    If last epoch is shorter than 'epoch_length', then it is omitted.

    :param nwb_file: nwb file name
    :param epoch_length: desired length of epochs (in seconds)
    :param relative_start: starting point of epoching (in seconds)
    :param ploss_threshold: threshold of maximum package loss (in milliseconds)
    :return: raw_epochs and filtered_epochs for this NWB file
    """
    with open('settings.json', "r") as f:
        settings = json.load(f)
    nwb_folder = settings["nwb_files_folder"]  # path to folder with nwb files
    epochs_folder = settings["epochs_folder"]  # path to folder with nwb files

    print(f"Epoching data for file {nwb_file}")
    nwb_file_path = os.path.join(nwb_folder, nwb_file)
    with NWBHDF5IO(nwb_file_path, "r") as io:
        nwb = io.read()

        filtered_eeg = nwb.acquisition['filtered_EEG'].data[:].T  # array of shape (9, 21.xxx.xxx)
        filtering = nwb.acquisition['filtered_EEG'].filtering
        locations = nwb.electrodes.location.data[:]  # get all electrode locations (1-d array)
        s_freq = nwb.acquisition['filtered_EEG'].rate  # sampling frequency of the EEG
        subject_id = nwb.subject.subject_id  # subject id
        genotype = nwb.subject.genotype  # genotype of the subject

    print(f"Data is loaded. Subject id: {subject_id}. Genotype: {genotype}. Sampling frequency: {s_freq}.")
    epochs_per_chan = {}  # to store epochs in
    start_end_times = []  # to keep the starting and end-point (in samples) of the epochs
    samples_per_epoch, relative_start = int(epoch_length * s_freq), int(relative_start * s_freq)

    print("Starting with creating epochs and filtering out bad ones.")
    epochs = range(0, filtered_eeg.shape[1], samples_per_epoch)  # with increments of size 'samples_per_epoch'
    epoch_annotations = np.ones(len(epochs), dtype=bool)  # True for good epoch, False for bad epoch

    for nth_epoch, start_sample in enumerate(epochs):  # loop through epochs in n_samples (not seconds)
        epoch_start = start_sample + relative_start
        epoch_end = epoch_start + samples_per_epoch
        start_end_times.append(f"{epoch_start}-{epoch_end}")

        filtered_eeg_epoch = get_filtered_eeg(nwb_file_path, (epoch_start, epoch_end), True)
        ploss, _ = get_package_loss(nwb_file_path, (epoch_start, epoch_end), locations, filtering)

        # loop through the epoched eeg data for per channel
        for location, eeg in filtered_eeg_epoch.items():
            if len(eeg) != samples_per_epoch:  # skip epochs that are not of length 'samples_per_epoch'
                continue
            if location not in epochs_per_chan:  # if not saved
                epochs_per_chan[location] = np.zeros((len(epochs), samples_per_epoch))
            epochs_per_chan[location][nth_epoch] = eeg  # save eeg data

            # if there's too much packages loss in this channel, tag this epoch as 'bad'
            if np.sum(np.isnan(ploss[location])) > int(s_freq * ploss_threshold / 1000):
                epoch_annotations[nth_epoch] = False

        print('\r', f"{round(nth_epoch / len(epochs) * 100, 1)}% done..", end='')
    print("\nDone.")

    # generate info object needed for creation of MNE RawArray object
    ch_types = ["emg" if "EMG" in chan else "eeg" for chan in locations]
    info = mne.create_info(ch_names=list(locations), ch_types=ch_types, sfreq=s_freq)

                                # Save raw epochs but with annotation

    raw_epochs_metadata = pd.DataFrame({
        'animal_id': subject_id,
        'genotype': genotype,
        'epoch_annotation': epoch_annotations
    })
    raw_epochs = mne.EpochsArray(
        data=np.stack(list(epochs_per_chan.values()), axis=1),
        info=info,
        metadata=raw_epochs_metadata
    )
                                    # Save epochs without bad ones

    start_end_times = np.array(start_end_times)
    good_epochs_start_end = start_end_times[epoch_annotations]  # keep only the start-end times of the good epochs

    filt_epoch_metadata = pd.DataFrame({
        'animal_id': subject_id,
        'genotype': genotype,
        'epochs_start_end': good_epochs_start_end
    })

    # if needed, remove the bad epochs via boolean masking (epoch_annotations is the mask here)
    cleaned_epochs = {channel: epochs_per_chan[channel][epoch_annotations] for channel in epochs_per_chan.keys()}

    filtered_epochs = mne.EpochsArray(
        data=np.stack(list(cleaned_epochs.values()), axis=1),
        info=info,
        metadata=filt_epoch_metadata
    )

    print(f"Done. {round(sum(epoch_annotations) / len(epochs) * 100, 1)}% of the epochs passed the filtering.")
    return raw_epochs, filtered_epochs
