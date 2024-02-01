"""
This script generates takes the filtered EEG data and creates epochs of 5 seconds.
The PSD of the epochs is averaged and saved to a dataframe. Eventually, the dataframe
holds n rows for every channel of every NWB file. One NWB file holds data on one subject.

author: Olle, based on work by Vasilis
"""
import os
import mne
import json
import ndx_events
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
from mne.time_frequency import psd_array_multitaper

# Script starts here
if __name__ == '__main__':
    # load settings
    with open('../settings.json', "r") as f:
        settings = json.load(f)
    nwb_folder = settings["nwb_files_folder"]  # path to folder that holds nwb files

    df = pd.DataFrame(columns=['freq', 'psd (means)', 'subject_id', 'genotype', 'channel'])

    # loop through all files (which contain signal for 9 diff channels)
    for i, file in enumerate(os.listdir(nwb_folder)):
        if not file.endswith(".nwb"):
            i += 1
            continue
        print("Loading data from NWB..")
        with NWBHDF5IO(f'{nwb_folder}/{file}', "r") as io:  # open it
            nwb = io.read()

            filtered_eeg = nwb.acquisition['filtered_EEG'].data[:].T  # array of shape (9, 21.xxx.xxx)
            locations = nwb.electrodes.location.data[:]  # get all electrode locations (1-d array)
            s_freq = nwb.acquisition['filtered_EEG'].rate  # sampling frequency of the EEG
            subject_id = nwb.subject.subject_id  # subject id
            genotype = nwb.subject.genotype  # genotype of the subject

            print(f"Data is loaded. Subject id: {subject_id}. Genotype: {genotype}. Sampling frequency: {s_freq}.")

            # generate info object needed for creation of MNE RawArray object
            ch_types = ["emg" if "EMG" in chan else "eeg" for chan in locations]
            info = mne.create_info(ch_names=list(locations), ch_types=ch_types, sfreq=s_freq)

            # create MNE RawArray object and split into fixed length epochs (5 secs)
            simulated_raw = mne.io.RawArray(filtered_eeg, info)
            epochs = mne.make_fixed_length_epochs(simulated_raw, 5.0)

            # calculate the PSD for each EEG channel from 0 through 100 Hz (omits the EMG channels)
            psds, freqs = psd_array_multitaper(epochs.get_data(picks=['eeg']), fmin=0, fmax=100, sfreq=s_freq, n_jobs=-1)
            print(f"PSD object has shape {psds.shape}. Freqs object has shape {freqs.shape}.")
            # psds shape description: (num_epochs, num_chans, num_dp_per_epoch)

            # for every channel for this subject
            for j, chan in enumerate(locations):
                if chan in ['EMG_L', 'EMG_R']:
                    continue
                # calculate the mean psd of the epochs in this channel
                mean_psd = np.mean(psds[:, j, :], axis=0)
                # append the data on this channel to the entire dataframe
                df = pd.concat([df, pd.DataFrame({
                    "freq": freqs,
                    "psd (means)": mean_psd,
                    "subject_id": np.repeat(subject_id, len(freqs)),
                    "genotype": np.repeat(genotype, len(freqs)),
                    "channel": np.repeat(chan, len(freqs))
                })])

            io.close()  # close the file
            i += 1  # increment counter
            print(f"Saved PSD data for subject {subject_id} ({file}). DF now has shape {df.shape}.")
            print(f"Progress: {round(i / len(os.listdir(nwb_folder)) * 100)}% done.")

    df.to_csv(os.path.join(settings['psd_data_folder'], 'psd_averaged_epochs.csv'))
