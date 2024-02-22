"""
This file load the NWB and the epoch object of each subject and adds a movement column
to the epoch object for each subject that has movement data. I.e. the epoch objects then contain
(within the metadata) a movement column that indicates whether the subject was moving in that
epoch or not.
"""
import os
import mne
import json
import ndx_events
import numpy as np
from pynwb import NWBHDF5IO


def calculate_offset(eeg_onsets, led_onsets, s_freq, video_fps):
    """
    Calculates the difference in time between the video and eeg that elapses from
    the start of the recording until the first onset (LED/TTL)

    :param eeg_onsets:
    :param led_onsets:
    :param s_freq:
    :param video_fps:
    :return:
    """
    # get the frames where the LED turned ON (i.e. get all boolean event changes from OFF to ON (0 to 1)
    led_turns_on_frames = np.where(np.logical_and(np.diff(led_onsets), led_onsets[1:]))[0] + 1

    first_ttl_onset = eeg_onsets[0]
    first_led_onset = led_turns_on_frames[0]

    first_ttl_onset_secs = first_ttl_onset / s_freq
    first_led_onset_secs = first_led_onset / video_fps

    return first_ttl_onset_secs - first_led_onset_secs


def get_epoch_array(subject_id, epochs_folder):
    """
    Loads the epoch array that belongs to the given subject.

    :param subject_id:
    :param epochs_folder:
    :return:
    """
    for file in os.listdir(epochs_folder):
        if file.startswith("filtered") and file.endswith(f"{subject_id}-epo.fif"):
            epochs_array = mne.read_epochs(os.path.join(epochs_folder, file), preload=True)
            return epochs_array


def get_movement_per_epoch(epochs_array, movement_data, non_movement_cutoff):
    """
    Checks whether the subject is moving during each epoch and this information is stored
    in an array. This array is eventually saved within the metadata of the subject's epoch object

    :param epochs_array:
    :param movement_data:
    :param non_movement_cutoff: integer that defines the cutoff for 'non-movement'
    :return:
    """
    # add new column to metadata holding the movement boolean value (initially we set them all to 'moving'
    movement_col = np.ones(len(epochs_array.metadata), dtype=bool)
    # set the new column to false for the epochs where there's no movement
    for i, epoch in enumerate(epochs_array):
        epoch_start_frame, epoch_end_frame = epochs_array.metadata.iloc[i]["epochs_start_end_frames"].split("-")

        # use the start and end in frames to get the accompanying movement data
        frame_start, frame_end = int(np.floor(epoch_start_frame)), int(np.ceil(epoch_end_frame))

        # if there's no/not so much movement in this epoch, tag this epoch as 'not moving'
        if np.sum(movement_data[frame_start:frame_end]) <= non_movement_cutoff:
            movement_col[i] = False
    return movement_col


def main():
    """
    Backbone of the script. Adds movement data to the epoch objects, concatenates
    all epochs and saves this to the filesystem.
    :return:
    """
    resting_cutoff = 0  # number of frames of movement that is allowed in one epoch

    with open("../settings.json", "r") as f:
        settings = json.load(f)
    nwb_folder, epochs_folder = settings["nwb_files_folder"], settings["epochs_folder"]

    all_epochs = []
    # loop through the nwb files (1 for each subject)
    for nwb_file in os.listdir(nwb_folder):
        if not nwb_file.endswith(".nwb"):
            continue
        print(f"Working with file {nwb_file}.")

        # with the opened nwb file, get needed info
        with NWBHDF5IO(f'{nwb_folder}/{nwb_file}', "r") as io:  # open it
            nwb = io.read()
            subject_id = nwb.subject.subject_id
            genotype = nwb.subject.genotype

            try:  # not all nwb files have movement data
                movement_data = nwb.processing["coordinate_data"]["motion"].data[:]
            except KeyError:
                print(f"No movement data for subject {subject_id}, proceeding..")
                continue

        print(f"Subject ID: {subject_id}, Genotype: {genotype}.")

        # load the accompanying filtered epochs file
        epochs_array = get_epoch_array(subject_id, epochs_folder)
        # generate the column that holds the movement data
        movement_col = get_movement_per_epoch(epochs_array, movement_data, resting_cutoff)

        # save the movement column to the subject's metadata
        epochs_array.metadata["movement"] = movement_col
        all_epochs.append(epochs_array)

        epochs_array.save(os.path.join(epochs_folder, f"filtered_epochs_w_movement_{subject_id}-epo.fif"))

    print("Saved epochs of each subject in individual file, now attempting to concatenate them as well..")
    # concatenate and save all epochs
    concatenated_epochs = mne.concatenate_epochs(all_epochs, add_offset=True)
    concatenated_epochs.save(os.path.join(epochs_folder, "filtered_epochs_w_movement-epo.fif"))
    print("Done, bye.")


# script starts here
if __name__ == "__main__":
    main()
