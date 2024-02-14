"""
This file load the NWB and the epoch object of each subject and adds a movement column
to the epoch object for each subject that has movement data. I.e. the epoch objects then contain
(within the metadata) a movement column that indicates whether the subject was moving in that
epoch or not.
"""
import os
import mne
import json
import pickle
import ndx_events
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO


def get_settings(settings_path):
    with open(settings_path, "r") as f:
        return json.load(f)


def frame_to_sample(frame_number, video_fps, offset):
    """
    Function that calculates the time-point of the video (in frames) given
     the sample number in the EEG.
    """
    tp_secs_video = frame_number / video_fps  # time-point on video in seconds
    secs_eeg = tp_secs_video + offset

    return secs_eeg


def sample_to_frame(eeg_sample, video_fps, s_freq, offset):
    """
    Function that calculates the time-point of the video (in frames) given
     the sample number in the EEG.
    """
    eeg_tp_secs = eeg_sample / s_freq  # from samples to seconds
    video_tp_secs = eeg_tp_secs - offset  # subtract the offset so we have the video tp in secs

    return video_tp_secs * video_fps  # go to frames


def get_led_onset_file(pickle_path):
    with open(f'{pickle_path}/led_states_all_videos.pickle', "rb") as f:
        return pickle.load(f)


def get_subject_metadata(metadata_path, mouse_id):
    metadata_df = pd.read_excel(metadata_path)
    return metadata_df[metadata_df['mouseId'] == mouse_id]


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


def get_led_onset_data(video_folder, movie_filename):
    """
    Loads the correct LED onset data from the pickle file which holds the LED
    onset data for all experiment videos.

    :param video_folder:
    :param movie_filename:
    :return:
    """
    print(f"Retrieved accompanying LED state data from file {movie_filename}")
    with open(f'{video_folder}/pickle/led_states_all_videos.pickle', "rb") as f:
        led_states = pickle.load(f)
        led_states = led_states[movie_filename]
        return led_states


def get_movement_per_epoch(epochs_array, first_onset_offset, fps, movement_data, non_movement_cutoff, s_freq):
    """
    Checks whether the subject is moving during each epoch and this information is stored
    in an array. This array is eventually saved within the metadata of the subject's epoch object

    :param epochs_array:
    :param first_onset_offset:
    :param fps:
    :param movement_data:
    :param non_movement_cutoff:
    :param s_freq:
    :return:
    """
    # add new column to metadata holding the movement boolean value (initially we set them all to 'moving'
    movement_col = np.ones(len(epochs_array.metadata), dtype=bool)
    # set the new column to false for the epochs where there's no movement
    for i, epoch in enumerate(epochs_array):
        epoch_start, epoch_end = epochs_array.metadata.iloc[i]["epochs_start_end"].split("-")

        # convert start end end-point of epoch from samples to frames
        frame_start = sample_to_frame(int(epoch_start), fps, s_freq, first_onset_offset)
        frame_end = sample_to_frame(int(epoch_end), fps, s_freq, first_onset_offset)

        # use the start and end in frames to get the accompanying movement data
        frame_start, frame_end = int(np.floor(frame_start)), int(np.ceil(frame_end))

        # if there's no movement in this epoch, update the dataframe
        if np.sum(movement_data[frame_start:frame_end]) == non_movement_cutoff:
            movement_col[i] = False
    return movement_col


def main():
    """
    Backbone of the script. Adds movement data to the epoch objects, concatenates
    all epochs and saves this to the filesystem.
    :return:
    """
    no_spatial_data_ids = [81218, 81217, 39508, 39489]  # from previous analysis we know these do not have xy data
    fps = 30  # frames-per-second of video
    resting_cutoff = 0  # number of frames of movement that is allowed in one epoch

    settings = get_settings("../settings.json")
    nwb_folder, metadata_path = settings["nwb_files_folder"], settings["metadata"]
    epochs_folder, video_folder = settings["epochs_folder"], settings["video_folder"]
    plot_folder = settings["plots_folder"]

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
            eeg_ttl_onsets_secs = list(nwb.acquisition["TTL_1"].timestamps)
            sfreq = nwb.acquisition['raw_EEG'].rate

            try:  # not all nwb files have movement data
                movement_data = nwb.processing["coordinate_data"]["motion"].data[:]
            except KeyError:
                print(f"No movement data for subject {subject_id}, proceeding..")
                continue

        print(f"Subject ID: {subject_id}, Genotype: {genotype}.")

        # load the accompanying filtered epochs file
        epochs_array = get_epoch_array(subject_id, epochs_folder)

        # get the subject's metadata from the file (holds video filename that points to right LED states)
        subject_metadata = get_subject_metadata(metadata_path, int(subject_id))
        movie_filename = subject_metadata["movie_filename"].iloc[0]  # movie fiilename the subject's in

        # get the LED states for this subject (i.e. get the LED states of the correct video)
        led_states = get_led_onset_data(video_folder, movie_filename)
        # calculate the delta (offset) between the start and the first ttl onset for both datastreams
        first_onset_offset = calculate_offset(eeg_ttl_onsets_secs, led_states, sfreq, fps)
        # generate the column that holds the movement data
        movement_col = get_movement_per_epoch(epochs_array, first_onset_offset, fps, movement_data, resting_cutoff, sfreq)

        epochs_array.metadata["movement"] = movement_col  # save the movement column to the subject's metadata
        all_epochs.append(epochs_array)

    # concatenate and save all epochs
    concatenated_epochs = mne.concatenate_epochs(all_epochs, add_offset=True)
    concatenated_epochs.save(os.path.join(epochs_folder, "filtered_epochs_w_movement-epo.fif"))


# script starts here
if __name__ == "__main__":
    main()
