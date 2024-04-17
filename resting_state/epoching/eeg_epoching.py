"""
This script epochs the EEG data. The result is a raw and a filtered epoch file for every subject.

author: Olle, based on work by Vasilis
"""
import os
import mne
import pickle
import ndx_events
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

from shared.nwb_retrieval_functions import get_filtered_eeg, get_package_loss
from resting_state.settings import paths_resting_state


def get_subject_metadata(metadata_path, subject_id):
    """
    Retrieves the metadata dataframe for the given subject

    :param metadata_path:
    :param subject_id:
    :return:
    """
    metadata_df = pd.read_excel(metadata_path)
    return metadata_df[metadata_df['mouseId'] == subject_id]


def get_led_onset_data(video_analysis_output_dir, movie_filename):
    """
    Loads the correct LED onset data from the pickle file which holds the LED
    onset data for all experiment videos.

    :param video_analysis_output_dir:
    :param movie_filename:
    :return:
    """
    print(f"Retrieving accompanying LED state data from file {movie_filename}")
    with open(f'{video_analysis_output_dir}/pickle/led_states_all_videos.pickle', "rb") as f:
        led_states = pickle.load(f)
        led_states = led_states[movie_filename]
        return led_states


def sample_to_frame(eeg_tp_in_samples, adjusted_fps, s_freq, offset):
    """
    Function that calculates the time-point of the video (in frames) given the sample number in the EEG.

    :param eeg_tp_in_samples:
    :param adjusted_fps:
    :param s_freq:
    :param offset:
    :return:
    """
    eeg_tp_secs = eeg_tp_in_samples / s_freq  # from samples to seconds
    video_tp_secs = eeg_tp_secs - offset  # subtract the offset so we have the video tp in secs

    return video_tp_secs * adjusted_fps  # go to frames


def adjust_fps_get_offset(eeg_signal, subject_id, eeg_onsets, s_freq):
    """
    Adjusts the FPS that is used to go from EEG sample number to Video Frame number.

    :param eeg_signal: one of the channel's signal to calculate the length from first to last eeg ttl onset
    :param subject_id:
    :param eeg_onsets: the timestamps (in seconds) of the EEG TTL pulses
    :param s_freq:
    :return: the adjusted framerate
    """
    metadata_path, video_analysis_output_dir = paths_resting_state["metadata"], paths_resting_state["video_analysis_output"]

    # get the subject's metadata from the file (holds video filename that points to right LED states)
    subject_metadata = get_subject_metadata(metadata_path, int(subject_id))
    movie_filename = subject_metadata["movie_filename"].iloc[0]  # movie filename the subject's in

    # get the LED states for this subject (i.e. get the LED states of the correct video)
    # and then get the frames where the LED turned ON (i.e. get all boolean event changes from OFF to ON (0 to 1)
    led_onsets = get_led_onset_data(video_analysis_output_dir, movie_filename)
    led_onsets = np.where(np.logical_and(np.diff(led_onsets), led_onsets[1:]))[0] + 1

    # find length of eeg signal between the two pulse combination (i.e. the number of samples between the two pulses)
    eeg_len = eeg_signal[int(s_freq * eeg_onsets[0]): int(s_freq * eeg_onsets[-1])].shape[0]
    # find length of video frames between the two pulse combination
    frame_len = led_onsets[-1] - led_onsets[0]

    # divide the number of frames recorded between the two pulses by the seconds that passed between the two EEG TTL
    # pulses, which we get by dividing the number of EEG samples recorded between the two pulses by the sampling freq
    adjusted_fps = (frame_len / (eeg_len / s_freq))

    # calculate the offset in seconds between the first EEG TTL and video LED TTL onset
    first_ttl_onset_secs = eeg_onsets[0] / s_freq  # scale back to seconds
    first_led_onset_secs = led_onsets[0] / adjusted_fps  # scale back to seconds using adjusted FPS
    offset_secs = first_ttl_onset_secs - first_led_onset_secs

    return adjusted_fps, offset_secs


def get_epochs(good_epochs, epochs_per_chan, genotype, info, se_tps_sample, se_tps_frames, subject_id):
    """
    Generates an EpochArray object for the raw and the filtered epochs. Some metadata on the start
    and end time-points of the epochs is also added to the object. This can be of use later.

    :param good_epochs: array holding boolean for each epoch: 1 is good, 0 is bad
    :param epochs_per_chan: dict holding all epochs per channel
    :param genotype:
    :param info:
    :param se_tps_sample: start and end time-point of the epoch in samples
    :param se_tps_frames: start and end time-point of the epoch in frames
    :param subject_id:
    :return:
    """
    se_tps_sample_arr, se_tps_frame_arr = np.array(se_tps_sample), np.array(se_tps_frames)

    # keep only the start-end times of the good epochs
    good_epochs_se_sample = se_tps_sample_arr[good_epochs]
    good_epochs_se_frame = se_tps_frame_arr[good_epochs]

    # generate filtered epochs
    filt_epoch_metadata = pd.DataFrame({
        'animal_id': subject_id,
        'genotype': genotype,
        'epochs_start_end_samples': good_epochs_se_sample,
        'epochs_start_end_frames': good_epochs_se_frame
    })
    # if needed, remove the bad epochs via boolean masking (epoch_annotations is the mask here)
    cleaned_epochs = {channel: epochs_per_chan[channel][good_epochs] for channel in epochs_per_chan.keys()}
    filtered_epochs = mne.EpochsArray(
        data=np.stack(list(cleaned_epochs.values()), axis=1),
        info=info,
        metadata=filt_epoch_metadata
    )

    # generate raw epochs
    raw_epochs_metadata = pd.DataFrame({'animal_id': subject_id, 'genotype': genotype, 'good_epochs': good_epochs})
    raw_epochs = mne.EpochsArray(
        data=np.stack(list(epochs_per_chan.values()), axis=1),
        info=info,
        metadata=raw_epochs_metadata
    )
    return raw_epochs, filtered_epochs


def epoch_eeg_fixed(nwb_file, epoch_length=5.0, ploss_threshold=500):
    """
    Creates epochs of a fixed length for EEG data of all channels and omits bad epochs
    based on a package-loss cutoff value (get_package_loss function). Returns both unfiltered
    and filtered epoch-arrays.

    If last epoch is shorter than 'epoch_length', then it is omitted.

    :param nwb_file: specific nwb file name
    :param epoch_length: desired length of epochs (in seconds)
    :param ploss_threshold: threshold of maximum package loss (in milliseconds)
    :return: raw_epochs and filtered_epochs for this NWB file
    """
    nwb_file_path = os.path.join(paths_resting_state["nwb_files_folder"], nwb_file)
    with NWBHDF5IO(nwb_file_path, "r") as io:
        nwb = io.read()

        filtered_eeg = nwb.acquisition['filtered_EEG'].data[:].T  # array of shape (9, 21.xxx.xxx)
        filtering = nwb.acquisition['filtered_EEG'].filtering
        locations = nwb.electrodes.location.data[:]  # get all electrode locations (1-d array)
        s_freq = nwb.acquisition['filtered_EEG'].rate  # sampling frequency of the EEG
        eeg_ttl_onsets_secs = list(nwb.acquisition["TTL_1"].timestamps)
        subject_id = nwb.subject.subject_id  # subject id
        genotype = nwb.subject.genotype  # genotype of the subject

    # as we noticed the fps is not exactly 30, we have to recalculate it to properly align the EEG and Video
    adjusted_fps, offset = adjust_fps_get_offset(filtered_eeg[0], subject_id, eeg_ttl_onsets_secs, s_freq)

    start_end_tps_s, start_end_tps_f = [], []  # to keep the starting and end-point of the epochs (samples & frames)

    # calculate the amount of samples that are in 1 epoch (samples_per_epoch), and generate a range with the length of
    # the EEG signal with increments of 'samples_per_epoch'
    samples_per_epoch = int(epoch_length * s_freq)
    epoch_start_points = range(0, filtered_eeg.shape[1], samples_per_epoch)

    # array that is later used as a mask for the 'good' epochs
    good_epochs = np.ones(len(epoch_start_points), dtype=bool)

    # create a dictionary in which all the epochs will be stored per channel (and fill it with placeholder arrays)
    epochs_per_chan = {}
    for chan in locations:
        epochs_per_chan[chan] = np.zeros((len(epoch_start_points), samples_per_epoch))

    print("Creating epochs.. (both raw and filtered)")
    # loop through all epochs start points, get the epoch end time-point and create the epochs
    for nth_epoch, epoch_start in enumerate(epoch_start_points):
        # store the start and end time-points in samples for this epoch
        epoch_end = epoch_start + samples_per_epoch
        start_end_tps_s.append(f"{epoch_start}-{epoch_end}")

        # convert the start and end time-points of this epoch from samples to frames
        frame_start = sample_to_frame(int(epoch_start), adjusted_fps, s_freq, offset)
        frame_end = sample_to_frame(int(epoch_end), adjusted_fps, s_freq, offset)
        # and store them as well
        start_end_tps_f.append(f"{frame_start}-{frame_end}")

        # get the filtered eeg belonging to this epoch, and get the package loss in this epoch
        filtered_eeg_epoch = get_filtered_eeg(nwb_file_path, (epoch_start, epoch_end), True)
        ploss, _ = get_package_loss(nwb_file_path, (epoch_start, epoch_end), locations, filtering)

        # loop through the eeg data per channel for this epoch
        for channel, eeg in filtered_eeg_epoch.items():
            # skip epochs that are not of length 'samples_per_epoch'
            if len(eeg) != samples_per_epoch:
                continue
            # add this epoch's eeg data for the looped channel to the dictionary
            epochs_per_chan[channel][nth_epoch] = eeg

            # if there's too much packages loss in this channel, tag this epoch as 'bad'
            if np.sum(np.isnan(ploss[channel])) > int(s_freq * ploss_threshold / 1000):
                good_epochs[nth_epoch] = False

        print('\r', f"{round(nth_epoch / len(epoch_start_points) * 100, 1)}% done..", end='')
    print("\nDone.")

    # generate info object needed for creation of MNE RawArray object
    ch_types = ["emg" if "EMG" in chan else "eeg" for chan in locations]
    info = mne.create_info(ch_names=list(locations), ch_types=ch_types, sfreq=s_freq)

    r_epochs, f_epochs = get_epochs(good_epochs, epochs_per_chan, genotype, info, start_end_tps_s, start_end_tps_f, subject_id)
    print(f"Done. {round(sum(good_epochs) / len(epoch_start_points) * 100, 1)}% of the epochs passed the filtering.")

    return r_epochs, f_epochs


def main():
    """
    Main function of the script. Loops through all NWB files and generates both raw and filtered
    epoch arrays for them. The raw and filtered epochs per subject are saved in individual files.
    These files can later be used to add movement/behaviour data to.
    :return:
    """
    for file in os.listdir(paths_resting_state["nwb_files_folder"]):
        if not file.endswith(".nwb"):
            continue

        # generate the raw and filtered epoch arrays
        raw_epochs, filtered_epochs = epoch_eeg_fixed(file)

        # save the raw and filtered epochs for this subject
        raw_epochs.save(os.path.join(paths_resting_state["epochs_folder"], f'raw_epochs_{file.split(".")[0]}-epo.fif'))
        filtered_epochs.save(os.path.join(paths_resting_state["epochs_folder"], f'filtered_epochs_{file.split(".")[0]}-epo.fif'))

        print(f"Done with file {file}.")
    print("Done with all NWB files.")


if __name__ == '__main__':
    main()
