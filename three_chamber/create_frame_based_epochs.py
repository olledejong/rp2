import pickle
import ndx_events
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

from shared.helper_functions import *
from settings import subject_id_dict, min_interaction_duration
from shared.eeg_video_alignment_functions import adjust_fps, get_first_ttl_offset


def merge_event_rows(beh_data):
    merged_df = pd.concat([
        beh_data.iloc[::2].reset_index(drop=True),  # only keep each start row
        beh_data.iloc[::2].reset_index(drop=True)['Image index'].rename('Frame start'),  # interaction start frame
        beh_data.iloc[1::2].reset_index(drop=True)['Image index'].rename('Frame stop'),  # interaction stop frame
        beh_data.iloc[1::2].reset_index(drop=True)['Time'] - beh_data.iloc[::2]['Time'].reset_index(drop=True),  # duration
    ], axis=1)
    # rename the last column as it represents the duration of the interaction
    merged_df = merged_df.set_axis([*merged_df.columns[:-1], 'Interaction duration'], axis=1)
    # drop the columns we don't need
    cols_to_drop = [
        'Image index', 'Time', 'Observation type', 'Source', 'Time offset (s)', 'Subject', 'Comment', 'Image file path',
        'Description', 'Behavioral category', 'Behavior type'
    ]
    return merged_df.drop(columns=cols_to_drop)


def get_led_onsets(video_analysis_dir, batch_cage_name):
    """
    Loads the correct LED onset data from the pickle file which holds the LED
    onset data for all experiment videos.

    :param video_analysis_dir:
    :param batch_cage_name: cage_batch combi name that we want the led-states for
    :return:
    """
    with open(os.path.join(video_analysis_dir, 'pickle/led_states.pickle'), "rb") as f:
        led_states = pickle.load(f)
        batch, cage = batch_cage_name.split('_')
        movie_filename = [filename for filename in led_states.keys() if batch in filename and cage in filename]

        # if there's a repeat file for this subject, use that instead of the original file
        if len(movie_filename) != 1:
            movie_filename = [movie_filename for movie_filename in movie_filename if 'repeat' in movie_filename]

        led_states = led_states[movie_filename[0]]
        led_onsets = np.where(np.logical_and(np.diff(led_states), led_states[1:]))[0] + 1
        return led_onsets


def frame_to_sample(video_frame, adjusted_fps, offset, s_freq):
    """
    Function that calculates the EEG sample from the video frame using the adjusted FPS and the calculated offset

    :param video_frame: frame in video that needs to be transformed to EEG sample
    :param adjusted_fps: adjusted FPS (see adjust_fps_get_offset function)
    :param s_freq: EEG sampling frequency
    :param offset:
    :return:
    """
    # go from video frame to seconds
    video_tp_secs = video_frame / adjusted_fps

    # first ttl onset always later in video than in EEG, so to go from video tp in seconds to the eeg tp in seconds
    # we subtract the absolute offset between the two TTLs
    eeg_tp_secs = video_tp_secs - np.abs(offset)

    return eeg_tp_secs * s_freq  # go to samples


def get_epochs(beh_data_subset, eeg_signal, adjusted_video_fps, offset, s_freq):
    """
    Generates frame based epochs using the frame-timestamped recorded behaviours in 'beh_data_subset'.
    We use the adjusted FPS and the offset between the two first TTL onsets to go from the frame timestamps
    (start/stop) to the correct (start/stop) sample in the EEG.

    :param beh_data_subset:
    :param eeg_signal:
    :param adjusted_video_fps:
    :param offset:
    :param s_freq:
    :return:
    """
    raw_epochs, filtered_epochs = [], []

    for index, event in beh_data_subset.iterrows():

        start_frame, stop_frame = int(event['Frame start']), int(event['Frame stop'])
        print(start_frame, stop_frame)
        start_sample = np.floor(frame_to_sample(start_frame, adjusted_video_fps, offset, s_freq))
        stop_sample = np.ceil(frame_to_sample(stop_frame, adjusted_video_fps, offset, s_freq))
        print(start_sample, stop_sample)

    return raw_epochs, filtered_epochs


def main():
    # nwb_folder = select_folder("Select the folder holding your 3-chamber experiment NWB files")
    # experiment_metadata = select_file("Select the experiment's metadata file")
    # behaviour_data = select_folder("Select the experiment's behaviour data folder")
    # video_analysis_folder = select_folder("Select the folder that holds the video analysis output (ROI df, pickle folder, etc)")

    nwb_folder = "/Users/olledejong/Documents/MSc_Biology/ResearchProject2/rp2_data/3C_sociability/output/nwb"
    experiment_metadata = "/Users/olledejong/Documents/MSc_Biology/ResearchProject2/rp2_data/3C_sociability/output/3c_sociability_metadata.xlsx"
    behaviour_data = "/Users/olledejong/Documents/MSc_Biology/ResearchProject2/rp2_data/3C_sociability/input/behavioural_data"
    video_analysis_folder = "/Users/olledejong/Documents/MSc_Biology/ResearchProject2/rp2_data/3C_sociability/output/videos"

    experiment_meta = pd.read_excel(experiment_metadata)

    # create a raw and filtered epochs file per subject (nwb)
    for file in sorted(os.listdir(nwb_folder)):
        if not file.endswith(".nwb"):
            continue

        with NWBHDF5IO(os.path.join(nwb_folder, file), "r") as io:
            nwb = io.read()

            filtered_eeg = nwb.acquisition['filtered_EEG'].data[:].T
            s_freq = nwb.acquisition['filtered_EEG'].rate  # sampling frequency of the EEG
            eeg_ttl_onsets_secs = list(nwb.acquisition["TTL_1"].timestamps)
            subject_id = nwb.subject.subject_id  # subject id

        # get the batch_cage combination name to retrieve the correct behaviour data
        batch_cage = [key for key, value in subject_id_dict.items() if value == int(subject_id)][0]
        print(f'Getting {batch_cage}.xlsx file belonging to subject {subject_id}')

        # load the behavioural data and then merge start/stop events
        # tracking data from BORIS software has 2 rows for each state event (start/stop), we want one for each
        beh_data = pd.read_excel(os.path.join(behaviour_data, f'{batch_cage}.xlsx'))
        beh_data = merge_event_rows(beh_data)

        print(f'Number of durations longer than {min_interaction_duration} seconds:'
              f' {len(beh_data[beh_data["Interaction duration"] > min_interaction_duration])}\n')

        beh_data_subset = beh_data[beh_data["Interaction duration"] > min_interaction_duration]

        # get the LED states for this subject (i.e. get the LED states of the correct video)
        # and then get the frames where the LED turned ON (i.e. get all boolean event changes from OFF to ON (0 to 1)
        led_onsets = get_led_onsets(video_analysis_folder, batch_cage)

        # as the video isn't recorded at exactly 30 fps, we calculate the true fps
        # the offset here is the difference in TTL onset between the EEG and LED (negative means that LED has delay)
        adjusted_fps = adjust_fps(filtered_eeg[0], eeg_ttl_onsets_secs, led_onsets, s_freq)
        first_ttl_offset = get_first_ttl_offset(eeg_ttl_onsets_secs, led_onsets, adjusted_fps, s_freq)

        raw_epochs, filtered_epochs = get_epochs(beh_data_subset, filtered_eeg[0], adjusted_fps, first_ttl_offset, s_freq)
        break


if __name__ == '__main__':
    main()
    print('Done')
