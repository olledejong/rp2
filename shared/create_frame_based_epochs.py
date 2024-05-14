import re
import mne
import pickle
import ndx_events
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

from settings_general import *
from shared.helper_functions import *
from shared.nwb_retrieval_functions import get_eeg, get_package_loss
from settings_general import subject_id_batch_cage_dict
from shared.eeg_video_alignment_functions import adjust_fps, get_first_ttl_offset


def merge_event_rows(beh_data):
    merged_df = pd.concat([
        beh_data.iloc[::2].reset_index(drop=True),  # only keep each start row
        beh_data.iloc[::2].reset_index(drop=True)['Image index'].rename('Frame start'),  # event start frame
        beh_data.iloc[1::2].reset_index(drop=True)['Image index'].rename('Frame stop'),  # event stop frame
        beh_data.iloc[1::2].reset_index(drop=True)['Time'] - beh_data.iloc[::2]['Time'].reset_index(drop=True),
        # duration
    ], axis=1)
    # rename the last column as it represents the duration of the event (in seconds)
    merged_df = merged_df.set_axis([*merged_df.columns[:-1], 'Event duration'], axis=1)
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
    with open(os.path.join(video_analysis_dir, 'led_states.pickle'), "rb") as f:
        led_states = pickle.load(f)
        batch, cage = batch_cage_name.split('_')
        movie_filename = [
            fn for fn in led_states.keys() if bool(re.search(f'{batch}_', fn)) and bool(re.search(cage, fn))
        ]

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


def get_epoch_overlap(event):
    """
    Calculates the overlap each fixed epoch of 'desired_epoch_length' needs to have with the previous one in order
    to capture all data that falls within an event.

    Example:
    To calculate the overlap in seconds we determine how many overlaps there are between the supposed epochs
    e.g. we have an event/interaction of duration 2,4 seconds. If we start at 0 and create epochs of 1 second, this
    would give us 3 epochs, however, the last one isn't of the same length (0.4 seconds)
    we then get the overlap needed to capture all data within epochs of 1 second by dividing the amount of seconds
    missing that would have created a full 3rd epoch (0.6 seconds) by the amount of epoch overlaps (3)
    in this case the overlap is thus 0.2 seconds.

    :param event: the interaction/event with start/stop time in frames and event duration
    :return:
    """
    total_duration = event['Event duration']

    # let's calculate the number of epochs of length 'desired_epoch_length' that fit into the event
    # this is also the number of overlaps between all epochs if you would cut it at increments of
    # 'desired_epoch_length'
    num_full_epochs = int(total_duration // desired_epoch_length)

    # get the amount of seconds that would've made the last epoch equal to 'desired_epoch_length'
    missing_seconds = 1 - (total_duration - num_full_epochs)

    # now, if we divide the 'missing_seconds' by the 'num_full_epochs' (i.e. amount of epoch overlaps), we get the
    # needed amount of seconds each epoch has to overlap with the previous one to capture all data
    overlap = missing_seconds / num_full_epochs

    return overlap


def get_eeg_start_end_tps(adjusted_video_fps, event, offset, s_freq):
    """
    Function that gets the start and en end time-points in frames from the event row and processes this to
    the same time-points, but in (eeg) sample. Also handles the cases where the min_event_duration (defined in
    settings_general.py) is smaller than the desired_epoch_length (defined in settings_general.py), or when
    min_event_duration is set to None

    :param adjusted_video_fps:
    :param event:
    :param offset:
    :param s_freq:
    :return:
    """
    # get the start and stop frame time-point of this event
    start_frame, stop_frame = int(event['Frame start']), int(event['Frame stop'])

    # total event duration
    event_duration = event['Event duration']

    # when the min_event_duration is None, we very likely get events that are shorter than the desired_epoch_length
    # however, it is also possible that the event_duration is shorter than the desired epoch length if we DO set the
    # min_event_duration
    # therefore we need to add EEG data on each side of the actual event to get to the desired epoch length.
    if event_duration < desired_epoch_length:
        missing_duration = desired_epoch_length - event_duration  # seconds
        missing_duration_frames = missing_duration * adjusted_video_fps  # in frames

        # get the amount of frames we need to add to each side of the event in order to make it of length
        # 'desired_epoch_length'
        to_subtract_and_add = missing_duration_frames / 2

        start_frame, stop_frame = np.floor(start_frame - to_subtract_and_add), np.ceil(stop_frame + to_subtract_and_add)

    # using the adjusted FPS and the offset of the first TTL, get the start/stop time-points of the event in samples
    event_start = int(frame_to_sample(start_frame, adjusted_video_fps, offset, s_freq))
    event_end = int(frame_to_sample(stop_frame, adjusted_video_fps, offset, s_freq))

    return event_start, event_end


def get_epochs(nwb_file_path, beh_data_subset, adjusted_video_fps, offset, s_freq, subject_id, genotype):
    """
    Generates frame based epochs using the frame-timestamped recorded behaviours in 'beh_data_subset'.
    We use the adjusted FPS and the offset between the two first TTL onsets to go from the frame timestamps
    (start/stop) to the correct (start/stop) sample in the EEG.

    :param nwb_file_path:
    :param beh_data_subset:
    :param adjusted_video_fps:
    :param offset:
    :param s_freq:
    :param genotype:
    :param subject_id:
    :return:
    """
    all_event_epochs = []
    skipped_events = 0

    # loop through all events
    for index, event in beh_data_subset.iterrows():

        event_start, event_end = get_eeg_start_end_tps(adjusted_video_fps, event, offset, s_freq)

        event_eeg, chans = get_eeg(nwb_file_path, 'filtered_EEG', (event_start, event_end), True)

        # do not handle events that have too much package loss
        if not resampled:
            event_duration = event_end - event_start  # in EEG samples
            too_much_package_loss = False

            # get package loss
            ploss, _ = get_package_loss(nwb_file_path, (event_start, event_end))

            # calc total package loss per channel, and if there's too much package loss in a channel, skip this event
            for chan in chans:
                package_loss = np.sum(np.isnan(ploss[chan]))  # for this channel in EEG samples

                too_much_package_loss = True if (package_loss / event_duration) > package_loss_cutoff else False

                # don't need to check them all if we know there's too much in one channel
                if too_much_package_loss:
                    break

            if too_much_package_loss:  # then skip this event
                # print(f'Skipping {event["Behavior"]} event {index} because there is more than '
                #       f'{package_loss_cutoff * 100:.2f}% package loss in one of the channels')
                skipped_events += 1
                continue

        overlap = 0.0
        if overlap_epochs:
            # get overlap each epoch needs to have with preceding one to capture all EEG data in epochs of desired len
            overlap = get_epoch_overlap(event)

            # check if the 'epoch_overlap_cutoff * desired_epoch_length', which is the duration of the epoch length that
            # we allow as overlap, is larger than the calculated overlap needed to capture all data.
            max_allowed_overlap = epoch_overlap_cutoff * desired_epoch_length  # in seconds
            if overlap > max_allowed_overlap:
                overlap = 0.0

        ch_types = ["emg" if "EMG" in chan else "eeg" for chan in chans]
        info = mne.create_info(ch_names=list(chans), sfreq=s_freq, ch_types=ch_types)
        raw = mne.io.RawArray(event_eeg, info)

        # make fixed length epochs of 'desired_epoch_length' length
        epochs = mne.make_fixed_length_epochs(
            raw, duration=desired_epoch_length, overlap=overlap, preload=True
        )

        # create metadata dataframe and add to epochs array
        metadata = pd.DataFrame({
            'subject_id': [subject_id] * len(epochs),
            'genotype': [genotype] * len(epochs),
            'event_n': [index + 1] * len(epochs),
            'event_part_n': range(1, len(epochs) + 1),
            'event_kind': [event["Behavior"]] * len(epochs),
            'total_event_duration': [event["Event duration"]] * len(epochs),
            'epoch_length': [desired_epoch_length] * len(epochs),
        })
        epochs.metadata = metadata

        # save this event's epochs
        all_event_epochs.append(epochs)

    # concatenate all epoch arrays
    all_epochs = mne.concatenate_epochs(all_event_epochs)

    print(f'\nSkipped {skipped_events} events ({skipped_events / len(beh_data_subset) * 100:.2f}%) due to exceeding '
          f'package-loss threshold')

    return all_epochs


def create_cleaned_event_df(beh_data, batch_cage, subject_id, genotype):
    """
    Checks if there are an equal amount of START and STOP events per event type. If not, it will try to fix it,
    if it seems unfixable, it will skip the event in question.

    :param beh_data:
    :param batch_cage:
    :param subject_id:
    :param genotype:
    :return:
    """
    beh_df = pd.DataFrame()

    for event_type in beh_data['Behavior'].unique():
        beh_dat_event = beh_data[beh_data['Behavior'] == event_type]
        starts = beh_dat_event[beh_dat_event['Behavior type'] == 'START']
        stops = beh_dat_event[beh_dat_event['Behavior type'] == 'STOP']

        if len(stops) < len(starts):
            print(f'({batch_cage}, {subject_id}) Number of STOPs is smaller than number of STARTs for {event_type}')
            if beh_dat_event.iloc[-1]['Behavior type'] == 'START':
                print('Removing last row because it is of type START')
                beh_dat_event = beh_dat_event.drop(beh_dat_event.index[-1])
            else:
                print(f'Number of STOPs is smaller than number of STARTs, but this is not caused by a START at '
                      f'the last row of the dataframe. Skipping..')
                continue
        if len(starts) < len(stops):
            print(f'({batch_cage}, {subject_id}) Number of STARTs is smaller than number of STOPs for {event_type}')
            if beh_dat_event.iloc[0]['Behavior type'] == 'STOP':
                print('Removing first row because it is of type STOP')
                beh_dat_event = beh_dat_event.drop(beh_dat_event.index[0])
            else:
                print(f'Number of STARTs is smaller than number of STOPs, but this is not caused by a STOP at '
                      f'the first row of the dataframe. Skipping..')
                continue

        beh_dat_event = merge_event_rows(beh_dat_event)
        # merge the start and stop rows and calculate some stuff (event duration etc)
        beh_dat_event.insert(1, 'subject_id', subject_id)
        beh_dat_event.insert(2, 'genotype', genotype)
        beh_df = pd.concat([beh_df, beh_dat_event], axis=0)

    return beh_df


def main():
    print('Select the folder holding your experiment NWB files')
    nwb_folder = select_folder("Select the folder holding your experiment NWB files")
    print("Select the experiment's behaviour data folder")
    behaviour_data = select_folder("Select the experiment's behaviour data folder")
    print("Select the folder that holds the video analysis output (ROI Excel, pickle file)")
    video_analysis_folder = select_folder(
        "Select the folder that holds the video analysis output (ROI Excel, pickle file)")
    print("Select or create a folder to where the epoch files are saved")
    epochs_folder = select_or_create_folder("Select or create a folder to where the epoch files are saved")

    # create a raw and filtered epochs file per subject (nwb)
    for file in sorted(os.listdir(nwb_folder)):
        if not file.endswith(".nwb"):
            continue

        nwb_file_path = os.path.join(nwb_folder, file)
        with NWBHDF5IO(nwb_file_path, "r") as io:
            nwb = io.read()

            subject_id = nwb.subject.subject_id
            if os.path.exists(os.path.join(epochs_folder, f'epochs_{subject_id}-epo.fif')):
                print(f'Epoch file epochs_{subject_id}-epo.fif already exists in the selected folder. Skipping..')
                continue

            filtered_eeg = nwb.acquisition['filtered_EEG'].data[:].T[0]  # only one channel needed to adjust the fps
            s_freq = nwb.acquisition['filtered_EEG'].rate  # sampling frequency/resampled frequency of the EEG
            eeg_ttl_onsets_secs = list(nwb.acquisition["TTL_1"].timestamps)  # timestamps of the TTL onsets in seconds
            genotype = nwb.subject.genotype
            io.close()

        # get the batch_cage combination name to retrieve the correct behaviour data
        batch_cage = subject_id_batch_cage_dict[int(subject_id)]
        print(f'\nGetting {batch_cage}.xlsx file belonging to subject {subject_id}')
        print(f'Sampling frequency: {s_freq}')

        # load the behavioural data and then merge start/stop events
        # tracking data from BORIS software has 2 rows for each state event (start/stop), we want one for each
        beh_data = pd.read_excel(os.path.join(behaviour_data, f'{batch_cage}.xlsx'))
        beh_data = create_cleaned_event_df(beh_data, batch_cage, subject_id, genotype)

        print(f'Total number of events: {len(beh_data)}.')
        print(f'Number of events per type: {np.unique(beh_data["Behavior"], return_counts=True)}')

        if min_event_duration is not None:
            longer_than = len(beh_data[beh_data["Event duration"] >= min_event_duration])
            shorter_than = len(beh_data[beh_data["Event duration"] < min_event_duration])
            print(f'Percentage of events shorter than {min_event_duration} seconds: '
                  f'{shorter_than / (shorter_than + longer_than) * 100:.2f}%')

            beh_data = beh_data[beh_data["Event duration"] >= min_event_duration]
            beh_data.reset_index(drop=True, inplace=True)

        # get the LED states for this subject (i.e. get the LED states of the correct video)
        # and then get the frames where the LED turned ON (i.e. get all boolean event changes from OFF to ON (0 to 1)
        led_onsets = get_led_onsets(video_analysis_folder, batch_cage)

        # as the video isn't recorded at exactly 30 fps, we calculate the true fps
        # the offset here is the difference in TTL onset between the EEG and LED (negative means that LED has delay)
        adjusted_fps = adjust_fps(filtered_eeg, eeg_ttl_onsets_secs, led_onsets, s_freq, verbose=False)
        first_ttl_offset = get_first_ttl_offset(eeg_ttl_onsets_secs, led_onsets, adjusted_fps, s_freq)

        # generate fixed length epochs
        all_epochs = get_epochs(nwb_file_path, beh_data, adjusted_fps, first_ttl_offset, s_freq, subject_id,
                                genotype)

        # save this subject's epochs
        all_epochs.save(os.path.join(epochs_folder, f'epochs_{subject_id}-epo.fif'), overwrite=True)
        print(f'\nSuccessfully created and saved {len(all_epochs)} epochs for subject {subject_id}.')


if __name__ == '__main__':
    main()
    print('\nDone!')
