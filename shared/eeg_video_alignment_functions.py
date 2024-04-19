"""
This file holds general, reusable eeg epoching functions
"""


def get_first_ttl_offset(eeg_ttl_onsets, led_ttl_onsets, adjusted_fps, s_freq):
    """
    Calculates an offset we use to align the EEG and the video data. This is needed because the video recording and the
    EEG recording didn't start at exactly the same moment, so we have to align the two data sources. we calculate the
    offset in seconds between the first EEG TTL and video LED TTL onset.

    :param eeg_ttl_onsets:
    :param led_ttl_onsets:
    :param adjusted_fps:
    :param s_freq:
    :return:
    """

    first_ttl_onset_secs = eeg_ttl_onsets[0] / s_freq  # scale back to seconds
    first_led_onset_secs = led_ttl_onsets[0] / adjusted_fps  # scale back to seconds using adjusted FPS
    offset_secs = first_ttl_onset_secs - first_led_onset_secs

    return offset_secs


def adjust_fps(eeg_signal, eeg_ttl_onsets, led_ttl_onsets, s_freq):
    """
    The experiment videos were recorded in 30 fps, thus, in theory the amount of frames in one second should be 30.
    However, the true framerate of the recordings seems to be lower. Therefore, we need to adjust the fps and know
    the offset to correctly align the behavioural data (time-stamped using frame numbers) and the EEG data.

    We return the adjusted fps, which is used to account for the video lacking behind (not exactly 30 FPS), and we
    return the offset between the first EEG TTL and the first LED ttl, as both recordings didn't start at the exact
    same time. This offset is used to align the data.

    :param eeg_signal:
    :param eeg_ttl_onsets:
    :param led_ttl_onsets:
    :param s_freq:
    :return:
    """
    # find length of eeg signal between the two pulse combination (i.e. the number of samples between the two pulses)
    eeg_len = eeg_signal[int(s_freq * eeg_ttl_onsets[0]): int(s_freq * eeg_ttl_onsets[-1])].shape[0]

    print(f'There are {eeg_len} EEG samples between the first and last TTL pulses, '
          f'which translates to {eeg_len / s_freq} seconds and {eeg_len / s_freq / 60} minutes of data')

    # find length of video frames between the two pulse combination
    frame_len = led_ttl_onsets[-1] - led_ttl_onsets[0]

    print(f'There are {frame_len} frames between the first and last LED pulses, which theoretically equals'
          f' to {frame_len / 30} seconds and {frame_len / 30 / 60} minutes of data')

    # there are fewer frames between the two LED pulses then there should be, so the camera isn't recording at exactly
    # the theoretical 30 frames per second.

    # therefore, we adjust the fps using the time spent between the two EEG TTL pulses (we assume this to be correct)
    # so, we divide the number of EEG samples recorded between the two pulses by the sampling freq to get the time spent
    # in seconds between those pulses, and we then calculate the true FPS by dividing the recorded frames by this value
    adjusted_fps = frame_len / (eeg_len / s_freq)

    print(f'Adjusted FPS: {adjusted_fps}. Total frames / adjusted_fps = {frame_len / adjusted_fps} seconds. '
          f'That value should be equal to EEG samples / sampling_frequency: {eeg_len / s_freq}.')

    return adjusted_fps
