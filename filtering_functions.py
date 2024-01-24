"""
Holds functions that are utilized to filter the raw EEG signal
"""
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d


def interpolate_nan(padata, pkind='linear'):
    """
    Interpolates NaNs and returns this interpolated array

    :param padata: data to be interpolated
    :param pkind: kind of interpolation
    :return:
    """
    aindexes = np.arange(padata.shape[0])
    agood_indexes, = np.where(np.isfinite(padata))
    f = interp1d(agood_indexes,
                 padata[agood_indexes],
                 bounds_error=False,
                 copy=False,
                 fill_value="extrapolate",
                 kind=pkind)
    return f(aindexes)


# Define general functions
# noinspection PyTupleAssignmentBalance
def filtering(x, s_freq, lp=0.5, hp=200, lower_val=0.006, higher_val=0.013, art=3):
    """
    Filters the EEG signal.

    Applies 5th-order Butterworth bandpass filter with corner frequencies lp and hp,
    where lp is the lower limit of the desired frequency band, and hp is the upper limit.
    The frequencies are normalized based on the sampling frequency sfreq.
    Only the frequencies within the band to pass through the filter.

    'art' is the multiple of the std at which signal is considered an artifact.

    Returns filtered EEG signal
    :param x: unfiltered data
    :param s_freq: sampling frequency
    :param lp: lower limit of desired band / filter (to be normalized)
    :param hp: upper limit of desired bad / filter (to be normalized)
    :param lower_val: lower value used for removal of artifacts caused by package loss
    :param higher_val: higher value used for removal of artifacts caused by package loss
    :param art: std of the signal is multiplied by this value to filter out additional artifacts
    :return:
    """
    # artifact rejection (package loss)
    rej = np.where(x > lower_val, x, np.nan)
    rej = interpolate_nan(rej, pkind='linear')
    rej = np.where(rej < higher_val, rej, np.nan)
    rej = interpolate_nan(rej, pkind='linear')

    # filter by applying a 5th Order Bandpass Butterworth Filter
    b, a = signal.butter(N=5, Wn=[lp / (s_freq / 2), hp / (s_freq / 2)], btype='bandpass')
    rej = signal.filtfilt(b, a, rej)

    # if art is provided as argument (int), perform artifact rejection based on art * std
    if art:
        rej = np.where((rej > np.mean(rej) + art * np.std(rej)) | (rej < np.mean(rej) - art * np.std(rej)), np.nan, rej)
        return interpolate_nan(rej, pkind='linear')
    return rej


def time_to_samples(time_str, s_freq):
    """
    The total number of seconds is multiplied by the sampling frequency to convert
    the time duration from seconds to samples. This step ensures that the time duration
    is expressed in the context of the discrete samples taken at the specified sampling rate.

    :param time_str: time as string (sep = '-')
    :param s_freq: sampling frequency
    :return:
    """
    # split the time string and convert to int
    time_parts = [int(x) for x in time_str.split('-')]

    # calculate the total number of seconds
    total_seconds = time_parts[0] * 3600 + time_parts[1] * 60 + time_parts[2]

    return total_seconds * int(s_freq)
