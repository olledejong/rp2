"""
Filter EEG data and create NWB files
"""
import mne
import numpy as np
import pandas as pd
import three_chamber.settings as three_camber_settings

from datetime import datetime
from dateutil import tz
from pynwb import NWBFile, NWBHDF5IO
from pynwb.file import Subject
from pynwb.ecephys import ElectricalSeries
from ndx_events import TTLs
from hdmf.backends.hdf5.h5_utils import H5DataIO

from shared.helper_functions import *
from shared.eeg_filtering_functions import filter_eeg
from settings_general import *


def create_nwb_file(ses_descr, start_t, id, ses_id, arena):
    """
    Creates the initial NWB file. Does not hold any electrode or EEG data yet.

    :param ses_descr:
    :param start_t:
    :param id:
    :param ses_id:
    :param arena:
    :return:
    """
    # load some project information from the settings_general.py file
    experimenter = general['experimenter']
    institution = general['institution']
    lab = general['lab']

    nwb = NWBFile(
        session_description=ses_descr,
        identifier=id,
        session_start_time=start_t,
        session_id=ses_id,
        experiment_description=arena,
        experimenter=experimenter,
        lab=lab,
        institution=institution
    )
    print('Created initial NWB')
    return nwb


def load_metadata(edf_file, experiment_edf_metadata):
    """
    Load EDF metadata file and return the needed information

    :param edf_file:
    :param experiment_edf_metadata:
    :return:
    """
    # Load metadata file
    _, filename = os.path.split(experiment_edf_metadata)
    experiment_name = filename.rsplit('_', 1)
    metadata = pd.read_excel(experiment_edf_metadata, dtype={'mouseName': str, 'mouseId': str})

    # Get metadata info
    directory, filename = os.path.split(edf_file)
    info = metadata[metadata['edf'] == filename].to_dict(orient='records')[0]

    # Prep NWB file metadata
    session_description = f"Animal {info['mouseId']} in {experiment_name} experiment"
    start_time = datetime.strptime(
        '-'.join([info['date'], info['time']]),
        '%Y-%m-%d-%H-%M-%S'
    ).replace(tzinfo=tz.tzlocal())

    identifier = f'{experiment_name}_{info["mouseId"]}'
    session_id = f'{info["mouseId"]}_{info["sesId"]}'
    arena = f'Arena_{info["arena"]}'

    # create NWB file using metadata
    nwb = create_nwb_file(session_description, start_time, identifier, session_id, arena)

    return info, identifier, nwb


def add_subject_info(nwb, info):
    """
    Adds the subject metadata to the NWB file.

    :param nwb:
    :param info:
    :return:
    """
    nwb.subject = Subject(
        description=info['mouseName'],  # name that we give it
        subject_id=info['mouseId'],  # unique animal id in the mouse card
        species=info['species'],
        sex=info['sex'],
        genotype=info['genotype'],  # WT or Drd2 KO for these social experiments
        weight=str(info['weight']),
        date_of_birth=info['birthday'].to_pydatetime().replace(tzinfo=tz.tzlocal())
    )
    return nwb


def add_electrode_info(nwb, info):
    """
    Adds the electrode information to the NWB file.
    :param nwb:
    :param info:
    :return:
    """
    electrode_info = filtering['electrode_info']

    # Add device and electrode information
    device = nwb.create_device(
        name=str(info['transmitterId']),
        description=str(info['transmitterId']),
        manufacturer='TaiNi'
    )

    nwb.add_electrode_column(name='label', description='label of electrode')

    for channel, details in electrode_info.items():
        location = details[0]
        AP, ML, DV = float(details[1]), float(details[2]), float(details[3])
        el_type = details[4]

        # create an electrode group for this channel
        electrode_group = nwb.create_electrode_group(
            name=channel,
            description=f'{channel}_{el_type}_{location}',
            device=device,
            location=location
        )
        # add this electrode's info to the NWB file
        nwb.add_electrode(
            x=AP, y=ML, z=DV, imp=np.nan,
            location=location,
            filtering='unknown',
            group=electrode_group,
            label=f'{el_type}_{location}'
        )
    print('Added electrode information')
    return nwb


def add_eeg_data(nwb, file):
    """
    Reads the EDF file, and adds the raw EEG data to the NWB file.

    :param nwb:
    :param file:
    :return:
    """
    print('Adding EEG data')

    # load the electrode info from the settings file
    electrode_info = filtering['electrode_info']

    # read the edf file
    raw_eeg = mne.io.read_raw_edf(file)
    sfreq = raw_eeg.info['sfreq']

    # get the data using the keys of the electrode_info dictionary (see settings_general.py)
    data = raw_eeg.get_data(picks=list(electrode_info.keys()))

    # create electrode table
    all_table_region = nwb.create_electrode_table_region(
        region=list(range(len(electrode_info.keys()))),  # reference row indices 0 to N-1
        description='all electrodes'
    )

    # create electrical series holding the raw EEG data for the recorded channels
    raw_elec_series = ElectricalSeries(
        name='raw_EEG',
        data=H5DataIO(data=data.T, compression=True),  # transpose the data because (channels, data) format doesn't work
        electrodes=all_table_region,
        starting_time=0.,
        rate=sfreq  # sampling frequency
    )
    nwb.add_acquisition(raw_elec_series)

    # also add the filtered EEG data to the NWB object
    nwb = add_filtered_eeg(nwb, raw_eeg, sfreq, all_table_region)  # add filtered eeg data

    print('Added EEG data')
    return nwb, raw_eeg


def add_filtered_eeg(nwb, raw, s_freq, all_table_region):
    """
    Filters the raw EEG data and adds this to the NWB file.

    :param nwb:
    :param raw:
    :param s_freq:
    :param all_table_region:
    :return:
    """
    electrode_info = filtering['electrode_info']
    lcut, hcut = filtering['lcut'], filtering['hcut']
    low_val, high_val = filtering['low_val'], filtering['high_val']
    art = filtering['art']

    # if 3-chamber, and we want to resample to a specific frequency, do so
    if three_camber_settings.filtering['resample_freq'] is not None:
        resample_freq = three_camber_settings.filtering['resample_freq']
        raw.resample(resample_freq)
        print(f'Resampled EEG data from {s_freq} Hz to {resample_freq} Hz')
        s_freq = resample_freq  # set s_freq to the sampling frequency that data has been resampled to

    # filter the EEG data (all channels) and put into an array
    filtered_eeg = np.array(
        [filter_eeg(raw[chan][0][0], s_freq, lcut, hcut, low_val, high_val, art) for chan in electrode_info.keys()]
    )

    # Create new ElectricalSeries object to hold the filtered EEG, and add to nwb
    filt_elec_series = ElectricalSeries(
        name='filtered_EEG',
        data=H5DataIO(data=filtered_eeg.T, compression=True),
        electrodes=all_table_region,
        starting_time=0.,
        rate=float(s_freq),
        filtering=f'5th Order Bandpass butterwort Filter. Low:{lcut} Hz, High: {hcut}, low_val:{low_val}, high_val:{high_val}, art:{art}'
    )
    nwb.add_acquisition(filt_elec_series)

    return nwb


def add_ttl(nwb, raw):
    """
    Adds the TTL pulse information to the NWB file. The points of the TTL pulses are later
    used to align the EEG and the video-derived spatial or behavioural information.

    :param nwb:
    :param raw:
    :return:
    """
    # we only need the SYNC_1 onsets, so get the time-stamps of these onsets
    ttl_data = raw.annotations.description
    ttl_data_sync_1 = np.delete(ttl_data, np.where(ttl_data == "SYNC_0"))
    ttl_timestamps = np.delete(raw.annotations.onset, np.where(ttl_data == "SYNC_0"))

    # TTL object needs data values, so parse these and keep only those of SYNC_1 onsets
    ttl_data_values = np.array([int(item.split('_')[1]) for item in ttl_data])
    ttl_data_values = np.delete(ttl_data_values, np.where(ttl_data == "SYNC_0"))

    # create a TTL events object
    ttl_events = TTLs(
        name='TTL_1',
        description='TTL 1 events from EEG annotations (SYNC_1 onsets)',
        timestamps=ttl_timestamps,
        data=ttl_data_values,
        labels=ttl_data_sync_1
    )

    nwb.add_acquisition(ttl_events)
    return nwb


def main():
    """
    Core of this file. Loops through all EDF files and creates a NWB file holding
    information on the subject, the electrodes, ttl etc. Of course also holds the raw
    and filtered EEG data.
    """
    edf_folder = select_folder("Select the folder that holds the EDF files")
    nwb_output_folder = select_or_create_folder("Select or create a folder to hold the output NWB files")
    metadata_file = select_file("Select the experiment's metadata file")

    # load all edf filenames or this experiment
    edf_files = get_all_edf_files(edf_folder)

    # for each file, generate a NWB file through multiple (processing) steps
    for i, file in enumerate(edf_files):
        print(f"Working on {file.split('/')[-1]}")

        # load needed information from experiment specific EDF metadata created using 'create_edf_metadata.py'
        info, identifier, nwb = load_metadata(file, metadata_file)

        # if a NWB file already exists for this animal
        if os.path.isfile(os.path.join(nwb_output_folder, f'{identifier}.nwb')):
            print(f"File {identifier} already exits, continuing..")
            continue  # skip creating nwb if already exists

        # add all kinds of data to the nwb file
        nwb = add_subject_info(nwb, info)
        nwb = add_electrode_info(nwb, info)
        nwb, raw = add_eeg_data(nwb, file)
        nwb = add_ttl(nwb, raw)  # we only have 1 TTL channel for the small social experiments

        with NWBHDF5IO(f'{nwb_output_folder}/{identifier}.nwb', 'w') as io:
            io.write(nwb)
        print(f"Saved file, {round(i / len(edf_files) * 100)}% done")

        # clean up
        raw.close()
        io.close()


# Starting point. Process begins here.
if __name__ == "__main__":
    main()
    print('Done')
