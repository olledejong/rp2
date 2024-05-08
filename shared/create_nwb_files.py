"""
Filter EEG data and create NWB files
"""
import re
import sys
import mne
import numpy as np
import pandas as pd

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


def create_nwb_file(metadata, experiment_name):
    """
    Creates the initial NWB file. Does not hold any electrode or EEG data yet.

    :param metadata:
    :param experiment_name:
    :return:
    """
    # load some project information from the settings_general.py file
    experimenter = general['experimenter']
    institution = general['institution']
    lab = general['lab']

    # prepare some metadata
    session_description = f'Animal {metadata["mouseId"]} in {experiment_name} experiment'
    start_time = datetime.strptime(
        '-'.join([metadata['date'], metadata['time']]),
        '%Y-%m-%d-%H-%M-%S'
    ).replace(tzinfo=tz.tzlocal())

    identifier = f'{experiment_name}_{metadata["mouseId"]}'
    session_id = f'{metadata["mouseId"]}_{metadata["sesId"]}'
    arena = f'Arena_{metadata["arena"]}'

    nwb = NWBFile(
        session_description=session_description,
        identifier=identifier,
        session_start_time=start_time,
        session_id=session_id,
        experiment_description=arena,
        experimenter=experimenter,
        lab=lab,
        institution=institution,

    )
    print('Created initial NWB')
    return nwb


def load_metadata(edf_filename, all_animals_metadata):
    """
    Function that combines metadata from the EDF filename and from a metadata file that holds
    supplementary metadata (genotype, birthday etc.)

    :param edf_filename:
    :param all_animals_metadata:
    :return:
    """
    # extract specific info from filename, if not correctly formatted, report to user and exit
    try:
        _, filename = os.path.split(edf_filename)

        # find parts from list using regex
        subject_id = re.search('_(\d{5})_', filename).group(1)
        mouse_name = re.search('_(\d\.\d+)_', filename).group(1)
        date = re.search('_(\d{4}-\d{2}-\d{2})_', filename).group(1).replace('_', '')
        time = re.search('_(\d{2}-\d{2}-\d{2})_', filename).group(1).replace('_', '')
        ses_id = re.search('_(\d{3})_', filename).group(1).replace('_', '')
        transmitter_id = re.search('_(\d{3}(\w|\d))_', filename).group(1).replace('_', '')

        subject_id, mouse_name = str(subject_id), str(mouse_name)

    except ValueError as err:
        sys.exit('Error: make sure the EDF file names are in the correct format. Split on underscores, there should'
                 ' be eight (8) parts: TAINI_$TransmID_$SubID_$ALIAS_%Y-%m-%d_%H-%M-%S_$SesID_$INC.edf')

    # extract info from all animal metadata file and form a dict that holds all subject metadata
    metadata = {
        'edf_filename': filename,
        'date': date,
        'time': time,
        'sesId': ses_id,
        'mouseId': subject_id,
        'mouseName': mouse_name,
        'transmitterId': transmitter_id,
        'arena': all_animals_metadata[all_animals_metadata['mouseId'] == subject_id]['arena'].iloc[0],
        'genotype': all_animals_metadata[all_animals_metadata['mouseId'] == subject_id]['genotype'].iloc[0],
        'birthday': all_animals_metadata[all_animals_metadata['mouseId'] == subject_id]['birthday'].iloc[0],
        'rfid': all_animals_metadata[all_animals_metadata['mouseId'] == subject_id]['RFID'].iloc[0],
        'weight': all_animals_metadata[all_animals_metadata['mouseId'] == subject_id]['weight'].iloc[0],
        'sex': all_animals_metadata[all_animals_metadata['mouseId'] == subject_id]['sex'].iloc[0],
        'species': all_animals_metadata[all_animals_metadata['mouseId'] == subject_id]['species'].iloc[0],
    }

    return metadata


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
        genotype=info['genotype'],  # WT or Drd2 KO
        weight=str(info['weight']),
        date_of_birth=info['birthday'].to_pydatetime().replace(tzinfo=tz.tzlocal())
    )
    return nwb


def add_electrode_info(nwb, transmitter_id):
    """
    Adds the electrode information to the NWB file.
    :param nwb:
    :param transmitter_id:
    :return:
    """
    electrode_info = filtering['electrode_info']

    # Add device and electrode information
    device = nwb.create_device(
        name=transmitter_id,
        description=transmitter_id,
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
    if resample_freq is not None:
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
    experiment_name = input('Experiment name (e.g. 3c_sociability or resting_state): ')
    print("Select the folder that holds the EDF files")
    edf_folder = select_folder("Select the folder that holds the EDF files")
    print("Select the excel file that holds information about all experimental animals")
    all_animals_metadata = select_file("Select the excel file that holds information about all experimental animals")
    print("Select or create a folder to hold the output NWB files")
    nwb_output_folder = select_or_create_folder("Select or create a folder to hold the output NWB files")

    # read the metadata
    all_animals_metadata = pd.read_excel(all_animals_metadata, dtype={'mouseName': str, 'mouseId': str})

    # load all edf filenames or this experiment
    edf_files = get_all_edf_files(edf_folder)

    # for each file, generate a NWB file through multiple (processing) steps
    for i, file in enumerate(edf_files):
        print(f"Creating NWB with EEG data from EDF file {file.split('/')[-1]}")

        # load needed metadata from edf filename as well as from the 'all_animals_metadata' Excel file
        metadata = load_metadata(file, all_animals_metadata)

        # create the NWB file using the metadata
        nwb = create_nwb_file(metadata, experiment_name)

        # if NWB file already exists for this animal, skip it
        nwb_filename = f'{experiment_name}_{metadata["mouseId"]}.nwb'
        save_nwb_to = os.path.join(nwb_output_folder, nwb_filename)
        if os.path.isfile(save_nwb_to):
            print(f"File {nwb_filename}.nwb already exits, continuing!\n")
            continue

        # add all kinds of data to the nwb file
        nwb = add_subject_info(nwb, metadata)
        nwb = add_electrode_info(nwb, str(metadata['transmitterId']))
        nwb, raw = add_eeg_data(nwb, file)
        nwb = add_ttl(nwb, raw)  # we only have 1 TTL channel for the small social experiments

        # save the NWB file
        with NWBHDF5IO(save_nwb_to, 'w') as io:
            io.write(nwb)

        print(f"Saved file, {round((i + 1) / len(edf_files) * 100)}% done\n")

        # clean up
        raw.close()
        io.close()


# Starting point. Process begins here.
if __name__ == "__main__":
    main()
    print('Done!')
