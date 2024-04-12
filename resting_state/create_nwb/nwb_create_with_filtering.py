"""
Filter EEG data and create NWB files
"""
import os
import mne
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil import tz
from pynwb import NWBFile, NWBHDF5IO
from pynwb.file import Subject
from pynwb.ecephys import ElectricalSeries
from ndx_events import TTLs
from hdmf.backends.hdf5.h5_utils import H5DataIO

from helper_functions import get_all_edf_files
from eeg_filtering_functions import filtering
from settings import general, paths_resting_state, filtering


def create_nwb_file(ses_descr, start_t, id, ses_id, arena):
    experimenter = general['experimenter']
    institution = general['institution']
    lab = general['lab']

    print('Creating NWB file...')
    # Create NWB file
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
    return nwb


def load_metadata(edf_file, metadata_file):
    # Load metadata file
    metadata = pd.read_excel(metadata_file, dtype={'mouseName': str, 'mouseId': str})

    # Get metadata info
    directory, filename = os.path.split(edf_file)
    info = metadata[metadata['edf'] == filename].to_dict(orient='records')[0]

    # Prep NWB file metadata
    session_description = f"Animal {info['mouseId']} in resting-state experiment"
    start_time = datetime.strptime('-'.join([info['date'], info['time']]), '%Y-%m-%d-%H-%M-%S').replace(
        tzinfo=tz.tzlocal())
    identifier = f'resting_state_{info["mouseId"]}'
    session_id = f'{info["mouseId"]}_{info["sesId"]}'
    arena = f'Arena_{info["arena"]}'

    # create NWB file using metadata
    return info, identifier, create_nwb_file(session_description, start_time, identifier, session_id, arena)


def add_subject_info(nwb, info):
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
    print('Adding electrode information...')

    electrode_info = filtering['electrode_info']

    # Add device and electrode information
    device = nwb.create_device(
        name=str(info['transmitterId']),
        description=str(info['transmitterId']),
        manufacturer='TaiNi')

    nwb.add_electrode_column(name='label', description='label of electrode')

    for channel, details in electrode_info.items():
        location = details[0]
        AP = float(details[1])
        ML = float(details[2])
        DV = float(details[3])
        el_type = details[4]

        # create an electrode group for this channel
        electrode_group = nwb.create_electrode_group(
            name=channel,
            description=f'{channel}_{el_type}_{location}',
            device=device,
            location=location
        )
        nwb.add_electrode(
            x=AP, y=ML, z=DV, imp=np.nan,
            location=location,
            filtering='unknown',
            group=electrode_group,
            label=f'{el_type}_{location}'
        )
    return nwb


def add_eeg_data(nwb, file):
    print('Adding EEG data...')

    electrode_info = filtering['electrode_info']

    # Add raw EEG data
    raw = mne.io.read_raw_edf(file)
    sfreq = raw.info['sfreq']
    data = raw.get_data(picks=list(electrode_info.keys()))

    all_table_region = nwb.create_electrode_table_region(
        region=list(range(len(electrode_info.keys()))),  # reference row indices 0 to N-1
        description='all electrodes'
    )
    raw_elec_series = ElectricalSeries(
        name='raw_EEG',
        data=H5DataIO(data=data.T, compression=True),
        # to transpose the data because the (channels, data) format doesn't work lul
        electrodes=all_table_region,
        starting_time=0.,  # relative to NWBFile.session_start_time
        rate=sfreq  # Sampling Frequency
    )
    nwb.add_acquisition(raw_elec_series)
    return nwb, raw, sfreq, all_table_region


def add_filtered_eeg(nwb, raw, sfreq, all_table_region):
    print('Filtering EEG..')

    electrode_info = filtering['electrode_info']
    lcut, hcut = filtering['lcut'], filtering['hcut']
    low_val, high_val = filtering['low_val'], filtering['high_val']
    art = filtering['art']

    filt = []
    for channel in electrode_info.keys():
        filt.append(filtering(raw[channel][0][0], sfreq, lcut, hcut, low_val, high_val, art))
    filt = np.array(filt)

    # Create new ElectricalSeries object to hold the filtered EEG, and add to nwb
    filt_elec_series = ElectricalSeries(
        name='filtered_EEG',
        data=H5DataIO(data=filt.T, compression=True),
        electrodes=all_table_region,
        starting_time=0.,
        rate=sfreq,
        filtering=f'5th Order Bandpass butterwort Filter. Low:{lcut} Hz, High: {hcut}, low_val:{low_val}, high_val:{high_val}, art:{art}'
    )
    nwb.add_acquisition(filt_elec_series)
    return nwb


def add_ttl(nwb, raw):
    # Add raw TTL annotations
    ttl_data = raw.annotations.description
    ttl_data_sync_1 = np.delete(ttl_data, np.where(ttl_data == "SYNC_0"))  # keep the onsets of SYNC_1 only
    ttl_timestamps = np.delete(raw.annotations.onset, np.where(ttl_data == "SYNC_0"))  # keep timestamps of SYNC_1 only

    # TTL object needs data values, so parse these and keep only those of SYNC_1 onsets
    ttl_data_values = np.array([int(item.split('_')[1]) for item in ttl_data])
    ttl_data_values = np.delete(ttl_data_values, np.where(ttl_data == "SYNC_0"))

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
    Core of this file. Calls all other functions.
    """
    edf_folder = paths_resting_state['edf_folder']
    nwb_output_folder = paths_resting_state['nwb_files_folder']
    metadata_file = paths_resting_state['metadata']

    edf_files = get_all_edf_files(edf_folder)

    # for each file, generate a NWB file through multiple (processing) steps
    i, tasks = 0, len(edf_files)
    for file in edf_files:
        print(f"Working on {file.split('/')[-1]}")
        info, identifier, nwb = load_metadata(file, metadata_file)
        if os.path.isfile(f'{nwb_output_folder}/{identifier}.nwb') or "TAINI_100E_80108_3.9" in file:
            print(f"File {identifier} already exits, continuing..")
            tasks -= 1
            continue  # skip creating nwb if already exists
        nwb = add_subject_info(nwb, info)  # add subject information
        nwb = add_electrode_info(nwb, info)  # add electrode info
        nwb, raw, sfreq, all_table_region = add_eeg_data(nwb, file)  # add raw eeg data
        nwb = add_filtered_eeg(nwb, raw, sfreq, all_table_region)  # add filtered eeg data
        nwb = add_ttl(nwb, raw)  # add ttl to the nwb file (we only have 1 channel for the small social experiments)

        print('Saving file...')
        with NWBHDF5IO(f'{nwb_output_folder}/{identifier}.nwb', 'w') as io:
            io.write(nwb)
        i += 1
        print(f"{round(i / tasks * 100)}% done")

        # clean up
        raw.close()
        io.close()


# Starting point. Process begins here.
if __name__ == "__main__":
    main()
    sys.exit(0)
