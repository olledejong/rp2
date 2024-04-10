"""
This file is used to generate unique metadata for an experiment. This metadata file holds
 a row for each subject. It holds information like: mouseId, genotype, but also the name of
the EDF file (EEG data) that belongs to that subject.
"""
import os
import re
import sys
import pandas as pd

from helper_functions import get_all_edf_files
from settings import *


def generate_experiment_metadata(paths):
    # load the subject metadata
    sub_meta = pd.read_excel(paths_general['all_animal_metadata'], dtype={'mouseName': str, 'mouseId': str})

    df = pd.DataFrame()  # empty dataframe to store all data in

    # find all .edf files (also works if all .edf files are in the root directory)
    edf_files = get_all_edf_files(paths['edf_folder'])

    for i, filename in enumerate(edf_files):  # loop over all edf files
        if ".edf" not in filename:  # only consider .edf type
            continue

        directory, filename = os.path.split(filename)
        info = re.split('_', filename)  # split filename to extract data

        # get specifics from filename
        transmitterId, subjectId, mouseName = info[1], str(info[2]), str(info[3])
        date, time, sesId = info[4], info[5], info[6]
        minfo = sub_meta[sub_meta['mouseId'] == subjectId]
        if len(minfo) == 0:
            sys.exit(f"Error: given mouseId ({subjectId}) could not be found in subject metadata")

        tmp = pd.DataFrame({
            'edf': filename,
            'date': date,
            'time': time,
            'sesId': sesId,
            'transmitterId': transmitterId, 'arena': minfo['arena'].tolist()[0],
            'mouseId': subjectId, 'mouseName': mouseName,
            'genotype': minfo['genotype'].tolist()[0],
            'birthday': minfo['birthday'].tolist()[0],
            'rfid': minfo['RFID'].tolist()[0],
            'weight': minfo['weight'].tolist()[0],
            'sex': minfo['sex'].tolist()[0],
            'species': minfo['species'].tolist()[0]
        }, index=[i])  # temp container

        df = pd.concat([df, tmp])  # concatenate metadata for this edf file to the rest

    df.to_excel(paths['metadata'], index=False)


if __name__ == '__main__':

    # depending on what experiment you want to create the metadata for, put the paths variable of that experiment
    # as the argument to the function below
    generate_experiment_metadata(paths_3c_sociability)  # TODO change this function argument when you run

    print('Done')

