"""
This file is used to generate unique metadata for each EDF file
from one single subject metadata file.
"""
import re
import sys
import json
import pandas as pd

from helper_functions import get_all_edf_files

# Starting point. Process starts here.
if __name__ == '__main__':

    # load the project settings (directory paths etc)
    with open('settings.json', "r") as f:
        settings = json.load(f)

    subject_metadata = settings['subject_metadata']
    edf_folder = settings['edf_folder']

    # load the subject metadata
    sub_meta = pd.read_excel(settings['subject_metadata'], dtype={'mouseName': str, 'mouseId': str})

    df = pd.DataFrame()  # empty dataframe to store all data in

    # find all .edf files (also works if all .edf files are in the root directory)
    edf_files = get_all_edf_files(edf_folder)

    for i, filename in enumerate(edf_files):  # loop over all edf files
        if ".edf" not in filename:  # only consider .edf type
            continue

        filename = re.split('/', filename)[-1]  # split absolute path on last occurrence of '/'
        info = re.split('_', filename)  # split filename to extract data

        # get specifics from filename
        transmitterId, subjectId, mouseName = info[1], str(info[2]), str(info[3])
        date, time, sesId = info[4], info[5], info[6]

        minfo = sub_meta[sub_meta['mouseId'] == subjectId]
        if len(minfo) == 0: sys.exit(f"Error: given mouseId ({subjectId}) could not be found in subject metadata")

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

    df.to_excel(settings['metadata'], index=False)

    sys.exit(0)  # done, terminating..
