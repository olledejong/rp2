"""
This file is used to generate unique metadata for an experiment. This metadata file holds
 a row for each subject. It holds information like: mouseId, genotype, but also the name of
the EDF file (EEG data) that belongs to that subject.
"""
import re
import sys
import pandas as pd
from shared.helper_functions import *


def generate_experiment_metadata():
    all_animals_metadata = select_file("Select the excel file that holds information about all experimental animals")
    sub_meta = pd.read_excel(all_animals_metadata, dtype={'mouseName': str, 'mouseId': str})

    # find all .edf files (also works if all .edf files are in the root directory)
    edf_files = get_all_edf_files(select_folder("Select the folder that holds the EDF files"))

    df = pd.DataFrame()  # empty dataframe to store all data in
    for i, filename in enumerate(edf_files):
        if ".edf" not in filename:
            continue

        _, filename = os.path.split(filename)

        # extract specific info from filename
        try:
            _, transmitterId, subjectId, mouseName, date, time, sesId, _ = re.split('_', filename)
            subjectId, mouseName = str(subjectId), str(mouseName)
        except ValueError:
            sys.exit('Error: make sure the EDF file names are in the correct format. Split on underscores, there should'
                     ' be eight (8) parts: TAINI_$TransmID_$SubID_$ALIAS_%Y-%m-%d_%H-%M-%S_$SesID_$INC.edf')

        other_animal_info = sub_meta[sub_meta['mouseId'] == subjectId]

        # if there's no additional info on this subject in the larger animal metadata dataframe, skip it
        if len(other_animal_info) == 0:
            print(f"Skipping ({subjectId}): could not be found in subject metadata.")
            continue

        df = pd.concat([df, pd.DataFrame({
            'edf': filename,
            'date': date, 'time': time, 'sesId': sesId,
            'transmitterId': transmitterId, 'arena': other_animal_info['arena'].tolist()[0],
            'mouseId': subjectId, 'mouseName': mouseName,
            'genotype': other_animal_info['genotype'].tolist()[0],
            'birthday': other_animal_info['birthday'].tolist()[0],
            'rfid': other_animal_info['RFID'].tolist()[0],
            'weight': other_animal_info['weight'].tolist()[0],
            'sex': other_animal_info['sex'].tolist()[0],
            'species': other_animal_info['species'].tolist()[0]
        }, index=[i])])  # concatenate metadata for this edf file to the rest

    save_to = get_save_path(
        "Select where you want to save the experiment metadata"
    )
    df.to_excel(save_to, index=False)


if __name__ == '__main__':
    generate_experiment_metadata()
    print('Done')
