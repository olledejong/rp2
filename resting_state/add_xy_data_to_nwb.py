"""
Adds spatial information about the mouse to the right nwb file
"""
import numpy as np
import pandas as pd
import ndx_events
from pynwb import NWBHDF5IO, TimeSeries
from pynwb.behavior import SpatialSeries

from shared.helper_functions import *


def get_coordinate_data(xy_filename):
    """
    Creates a clean dataframe with information all the relevant
    spatial information.

    In the case of resting-state experiments, there's only xy data on
    one mouse in every coordinates-data Excel file.
    """
    df = pd.read_excel(xy_filename, header=None)  # read coordinates file
    split = df.iloc[:, 0].str.find("Trial time").idxmax()  # find row index of actual xy data header
    df_meta, df_xy = df.iloc[:split, ], df.iloc[split:, ]  # split in metadata and xy data
    df_xy.columns = df_xy.iloc[0]  # set column names of xy data to correct row
    df_xy = df_xy.drop([split, split+1])  # remove unwanted rows (unit and row that is now used as colnames)
    df_xy.replace('-', np.nan, inplace=True)  # replace missing values (-) with nan
    df_xy = df_xy.reset_index(drop=True)  # reset index

    # return only wanted columns
    return df_xy[["Recording time", "X center", "Y center", "X nose", "Y nose", "Velocity", "Direction",
                  "Movement_center(Moving / Center-point)", "Movement_nose(Moving / Nose-point)"]]


def generate_module_components(data):
    """
    Generates the components that will be added to the behavior module
    """
    numerical_x = np.array([x for x in data["X center"] if isinstance(x, (int, float))])
    numerical_y = np.array([x for x in data["Y center"] if isinstance(x, (int, float))])

    ref_frame = (
        f'left: {np.min(numerical_x)},'
        f' right: {np.max(numerical_x)},'
        f' top: {np.max(numerical_y)},'
        f' bottom: {np.min(numerical_y)}'
    )

    # generate behavior_model components
    ss_center = SpatialSeries(
        name="xy_center",
        description="(x,y) center position",
        data=data[['X center', 'Y center']].to_numpy(),
        timestamps=data['Recording time'].to_numpy(),
        reference_frame=ref_frame,
        unit='cm'
    )
    ss_nose = SpatialSeries(
        name="xy_nose",
        description="(x,y) nose position",
        data=data[['X nose', 'Y nose']].to_numpy(),
        timestamps=data['Recording time'].to_numpy(),
        reference_frame=ref_frame,
        unit='cm'
    )
    velocity_series = TimeSeries(
        name="velocity",
        description="velocity of animal",
        data=data['Velocity'].to_numpy(),
        timestamps=data['Recording time'].to_numpy(),
        unit='cm/s',
    )
    direction_series = TimeSeries(
        name=f"direction",
        description=f"direction of the animal ",
        data=data['Direction'].to_numpy(),
        timestamps=data['Recording time'].to_numpy(),
        unit='degrees',
    )
    motion_series = TimeSeries(
        name=f"motion",
        description="whether the animal is moving (boolean)",
        data=data['Movement_center(Moving / Center-point)'].to_numpy(),
        timestamps=data['Recording time'].to_numpy(),
        unit='bool',
    )
    motion_nose_series = TimeSeries(
        name=f"motion_nose",
        description="whether the nose of the animal is moving (boolean)",
        data=data['Movement_nose(Moving / Nose-point)'].to_numpy(),
        timestamps=data['Recording time'].to_numpy(),
        unit='bool',
    )
    return [ss_center, ss_nose, velocity_series, direction_series, motion_series, motion_nose_series]


def main():
    print("Select the folder to hold the NWB files")
    nwb_folder = select_folder("Select the folder to hold the NWB files")
    print("Select the folder that holds the from EthoVision exported movement (xy) data")
    coordinates_folder = select_folder("Select the folder that holds the from EthoVision exported movement (xy) data")

    # loop over all created NWB files
    nwb_files = os.listdir(nwb_folder)
    jobs = len(nwb_files)
    for i, nwb_filename in enumerate(nwb_files):
        if ".nwb" not in nwb_filename:
            continue  # skip non-nwb files

        with NWBHDF5IO(f'{nwb_folder}/{nwb_filename}', "a") as io:  # open it
            nwb = io.read()

            animal_id = nwb.subject.subject_id  # get animal id for retrieving the right coordinates file
            print(f"Handling file {nwb_filename} (mouseId: {animal_id}).")

            if 'coordinate_data' in nwb.processing.keys():
                print("Spatial data present, proceeding..")
                continue

            print("Spatial data not yet present, adding it..")
            # if not coordinates file for this animal, skip
            files = os.listdir(coordinates_folder)
            if not any(animal_id in file for file in files):
                print(f'No coordinates file found for animal {animal_id}')
                continue

            # load the xy data for the animal in question
            for file in files:
                if animal_id in file and file.endswith(".xlsx"):
                    xy_data = get_coordinate_data(os.path.join(coordinates_folder, file))

            # make new behavioral module and add to nwb file
            behavior_module = nwb.create_processing_module(
                name="coordinate_data", description="Raw coordinate/motion/head orientation data"
            )
            behavior_components = generate_module_components(xy_data)
            [behavior_module.add(comp) for comp in behavior_components]

            # write altered nwb file
            io.write(nwb)
            print(f'Successfully added spatial info to {nwb_filename}.')
            print(f"Progress: {round(i / jobs * 100)}% done")


# Starting point. Process starts here.
if __name__ == '__main__':
    main()
