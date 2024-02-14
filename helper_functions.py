"""
File that contains helper functions
"""
import os
import glob
import matplotlib.pyplot as plt


def get_all_edf_files(root_dir):
    # find all .edf files (also works if all .edf files are in the root directory)
    edf_files = []
    for root, dirs, files in os.walk(root_dir):
        if "trash_recordings" in dirs:  # do not handle recordings that are in trash folder
            dirs.remove("trash_recordings")
        edf_files.extend(glob.glob(os.path.join(root, '*.edf')))
    return edf_files


def save_figure(path, bbox_inches='tight', dpi=300):
    """
    Custom function that lets you save a pyplot figure and creates the directory where necessary
    """
    directory = os.path.split(path)[0]
    filename = os.path.split(path)[1]
    if directory == '':
        directory = '.'

    if not os.path.exists(directory):
        os.makedirs(directory)

    save_path = os.path.join(directory, filename)

    # Actually save the figure
    plt.savefig(save_path, bbox_inches=bbox_inches, dpi=dpi)
    plt.close()
