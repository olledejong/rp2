"""
File that contains helper functions
"""
import os
import glob


def get_all_edf_files(root_dir):
    # find all .edf files (also works if all .edf files are in the root directory)
    edf_files = []
    for root, dirs, files in os.walk(root_dir):
        if "trash_recordings" in dirs:  # do not handle recordings that are in trash folder
            dirs.remove("trash_recordings")
        edf_files.extend(glob.glob(os.path.join(root, '*.edf')))
    return edf_files
