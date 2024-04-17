"""
File that contains helper functions
"""
import os
import glob
from tkinter import Tk, filedialog

import matplotlib.pyplot as plt


def get_all_edf_files(root_dir):
    """
    Finds all .edf files. Works for edf files located in the root of the path, but
    also for all nested edf files

    :param root_dir:
    :return:
    """
    edf_files = []
    for root, dirs, files in os.walk(root_dir):
        if "trash_recordings" in dirs:  # do not handle recordings that are in trash folder
            dirs.remove("trash_recordings")
        edf_files.extend(glob.glob(os.path.join(root, '*.edf')))
    return edf_files


def save_figure(path, bbox_inches='tight', dpi=300):
    """
    Custom function that lets you save a figure and creates the directory where necessary
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


def get_folder_path(title):
    root = Tk()
    root.withdraw()  # we don't want a full GUI, so keep the root window from appearing
    # show an "Open" dialog box and return the path to the selected folder
    return filedialog.askdirectory(title=title)


def get_file_path(title, filetypes=(("Excel files", "*.xlsx"), ("All files", "*.*"))):
    root = Tk()
    root.withdraw()  # we don't want a full GUI, so keep the root window from appearing
    # show an "Open" dialog box and return the path to the selected file
    return filedialog.askopenfilename(
        title=title,
        filetypes=filetypes
    )


def get_save_path(title, filetypes=(("Excel files", "*.xlsx"), ("All files", "*.*"))):
    root = Tk()
    root.withdraw()

    return filedialog.asksaveasfilename(
        title=title,
        defaultextension=".xlsx",
        filetypes=filetypes
    )
