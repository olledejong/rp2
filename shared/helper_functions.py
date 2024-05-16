"""
File that contains helper functions
"""
import os
import glob
import matplotlib.pyplot as plt

from tkinter import Tk, filedialog


def get_all_edf_files(root_dir):
    """
    Finds all .edf files. Works for edf files located in the root of the path, but
    also for all nested edf files. Skips the 'trash_recordings' folder.

    :param root_dir:
    :return:
    """
    edf_files = []
    for root, dirs, files in os.walk(root_dir):
        if "trash_recordings" in dirs:  # do not handle recordings that are in trash folder
            dirs.remove("trash_recordings")
        edf_files.extend(glob.glob(os.path.join(root, '*.edf')))
    return edf_files


def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_figure(path, bbox_inches='tight', dpi=300):
    """
    Custom function that lets you save a figure and creates the directory where necessary

    :param path: path to save to (including filename)
    :param bbox_inches:
    :param dpi:
    :return:
    """
    directory, filename = os.path.split(path)
    if directory == '':
        directory = '.'

    if not os.path.exists(directory):
        os.makedirs(directory)

    save_path = os.path.join(directory, filename)

    # Actually save the figure
    plt.savefig(save_path, bbox_inches=bbox_inches, dpi=dpi)


# Tkinter functions used to request file/folder locations from the user

def select_folder(title):
    root = Tk()
    root.withdraw()

    return filedialog.askdirectory(title=title)


def select_file(title, filetypes=(("Excel files", "*.xlsx"), ("All files", "*.*"))):
    root = Tk()
    root.withdraw()

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


def select_or_create_folder(title):
    root = Tk()
    root.withdraw()  # Hide the main window

    # Prompt user to select an existing folder or create a new one
    return filedialog.askdirectory(
        title=title,
        mustexist=False
    )
