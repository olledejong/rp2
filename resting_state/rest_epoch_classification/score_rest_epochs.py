"""
This file can be used to score clips from experiment recordings that were generated using the
'generate_epoch_clips.py' script.
"""
import os
import cv2
import mne
import numpy as np
import pandas as pd
from tkinter import *

from settings import paths, cluster_annotations


def score_epoch_clip(input_video, epoch_n, subject_id):
    cap = cv2.VideoCapture(input_video)  # open the video

    if not cap.isOpened():
        print("Error: Could not open the video file.")
        return

    # make tk window and hide it
    root = Tk()
    root.withdraw()

    behaviour = 'unknown'
    # loop and show clip as long as the user hasn't provided a behaviour label for it
    while True:
        ret, frame = cap.read()

        # show video clip
        if ret:
            width = int(frame.shape[1] * 2)
            height = int(frame.shape[0] * 2)
            resized_frame = cv2.resize(frame, (width, height))

            cv2.imshow(f'Subject {subject_id}, epoch {epoch_n}', resized_frame)
            key = cv2.waitKey(1)  # Use a small delay (1 millisecond) to allow video to play smoothly
            if key != -1:  # Check if a key is pressed
                if key == 27:  # press Esc
                    break
                elif key == 49:  # press 1 on keyboard
                    behaviour = 'resting'
                    break
                elif key == 50:  # press 2 on keyboard
                    behaviour = 'non-resting'
                    break
        # set position back to frame 0 for when it possibly needs to be played again
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # lose and destroy windows
    cap.release()
    cv2.destroyAllWindows()
    root.quit()

    return behaviour


def score_epoch_clips(subject_clips, subject_id):
    print("To score each clip, press one of the following keys: 1 for sleeping, 2 for resting, and 3 for other.")

    behaviours = []

    # loop through all subject's clips
    for clip_filename in subject_clips:
        path_to_clip = os.path.join(paths['video_analysis_output'], 'clips', clip_filename)
        epoch_n = path_to_clip.split("_")[-1].split(".")[0]

        # save this clip's behaviour label
        behaviours.append(score_epoch_clip(path_to_clip, epoch_n, subject_id))

    # return behaviour for all epochs / clips
    return behaviours


def main():
    # load some info about the subject
    metadata_df = pd.read_excel(paths["metadata"])

    for i, subject_id in enumerate(metadata_df['mouseId']):
        subject_meta = metadata_df[metadata_df['mouseId'] == int(subject_id)]
        print('Mouse name:', subject_meta['mouseName'].iloc[0])

        # get the filenames of this subject's clips
        clips = os.listdir(os.path.join(paths['video_analysis_output'], 'clips'))
        subject_clips = [clip for clip in clips if str(subject_id) in clip]

        man_annotations = score_epoch_clips(subject_clips, subject_id)

        # load the subject epochs
        epochs = mne.read_epochs(
            os.path.join(paths['epochs_folder'], f"filtered_epochs_w_clusters_{subject_id}-epo.fif"),
            preload=True
        )
        resting_epochs = epochs[epochs.metadata["cluster"] == cluster_annotations[subject_id]['rest']]
        resting_epochs.metadata['behaviour'] = man_annotations

        # save resting-state epochs with manual annotations
        resting_epochs.save(
            os.path.join(paths['epochs_folder'], f"resting_epochs_man_annotated_{subject_id}-epo.fif"),
            overwrite=True
        )


if __name__ == '__main__':
    main()
