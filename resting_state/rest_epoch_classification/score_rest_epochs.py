"""
This file can be used to score clips from experiment recordings that were generated using the
'generate_epoch_clips.py' script.
"""
import cv2
import mne
import numpy as np
import pandas as pd

from resting_state.settings import omitted_other, omitted_after_clustering, cluster_annotations
from shared.helper_functions import *


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
                    behaviour = 'sleeping'
                    break
                elif key == 51:  # press 3 on keyboard
                    behaviour = 'other'
                    break
        # set position back to frame 0 for when it possibly needs to be played again
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # lose and destroy windows
    cap.release()
    cv2.destroyAllWindows()
    root.quit()

    return behaviour


def score_epoch_clips(subject_clips, subject_id, clips_folder):
    print("To score each clip, press one of the following keys: 1 for resting, and 2 for sleeping, and 3 for other.")

    behaviours = []

    # loop through all subject's clips
    for clip_filename in subject_clips:
        path_to_clip = os.path.join(clips_folder, clip_filename)
        epoch_n = path_to_clip.split("_")[-1].split(".")[0]

        # save this clip's behaviour label
        behaviours.append(score_epoch_clip(path_to_clip, epoch_n, subject_id))

    # return behaviour for all epochs / clips
    return behaviours


def main():
    print("Select the folder holding the resting-state epoch files")
    epochs_folder = select_folder("Select the folder holding the resting-state epoch files")
    print("Select the folder holding the clips that need to be scored")
    clips_folder = select_folder("Select the folder holding the clips that need to be scored")

    # get the unique subject ids for which there are clips created
    unique_subject_ids = []
    clips = os.listdir(clips_folder)
    for clip in clips:
        subj_id = int(clip.split('_')[0])
        if subj_id not in unique_subject_ids:
            unique_subject_ids.append(subj_id)

    # for each subject that there's clips for, score each clip and save the annotations to the epochs file
    for i, subject_id in enumerate(unique_subject_ids):

        # PERFORM SANITY CHECKS #

        if subject_id in omitted_other:
            print(f'Subject {subject_id} was omitted for some reason, proceeding..')
            continue

        if subject_id in omitted_after_clustering:
            print(f'Subject {subject_id} was omitted because clustering results were inconclusive/of bad quality, '
                  f'proceeding..')
            continue

        annotated_epoch_files = os.listdir(epochs_folder)
        if any(str(subject_id) in file for file in annotated_epoch_files if file.startswith('resting_epochs_man')):
            print(f'Clips for subject {subject_id} have already been scored, proceeding..')
            continue

        # GET THE CLIP FILENAMES FOR THIS SUBJECT #

        print(f'Working with clips of subject {subject_id}')

        # get the filenames of this subject's clips and make sure the clips are sorted such that they align with the
        # epochs in the epoch object/metadata
        subject_clips = sorted(
            [clip for clip in clips if str(subject_id) in clip],
            key=lambda x: int(x.split("_")[-1].split(".")[0])
        )

        # LOAD THE EPOCHS FOR THIS SUBJECT #

        epochs = mne.read_epochs(
            os.path.join(epochs_folder, f"filtered_epochs_w_clusters_{subject_id}-epo.fif"),
            preload=True
        )
        # get the resting-state cluster epochs
        resting_epochs = epochs[epochs.metadata["cluster"] == cluster_annotations[subject_id]['rest']]
        print(f'There are {len(resting_epochs)} resting-state epochs to be scored for this subject')

        # remove epochs from epoch object if there's no clip for them (epoch was skipped while creating the clips)
        if len(resting_epochs) != len(subject_clips):
            clips_epoch_n = [int(clip.split('_')[-1].split('.')[0]) for clip in clips if str(subject_id) in clip]
            missing_clips = np.setdiff1d(np.array(resting_epochs.metadata.index), np.array(clips_epoch_n))
            row_numbers = resting_epochs.metadata.index.get_indexer(missing_clips)
            resting_epochs = [epoch for i, epoch in enumerate(resting_epochs) if i not in row_numbers]

            print(f'Removed {len(missing_clips)} epoch(s) because there was no clip. '
                  f'Current # epochs: {len(resting_epochs)}')

        # SCORE THE CLIPS AND STORE THE ANNOTATIONS IN THE METADATA OF THE EPOCHS #

        man_annotations = score_epoch_clips(subject_clips, subject_id, clips_folder)
        resting_epochs.metadata['behaviour'] = man_annotations

        # save resting-state epochs with manual annotations
        resting_epochs.save(
            os.path.join(epochs_folder, f"resting_epochs_man_annotated_{subject_id}-epo.fif"),
            overwrite=True
        )


if __name__ == '__main__':
    main()
