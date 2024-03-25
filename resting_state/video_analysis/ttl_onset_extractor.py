"""
This file holds logic that extracts the state of the LED (on/off) in every frame
of all videos located in the given video folder. This LED state extraction is done
using the ROIs that are created using the identify_led_rois.py script.
the ROIs
"""
import os
import ast
import cv2
import pickle
import pandas as pd
import numpy as np

from settings import paths


def is_led_on(roi, threshold=245):
    """
    Returns True when the LED is on, False otherwise

    :param roi:
    :param threshold:
    :return:
    """
    return np.max(roi) >= threshold


def export_frame(video_name, frame_number, frame, state, images_path):
    """
    Exports a snapshot/frame to a PNG file for validation purposes.

    :param video_name:
    :param frame_number:
    :param frame:
    :param state:
    :param images_path:
    :return:
    """
    filename = f"{os.path.splitext(video_name)[0]}_frame{frame_number}_{state}.png"
    save_to = f"{images_path}/{filename}"
    cv2.imwrite(save_to, frame)


def get_led_states(rois_df):
    """
    For every row in the rois_df (holds the movie filename and roi info), the LED
    state is extracted and is stored. All saved info is later saved to a file.

    :param rois_df: every row had a filename (mp4/avi) and a ROI
    :return:
    """
    vid_folder_path = paths["video_folder"]
    snapshot_path = os.path.join(vid_folder_path, "snapshots")
    all_led_states = {}

    for index, row in rois_df.iterrows():
        print(f"Working with video {row['Video']}.")
        video_path = os.path.join(vid_folder_path, row['Video'])
        roi = ast.literal_eval(row['ROI'])
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_number, prev_frame, prev_state = 0, None, 0

        states = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:  # if not successfully read, stop iterating frames of this video
                break

            x, y, w, h = roi  # unpack tuple object
            roi_frame = frame[y:y + h, x:x + w]  # slice the roi data from the entire frame
            state = is_led_on(roi_frame)  # check if on or off

            # export frames when LED is on ór when the state changes
            if state or (state != prev_state):
                export_frame(row['Video'], frame_number, frame, 'ON' if state else 'OFF', snapshot_path)
                # if there is a ON state, of if there is a change in state, and if there is a previously evaluated
                # frame: also export the previous frame. This ensures the OFF frame before first ON frame is also saved.
                if prev_frame is not None:
                    export_frame(row['Video'], frame_number - 1, prev_frame, 'ON' if prev_state else 'OFF', snapshot_path)

            prev_frame = frame.copy()  # save frame for next iteration
            prev_state = state  # update state
            states.append(int(state))  # save state
            frame_number += 1  # increment frame counter

            print('\r', f"{round(frame_number / total_frames * 100, 2)}% done..", end='')

        # close video and save led states
        cap.release()
        all_led_states[row['Video']] = np.array(states)

    return all_led_states


def main():
    roi_df_path = os.path.join(paths["video_folder"], "video_rois.xlsx")
    pickle_path = os.path.join(paths["video_folder"], "pickle")
    roi_df = pd.read_excel(roi_df_path)  # read the roi dataframe created with the identify_led_rois.py script
    led_states = get_led_states(roi_df)  # get the LED states for all frames of every file
    # save the LED states for every frame of every video file
    with open(f'{pickle_path}/led_states.pickle', "wb") as f:
        pickle.dump(led_states, f, pickle.HIGHEST_PROTOCOL)


# process starts here
if __name__ == '__main__':
    main()
