""" 
Function to preprocess long-term muscle recordings in pupae in bulk,
processing all the .mkv videos in a subfolders. 
This includes:
- generate "small" mp4 files from all the raw .mkv videos
- save an image with the overall change in fluorescence intensity across the video
- save a max projection and min projection image,
- make a grid image with the max projection for each video in the folder (if subfolders pressent)

Example usages from the terminal: 
        python3 preprocess_pupa_muscle.py
        python3 preprocess_pupa_muscle.py --folder_path path/to/recordings/folder
        
"""

import os
import numpy as np
import av
import cv2
import h5py
import datetime
import argparse

import matplotlib
matplotlib.use('Agg')  # Force Matplotlib to use a non-interactive backend
import matplotlib.pyplot as plt

MAIN_PATH = '/mnt/labserver/data/MA/Development_project/Pupa_muscle_long_recordings/'

# set the saving names
MP4_SAVE_NAME = 'summary_video.mp4'
ACT_SAVE_NAME = 'activity.png'
MIN_SAVE_NAME = 'min_projection.png'
MAX_SAVE_NAME = 'max_projection.png'
FIGS_FOLDER = '00_Figs'
GRID_SAVE_NAME = 'max_projection_grid.png'


def preprocess_pupa_muscle(folder_path=''):
    """ folder_path can be a single recording folder (with one .mkv file inside it)
    or a parent folder with subfolders (each with one .mkv file inside)"""

    # if path is empty, open a file dialog to select the parent folder
    if not folder_path:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        folder_path = filedialog.askdirectory(initialdir=MAIN_PATH, title='Select folder containing recordings')

    # check that the path is valid
    assert os.path.exists(folder_path), f'Folder path {folder_path} does not exist'
    assert os.path.isdir(folder_path), f'Folder path {folder_path} must be a directory'

    # get all .mkv files in the folder and subfolders
    mkv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.mkv'):
                mkv_files.append(os.path.join(root, file))

    # sort the mkv files by name
    mkv_files.sort()

    # check which videos have already been processed and skip
    skip_idx = []
    for i, mkv_file in enumerate(mkv_files):

        # check if all the outputs exist - if so, skip
        dir = os.path.dirname(mkv_file)
        if os.path.exists(os.path.join(dir, MP4_SAVE_NAME)) & \
            os.path.exists(os.path.join(dir, ACT_SAVE_NAME)) & \
            os.path.exists(os.path.join(dir, MIN_SAVE_NAME)) & \
            os.path.exists(os.path.join(dir, MAX_SAVE_NAME)):
                print(f'Already processed {os.path.basename(mkv_file)}, skip.')
                skip_idx.append(i)
    mkv_files_todo = [f for i, f in enumerate(mkv_files) if i not in skip_idx]
    print('')

    # process each image - only if not already done
    if len(mkv_files_todo) == 0:
        print('All videos already processed, make grid image.')
    else:
        for i, mkv_file in enumerate(mkv_files_todo):
            name = os.path.basename(mkv_file)
            print(f'Processing {i+1}/{len(mkv_files_todo)}: {name}')
            _preprocess(mkv_file)

    # save grid image (if there are subfolders)
    if len(mkv_files) > 1:

        # get the number of subplot columns needed (4 rows)
        rows = 4
        cols = int(np.ceil(len(mkv_files) / rows))

        # plot grid of max projection image for each video
        fig, axes = plt.subplots(rows, cols, figsize=(22, 12))
        axes = axes.flatten()

        for i, vid in enumerate(mkv_files):
            if i >= len(axes):
                break

            # load the max projection image
            dir = os.path.dirname(vid)
            max_proj_path = os.path.join(dir, MAX_SAVE_NAME)
            frame_0 = cv2.imread(max_proj_path, cv2.IMREAD_UNCHANGED)

            # if image is horizontal, rotate the image 90 degrees clockwise
            if frame_0.shape[1] > frame_0.shape[0]:
                frame_0 = np.rot90(frame_0, k=-1)

            # get the date/time of the video from the filename
            time_t_str = os.path.basename(vid)[-24:-15]
            time_t = datetime.datetime.strptime(time_t_str, '%m%d_%H%M')
            # calculate hours since puparium formation
            time_0_str = '_'.join(folder_path.split('_')[-2:])
            time_0 = datetime.datetime.strptime(time_0_str, '%m-%d_%Hh%M')
            hours = np.round((time_t - time_0).total_seconds() / 3600).astype(int)

            # Plot the average frame
            axes[i].imshow(frame_0, cmap='gray', vmin=50, vmax=500)
            axes[i].axis('off')
            axes[i].set_title(f'{time_t_str} - {hours}h APF', fontsize=10)

        plt.tight_layout()

        # create figs folder if it doesn't exist
        if not os.path.exists(os.path.join(folder_path, FIGS_FOLDER)):
            os.makedirs(os.path.join(folder_path, FIGS_FOLDER))
        
        # save
        save_path = os.path.join(folder_path, FIGS_FOLDER, GRID_SAVE_NAME)
        plt.savefig(save_path, dpi=300)
        plt.close()



def _preprocess(video):

    # get the directory of the video (to save the outputs in the same folder)
    dir = os.path.dirname(video)

    # get metadata file
    metadata = video.replace('_raw_tiff.mkv', '_tif_metadata.h5')

    # extract timestamps from metadata to get the frame rate
    with h5py.File(metadata, 'r') as f:
        timestamps = f['timestamps'][:] #in MICROSECONDS
    # extract the number from the timestamps (saved as objects)
    timestamps = timestamps.astype(np.int64)
    # set time as relative
    timestamps = timestamps - timestamps[0]
    # add a full day of microseconds if the values are negative
    timestamps[timestamps < 0] = timestamps[timestamps < 0] + 24*3600*1e6
    # convert to seconds
    timestamps = timestamps / 1e6

    # get frame rate
    frame_rate = np.round(1 / np.diff(timestamps).mean()).astype(int)


    # Load the first frame (to get frame shape)
    container = av.open(video)
    # Grab the first video frame
    frame_0 = next(container.decode(video=0))
    frame_0 = frame_0.to_ndarray(format='rgba64le')
    frame_0 = frame_0[..., 0]
    container.close()

    # initialize the mean fluorescence (activity)
    act = []

    # initialize the max projection and min projection images
    max_proj = np.zeros_like(frame_0, dtype=np.uint16)
    min_proj = np.full_like(frame_0, fill_value=65535, dtype=np.uint16)

    # Define the codec and create VideoWriter object for the mp4 video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    mp4_path = os.path.join(dir, MP4_SAVE_NAME)
    mp4 = cv2.VideoWriter(mp4_path, fourcc, frame_rate, frame_0.shape[::-1])

    # iterate through all the frames
    container = av.open(video)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO" 

    for frame in container.decode(stream):

        # extract 4 channels (each frame includes 4 real frames)
        img = frame.to_ndarray(format='rgba64le')
        for j in range(img.shape[2]):

            img_i = img[...,j]

            # get mean activity
            act.append(np.mean(img[:,:,j].astype(np.int32)))

            # convert to uint8 by normalizing 0-255 (for mp4 video)
            clip = 600
            img_i = np.clip(img_i, 0, clip) / clip * 255
            img_i = img_i.astype(np.uint8)

            # save to video
            mp4.write(cv2.cvtColor(img_i, cv2.COLOR_RGB2BGR))

            # update max projection and min projection
            max_proj = np.maximum(max_proj, img[...,j])
            min_proj = np.minimum(min_proj, img[...,j])


    mp4.release()
    container.close()


    # save the overall fluorescence across the video as a png image
    act = np.array(act)
    plt.figure(figsize=(10, 3))
    plt.plot(timestamps/60, act, linewidth=0.5) # in minutes
    plt.xlim(0, timestamps[-1]/60)
    plt.xlabel('Time (min)')
    plt.ylabel('Mean fluorescence')
    plt.title(os.path.basename(video))
    
    # Save the figure
    plt.savefig(os.path.join(dir, ACT_SAVE_NAME), dpi=300)
    
    # Close the plot to free up memory
    plt.close()


    # save the max projection and min projection images (clip for visualization)
    clip = 800
    max_proj_clip = np.clip(max_proj, 0, clip) / clip * 255
    max_proj_clip = max_proj_clip.astype(np.uint8)
    clip = 600
    min_proj_clip = np.clip(min_proj, 0, clip) / clip * 255
    min_proj_clip = min_proj_clip.astype(np.uint8)
    cv2.imwrite(os.path.join(dir, MAX_SAVE_NAME), max_proj_clip)
    cv2.imwrite(os.path.join(dir, MIN_SAVE_NAME), min_proj_clip)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MKV video to MP4 and save difference in fluorescence as PNG")

    parser.add_argument("--folder_path", type=str, default='',
                        help="Path to folder containing MKV videos (can be a parent folder with subfolders)")

    args = parser.parse_args()

    preprocess_pupa_muscle(
        folder_path=args.folder_path,
    )

