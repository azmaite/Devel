""" Function to generate "small" mp4 files from all the raw .mkv videos in a folder.
Also save an image with the overall change in fluorescence intensity across the video, 
to help with quick visualization of the data (and to know when activity happened).

Example usages from the terminal: 
        python3 mkv2mp4_bulk.py
        python3 mkv2mp4_bulk.py --folder_path path/to/recordings/folder
        
"""

import os
import numpy as np
import av
import cv2
import h5py
import argparse

import matplotlib
matplotlib.use('Agg')  # Force Matplotlib to use a non-interactive backend
import matplotlib.pyplot as plt

MAIN_PATH = '/mnt/labserver/data/MA/Development_project'

# set the saving names for the mp4 video and the diff image
MP4_SAVE_NAME = 'summary_video.mp4'
DIF_SAVE_NAME = 'activity_diff.png'


def mkv2mp4_bulk(folder_path=''):
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
                # check if the matching mp4 video already exists, if so skip
                if os.path.exists(os.path.join(root, MP4_SAVE_NAME)):
                    print(f'MP4 video already exists for {file}, skipping...')
                else:
                    mkv_files.append(os.path.join(root, file))
    print('')

    # sort the mkv files by name
    mkv_files.sort()

    # convert each .mkv file to .mp4
    for i, mkv_file in enumerate(mkv_files):
        name = os.path.basename(mkv_file)
        print(f'Converting {i+1}/{len(mkv_files)} to mp4: {name}')
        _mkv2mp4(mkv_file)



def _mkv2mp4(video):

    # get metadata file
    metadata = video.replace('_raw_tiff.mkv', '_tif_metadata.h5')

    # extract timestamps from metadata to get the frame rate
    with h5py.File(metadata, 'r') as f:
        timestamps = f['timestamps'][:] #in MICROSECONDS
    # extract the number from the timestamps (saved as objects)
    timestamps = timestamps.astype(np.int64)
    # set time as relative
    timestamps = timestamps - timestamps[0]
    # convert to seconds
    timestamps = timestamps / 1e6

    # get frame rate
    frame_rate = np.round(1 / np.diff(timestamps).mean()).astype(int)


    # Load the first frame as reference (to calculate differences in fluorescence across the video)
    container = av.open(video)
    # Grab the first video frame
    frame_0 = next(container.decode(video=0))
    frame_0 = frame_0.to_ndarray(format='rgba64le')
    frame_0 = frame_0[..., 0]
    # convert to int32
    frame_0 = frame_0.astype(np.int32)
    container.close()

    # initialize the difference in fluorescence with frame_0 for each frame
    dif = []

    # Define the codec and create VideoWriter object for the mp4 video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    dir = os.path.dirname(video)
    mp4_path = os.path.join(dir, MP4_SAVE_NAME)
    mp4 = cv2.VideoWriter(mp4_path, fourcc, frame_rate, frame_0.shape[::-1])

    # iterate through all the frames and get the differenes in fluorescence with frame_0
    container = av.open(video)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO" 

    for frame in container.decode(stream):

        # extract 4 channels (each frame includes 4 real frames)
        img = frame.to_ndarray(format='rgba64le')
        for j in range(img.shape[2]):

            img_i = img[...,j]

            # get difference with frame_0
            dif.append(np.mean(np.abs(img[:,:,j].astype(np.int32) - frame_0)))

            # convert to uint8 by normalizing 0-255 (for mp4 video)
            clip = 600
            img_i = np.clip(img_i, 0, clip) / clip * 255
            img_i = img_i.astype(np.uint8)

            # save to video
            mp4.write(cv2.cvtColor(img_i, cv2.COLOR_RGB2BGR))

    mp4.release()
    container.close()


    # save the difference in fluorescence across the video as a png image
    dif = np.array(dif)
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, dif, linewidth=1)
    plt.xlim(timestamps[1], timestamps[-1])
    plt.xlabel('Time (s)')
    plt.ylabel('Mean abs difference in fluorescence')
    plt.title(video.replace(MAIN_PATH, ""))
    
    # Save the figure
    plt.savefig(os.path.join(dir, DIF_SAVE_NAME), dpi=300)
    
    # Close the plot to free up memory
    plt.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MKV video to MP4 and save difference in fluorescence as PNG")

    parser.add_argument("--folder_path", type=str, default='',
                        help="Path to folder containing MKV videos (can be a parent folder with subfolders)")

    args = parser.parse_args()

    mkv2mp4_bulk(
        folder_path=args.folder_path,
    )

