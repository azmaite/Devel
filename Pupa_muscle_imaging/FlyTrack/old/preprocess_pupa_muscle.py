#!/usr/bin/env python3
"""
Preprocess long-term fluorescent muscle recordings in Drosophila pupae in bulk.

Operates on a folder containing one or more subfolders, each holding a single
16-bit .mkv video (4 real frames packed per container frame, recorded at 40 Hz)
and a matching .h5 timestamp file. For each recording the script produces:

- summary_video.mp4      : 8-bit compressed preview video
- activity.png           : mean fluorescence over time
- max_projection.png     : pixel-wise maximum across all frames
- min_projection.png     : pixel-wise minimum across all frames
- median.png             : per-pixel median frame (preview, clipped)
- max_min_proj.h5        : unclipped max, min, median, and std projections

If multiple recordings are found and the parent folder name encodes the
puparium-formation time (format ``MM-DD_HHhMM``), a grid of max-projection
images labelled with hours after puparium formation (APF) is also saved to
``00_Figs/max_projection_grid.png``.

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
import h5py

import matplotlib
matplotlib.use('Agg')  # Force Matplotlib to use a non-interactive backend
import matplotlib.pyplot as plt

import utils

MAIN_PATH = '/mnt/labserver/data/MA/Development_project/Pupa_muscle_long_recordings/'

# set the saving names
MP4_SAVE_NAME = 'summary_video.mp4'
ACT_SAVE_NAME = 'activity.png'
MIN_SAVE_NAME = 'min_projection.png'
MAX_SAVE_NAME = 'max_projection.png'
MEDIAN_SAVE_NAME = 'median.png'
H5_SAVE_NAME = 'max_min_proj.h5'
FIGS_FOLDER = '00_Figs'
GRID_SAVE_NAME = 'max_projection_grid.png'


def preprocess_pupa_muscle(folder_path: str = '') -> None:
    """
    Preprocess all .mkv recordings found under `folder_path`.

    Discovers every .mkv file recursively, skips recordings whose outputs
    already exist, and delegates per-file processing to `_preprocess`. After
    all files are processed, generates a grid summary image when multiple
    recordings are present.

    Parameters
    ----------
    folder_path : str, optional
        Path to a single recording folder or a parent folder containing
        recording subfolders. If empty, opens a file-dialog for the user to
        select the folder interactively.
    """

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
    mkv_files = utils.get_mkv_list(folder_path=None)

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
        print('All videos already processed. Exiting.')
    else:
        for i, mkv_file in enumerate(mkv_files_todo):
            name = os.path.basename(mkv_file)
            # print the name of the video being processed and the time
            print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - {i+1}/{len(mkv_files_todo)}: {name}')
            _preprocess(mkv_file)

    # save grid image (if there are subfolders and
    # the puparium formation time can be extracted from the folder name)
    if len(mkv_files) > 1:
        _save_grid_image(folder_path, mkv_files)




def _preprocess(mkv_file: str) -> None:
    """
    Process a single .mkv recording and write all output files to its folder.

    Reads every packed frame (4 real frames per container frame), builds the
    mp4 preview, computes per-pixel projections (max, min, median via
    `np.partition`, std), and saves them as PNG previews and an unclipped
    HDF5 file. Also plots mean fluorescence over time.

    Parameters
    ----------
    mkv_file : str
        Absolute path to the source .mkv video file.
    """
    # get the directory of the mkv_file video (to save the outputs in the same folder)
    dir = os.path.dirname(mkv_file)

    # get metadata file
    metadata = mkv_file.replace('_raw_tiff.mkv', '_tif_metadata.h5')

    timestamps, frame_rate = utils.load_timestamps(mkv_file)

    # Load the first frame (to get frame shape)
    container = av.open(mkv_file)
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

    # initialize the frames list to save the frames for median calculation
    frames_array = np.zeros((24000, frame_0.shape[0], frame_0.shape[1]), dtype=np.uint16)

    # Define the codec and create VideoWriter object for the mp4 video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    mp4_path = os.path.join(dir, MP4_SAVE_NAME)
    mp4 = cv2.VideoWriter(mp4_path, fourcc, frame_rate, frame_0.shape[::-1])

    # iterate through all the frames
    container = av.open(mkv_file)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO" 

    n = 0
    for frame in container.decode(stream):

        # extract 4 channels (each frame includes 4 real frames)
        img = frame.to_ndarray(format='rgba64le')
        for j in range(img.shape[2]):

            img_i = img[...,j]

            # get mean activity
            act.append(np.mean(img[:,:,j].astype(np.int32)))

            # add to frames list for median calculation
            frames_array[n] = img_i

            # convert to uint8 by normalizing 0-255 (for mp4 video)
            clip = 600
            img_i_mp4 = np.clip(img_i, 0, clip) / clip * 255
            img_i_mp4 = img_i_mp4.astype(np.uint8)

            # save to video
            mp4.write(cv2.cvtColor(img_i_mp4, cv2.COLOR_RGB2BGR))

            # update max projection and min projection
            max_proj = np.maximum(max_proj, img[...,j])
            min_proj = np.minimum(min_proj, img[...,j])

            n += 1


    mp4.release()
    container.close()

    print('Finished creating video')

    # get std image
    std_frame = np.std(frames_array, axis=0)

    # save the overall fluorescence across the video as a png image
    act = np.array(act)
    if len(timestamps) != len(act):
        print(f'Warning: number of timestamps ({len(timestamps)}) does not match number of frames ({len(act)}), LOST FRAMES.')
        timestamps = timestamps[:len(act)]  # trim timestamps to match the number of frames
    plt.figure(figsize=(10, 3))
    plt.plot(timestamps/60, act, linewidth=0.5) # in minutes
    plt.xlim(0, timestamps[-1]/60)
    plt.xlabel('Time (min)')
    plt.ylabel('Mean fluorescence')
    plt.title(os.path.basename(mkv_file))
    plt.savefig(os.path.join(dir, ACT_SAVE_NAME), dpi=300)
    plt.close()


    # save the min, max projection and median images as png (clip for visualization)
    clip = 800
    max_proj_clip = np.clip(max_proj, 0, clip) / clip * 255
    max_proj_clip = max_proj_clip.astype(np.uint8)
    clip = 600
    min_proj_clip = np.clip(min_proj, 0, clip) / clip * 255
    min_proj_clip = min_proj_clip.astype(np.uint8)
    cv2.imwrite(os.path.join(dir, MAX_SAVE_NAME), max_proj_clip)
    cv2.imwrite(os.path.join(dir, MIN_SAVE_NAME), min_proj_clip)

    # get median frame - use np.partition because it's faster than np.median for large arrays (does not sort the whole array, just finds the median value)
    print('Calculating median frame...')
    median_frame = np.partition(frames_array, frames_array.shape[0] // 2, axis=0)[frames_array.shape[0] // 2]
    median_clip = np.clip(median_frame, 0, clip) / clip * 255
    median_clip = median_clip.astype(np.uint8)
    cv2.imwrite(os.path.join(dir, MEDIAN_SAVE_NAME), median_clip)
    print('Finished calculating median frame. Saving all outputs in h5 file...')

    # save also as h5 file (without clipping, for analysis), as well as the std image
    h5_path = os.path.join(dir, H5_SAVE_NAME)
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('max_proj', data=max_proj)
        f.create_dataset('min_proj', data=min_proj)
        f.create_dataset('median', data=median_frame)
        f.create_dataset('std', data=std_frame)


def _save_grid_image(folder_path: str, mkv_files: list[str]) -> None:
    """
    Save a grid of max-projection images labelled with hours APF.

    Extracts the puparium formation time from the last two underscore-delimited
    tokens of `folder_path` (expected format ``MM-DD_HHhMM``). If the format
    does not match, the function prints a warning and returns without saving.
    The grid is arranged in 4 rows and saved to
    ``<folder_path>/00_Figs/max_projection_grid.png``.

    Parameters
    ----------
    folder_path : str
        Parent folder whose name encodes the puparium formation time.
    mkv_files : list of str
        Ordered list of .mkv paths, one per subplot.
    """
    # try to extract the puparium formation time from the folder name
    # if not possible, skip the grid image generation
    try:
        time_0_str = '_'.join(folder_path.split('_')[-2:])
        time_0 = datetime.datetime.strptime(time_0_str, '%m-%d_%Hh%M')
    except ValueError:
        print("Could not extract puparium formation time from folder name.")
        print("  Will not generate grid image.")
        return
    
    print(f'Generating grid image for {len(mkv_files)} videos...')

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MKV video to MP4 and save difference in fluorescence as PNG")

    parser.add_argument("--folder_path", type=str, default='',
                        help="Path to folder containing MKV videos (can be a parent folder with subfolders)")

    args = parser.parse_args()

    preprocess_pupa_muscle(
        folder_path=args.folder_path,
    )

