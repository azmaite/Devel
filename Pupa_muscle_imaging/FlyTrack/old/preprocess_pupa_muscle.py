#!/usr/bin/env python3
"""
Preprocess long-term fluorescent muscle recordings in Drosophila pupae in bulk.

Operates on a folder containing one or more subfolders, each holding a single
16-bit .mkv video (4 real frames packed per container frame, recorded at 40 Hz)
and a matching .h5 timestamp file. For each recording the script produces:

- summary_video.mp4      : 8-bit compressed preview video, at the acquisition
                           frame rate read from ``config_concatenated.yaml``
- max_projection.png     : pixel-wise maximum across all frames
- min_projection.png     : pixel-wise minimum across all frames
- median.png             : per-pixel median frame (preview, clipped)
- max_min_proj.h5        : unclipped max, min, median, and std projections

The mean fluorescence over time (activity) plots of all recordings are saved
together in ``<folder_path>/activity/``, named after the path from `folder_path`
to the recording folder (subfolder names joined with ``_``, e.g.
``Fly1_001`` for ``<folder_path>/Fly1/001/``) plus the hours after puparium
formation (APF), e.g. ``Fly1_001_XXhAPF.png``. If
the parent folder name does not encode the puparium-formation time (format
``MM-DD_HHhMM``), the hAPF suffix is omitted.

If multiple recordings are found and the puparium-formation time can be
extracted, a grid of max-projection images labelled with hours APF is also
saved to ``00_Figs/max_projection_grid.png``. Each grid figure holds at most
4 rows x 12 columns; if there are more recordings, several figures are saved
(``max_projection_grid_1.png``, ``max_projection_grid_2.png``, ...). All grid
images share the same contrast, computed from the saved (unclipped) max
projections: vmin is their overall min and vmax is the median of their
per-image max values.

Example usages from the terminal:
        python3 preprocess_pupa_muscle.py
        python3 preprocess_pupa_muscle.py --folder_path path/to/recordings/folder
"""

import os
import yaml
import numpy as np
import av
import cv2
import h5py
import datetime
import argparse
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')  # Force Matplotlib to use a non-interactive backend
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path('/home/azcorra/git/Devel/Pupa_muscle_imaging')))
import utils

MAIN_PATH = '/mnt/labserver/data/MA/Development_project/Pupa_muscle_long_recordings/'

# set the saving names
MP4_SAVE_NAME = 'summary_video.mp4'
CONFIG_NAME = 'config_concatenated.yaml'  # acquisition config, next to each .mkv
ACT_FOLDER = 'activity'
MIN_SAVE_NAME = 'min_projection.png'
MAX_SAVE_NAME = 'max_projection.png'
MEDIAN_SAVE_NAME = 'median.png'
H5_SAVE_NAME = 'max_min_proj.h5'
FIGS_FOLDER = '00_Figs'
GRID_SAVE_NAME = 'max_projection_grid.png'
GRID_MAX_ROWS = 4
GRID_MAX_COLS = 12


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
        utils.sort_dialog_descending(root)  # newest (YYYYMMDD_...) folders first
        folder_path = filedialog.askdirectory(initialdir=MAIN_PATH, title='Select folder containing recordings')

    # check that the path is valid
    assert os.path.exists(folder_path), f'Folder path {folder_path} does not exist'
    assert os.path.isdir(folder_path), f'Folder path {folder_path} must be a directory'

    # get all .mkv files in the folder and subfolders
    mkv_files = utils.get_mkv_list(folder_path=folder_path)

    # get the puparium formation time from the folder name (None if not possible)
    time_0 = _get_time_0(folder_path)

    # check which videos have already been processed and skip
    skip_idx = []
    for i, mkv_file in enumerate(mkv_files):

        # check if all the outputs exist - if so, skip
        dir = os.path.dirname(mkv_file)
        if os.path.exists(os.path.join(dir, MP4_SAVE_NAME)) & \
            os.path.exists(_get_activity_path(folder_path, mkv_file, time_0)) & \
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
            _preprocess(mkv_file, _get_activity_path(folder_path, mkv_file, time_0))

    # save grid image (if there are subfolders and
    # the puparium formation time can be extracted from the folder name)
    if len(mkv_files) > 1:
        if time_0 is None:
            print('Could not extract puparium formation time from folder name.')
            print('  Will not generate grid image.')
        else:
            _save_grid_image(folder_path, mkv_files, time_0)


def _get_time_0(folder_path: str) -> Optional[datetime.datetime]:
    """
    Extract the puparium formation time from the name of `folder_path`.

    Uses the last two underscore-delimited tokens of the path (expected format
    ``MM-DD_HHhMM``).

    Parameters
    ----------
    folder_path : str
        Parent folder whose name encodes the puparium formation time.

    Returns
    -------
    datetime.datetime or None
        Puparium formation time, or None if the folder name does not match.
    """
    try:
        time_0_str = '_'.join(folder_path.rstrip('/').split('_')[-2:])
        return datetime.datetime.strptime(time_0_str, '%m-%d_%Hh%M')
    except ValueError:
        return None


def _get_recording_time(mkv_file: str) -> tuple[str, datetime.datetime]:
    """
    Get the recording date/time encoded in the .mkv filename.

    Parameters
    ----------
    mkv_file : str
        Path to the .mkv video file.

    Returns
    -------
    time_t_str : str
        Date/time string as found in the filename (``MMDD_HHMM``).
    time_t : datetime.datetime
        Parsed recording time.
    """
    time_t_str = os.path.basename(mkv_file)[-24:-15]
    time_t = datetime.datetime.strptime(time_t_str, '%m%d_%H%M')
    return time_t_str, time_t


def _get_hours_apf(mkv_file: str,
                   time_0: datetime.datetime) -> int:
    """
    Calculate the hours after puparium formation (APF) of a recording.

    Parameters
    ----------
    mkv_file : str
        Path to the .mkv video file (its name encodes the recording time).
    time_0 : datetime.datetime
        Puparium formation time, see `_get_time_0()`.

    Returns
    -------
    int
        Hours APF, rounded to the nearest hour.
    """
    _, time_t = _get_recording_time(mkv_file)
    return int(np.round((time_t - time_0).total_seconds() / 3600))


def _get_activity_path(folder_path: str,
                       mkv_file: str,
                       time_0: Optional[datetime.datetime]) -> str:
    """
    Get the save path of the activity plot of a recording.

    All activity plots are saved in ``<folder_path>/activity/``, named as the
    path from `folder_path` to the recording folder, with the subfolder names
    joined by ``_`` (e.g. ``Fly1_001`` for ``<folder_path>/Fly1/001/``), so
    recordings in identically named subfolders do not overwrite each other.
    The hours APF are appended (``Fly1_001_XXhAPF.png``); if `time_0` is None,
    the hAPF suffix is omitted. If the .mkv sits directly in `folder_path`, the
    name of `folder_path` itself is used.

    Parameters
    ----------
    folder_path : str
        Parent folder containing the recordings.
    mkv_file : str
        Path to the .mkv video file.
    time_0 : datetime.datetime or None
        Puparium formation time, see `_get_time_0()`.

    Returns
    -------
    str
        Path to the activity plot png.
    """
    rec_dir = Path(mkv_file).parent.resolve()
    rel_parts = rec_dir.relative_to(Path(folder_path).resolve()).parts
    rec_name = '_'.join(rel_parts) if rel_parts else rec_dir.name
    if time_0 is not None:
        rec_name += f'_{_get_hours_apf(mkv_file, time_0)}hAPF'
    return os.path.join(folder_path, ACT_FOLDER, rec_name + '.png')


def _get_framerate(mkv_file: str,
                   fallback: float) -> float:
    """
    Get the acquisition frame rate (real frames per second) of a recording.

    Read from ``arduino: framerate`` in the ``config_concatenated.yaml`` saved
    next to the .mkv file (a concatenation of several yaml documents, of which
    the main config is the one holding the ``arduino`` section). If the config
    is missing or has no frame rate, `fallback` is used instead.

    Parameters
    ----------
    mkv_file : str
        Path to the .mkv video file.
    fallback : float
        Frame rate to use if it can't be read from the config, e.g. the one
        estimated from the timestamps by `utils.load_timestamps()`.

    Returns
    -------
    float
        Frame rate in Hz.
    """
    config_path = os.path.join(os.path.dirname(mkv_file), CONFIG_NAME)
    if os.path.exists(config_path):
        with open(config_path) as f:
            for doc in yaml.safe_load_all(f):
                if isinstance(doc, dict) and 'framerate' in doc.get('arduino', {}):
                    return float(doc['arduino']['framerate'])
    print(f'  No frame rate found in {CONFIG_NAME}, using {fallback:.1f} Hz from the timestamps.')
    return fallback


def _preprocess(mkv_file: str,
                act_path: str) -> None:
    """
    Process a single .mkv recording and write all output files to its folder.

    Reads every packed frame (4 real frames per container frame), builds the
    mp4 preview (at the acquisition frame rate, see `_get_framerate()`), computes per-pixel projections (max, min, median via
    `np.partition`, std), and saves them as PNG previews and an unclipped
    HDF5 file. Also plots mean fluorescence over time and saves it to
    `act_path`.

    Parameters
    ----------
    mkv_file : str
        Absolute path to the source .mkv video file.
    act_path : str
        Save path of the activity plot, see `_get_activity_path()`.
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
    # (not all frames, but no space, should be fine)
    max_n = 12000
    frames_array = np.zeros((max_n, frame_0.shape[0], frame_0.shape[1]), dtype=np.uint16)

    # Define the codec and create VideoWriter object for the mp4 video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    mp4_path = os.path.join(dir, MP4_SAVE_NAME)
    frame_rate = _get_framerate(mkv_file, fallback=frame_rate)
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
            if n < len(frames_array):
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
    os.makedirs(os.path.dirname(act_path), exist_ok=True)
    plt.savefig(act_path, dpi=300)
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


def _save_grid_image(folder_path: str,
                     mkv_files: list[str],
                     time_0: datetime.datetime) -> None:
    """
    Save grids of max-projection images labelled with hours APF.

    Each figure holds at most `GRID_MAX_ROWS` rows x `GRID_MAX_COLS` columns;
    if there are more recordings, several figures are saved. The unclipped max
    projections (from the h5 file) are all loaded first and shown with a shared
    contrast: vmin is the min value across all of them and vmax is the median
    of the per-image max values (robust to a few outlier bright pixels).
    Figures are saved to ``<folder_path>/00_Figs/max_projection_grid.png``, or
    ``max_projection_grid_1.png``, ``max_projection_grid_2.png``, ... if more
    than one figure is needed.

    Parameters
    ----------
    folder_path : str
        Parent folder containing the recordings.
    mkv_files : list of str
        Ordered list of .mkv paths, one per subplot.
    time_0 : datetime.datetime
        Puparium formation time, see `_get_time_0()`.
    """
    print(f'Generating grid image for {len(mkv_files)} videos...')

    # load all the unclipped max projection images
    max_projs = []
    for vid in mkv_files:
        _, max_proj = utils.load_min_max_proj(vid)

        # if image is horizontal, rotate the image 90 degrees clockwise
        if max_proj.shape[1] > max_proj.shape[0]:
            max_proj = np.rot90(max_proj, k=-1)
        max_projs.append(max_proj)

    # get the contrast limits across all the max projections: vmin is the global
    # min, vmax is the median of the per-image max (the global max is dominated
    # by a few outlier pixels and makes most images look black)
    vmin = min(int(m.min()) for m in max_projs)
    vmax = float(np.median([m.max() for m in max_projs]))

    # get the grid layout (same for all figures) and split the videos into figures
    rows = GRID_MAX_ROWS
    cols = min(GRID_MAX_COLS, int(np.ceil(len(mkv_files) / rows)))
    n_per_fig = rows * cols
    fig_chunks = [range(i, min(i + n_per_fig, len(mkv_files)))
                  for i in range(0, len(mkv_files), n_per_fig)]

    # create figs folder if it doesn't exist
    os.makedirs(os.path.join(folder_path, FIGS_FOLDER), exist_ok=True)

    for k, chunk in enumerate(fig_chunks):

        # plot grid of max projection image for each video
        fig, axes = plt.subplots(rows, cols, figsize=(22, 12), squeeze=False)
        axes = axes.flatten()
        for ax in axes:
            ax.axis('off')

        for i, idx in enumerate(chunk):
            vid = mkv_files[idx]

            # get the date/time of the video and the hours since puparium formation
            time_t_str, _ = _get_recording_time(vid)
            hours = _get_hours_apf(vid, time_0)

            # Plot the max projection
            axes[i].imshow(max_projs[idx], cmap='gray', vmin=vmin, vmax=vmax)
            axes[i].set_title(f'{time_t_str} - {hours}h APF', fontsize=10)

        plt.tight_layout()

        # save (add figure number to the name if there are several figures)
        save_name = GRID_SAVE_NAME
        if len(fig_chunks) > 1:
            save_name = save_name.replace('.png', f'_{k + 1}.png')
        plt.savefig(os.path.join(folder_path, FIGS_FOLDER, save_name), dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MKV video to MP4 and save difference in fluorescence as PNG")

    parser.add_argument("--folder_path", type=str, default='',
                        help="Path to folder containing MKV videos (can be a parent folder with subfolders)")

    args = parser.parse_args()

    preprocess_pupa_muscle(
        folder_path=args.folder_path,
    )

