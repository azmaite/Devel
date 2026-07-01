#!/usr/bin/env python3
"""
Preprocess FlyTrack .mkv muscle recordings and build grouped projection images.

This script never motion-corrects; to remove within-recording rigid drift, run
`motion_correct_flytrack.py` on the relevant recordings (before or after this
script) and then re-run this script, which will pick up the corrected images.

Workflow
--------
1. Ask for an input folder. If the selected folder already contains a
   ``mkv_paths.json`` it is treated as an existing analysis folder: it is reused
   as-is and its recorded .mkv paths are loaded (already-processed recordings
   are skipped in step 3; anything missing is redone). Otherwise a new analysis
   folder (``YYYY-MM-DD_analysis``, renamable later) is created inside it.
2. For a new analysis folder, list the input folder's immediate subfolders
   (excluding analysis folders and any subfolder with no top-level .mkv files)
   and let the user pick which to include by number and/or range
   (e.g. ``1,3,5-7``), or ``0``/``all`` to include every listed subfolder. Each
   chosen subfolder is searched (top level only) for .mkv files. All selected
   paths are written to ``mkv_paths.json`` in the analysis folder under the key
   ``mkv_path``.
3. For each .mkv, check whether its per-recording projection images and
   metadata already exist (``projection_images.h5`` and ``metadata.txt`` in the
   mkv's own folder). For any that are missing, decode the video once
   (4 real frames packed per container frame, 16-bit) and compute:
       - median image (from an evenly-spaced frame subsample; approximate)
       - min projection
       - max projection
       - cumulative mean, mean of squares, and frame count (for the std image)
       - the mean fluorescence of every frame over time (``act``)
   These are saved to ``projection_images.h5`` next to the video, along with the
   whole-video ``std`` image computed from the cumulative mean/mean_sq. The mean
   fluorescence is plotted over time and saved as ``activity.png``, the median,
   (max projection - median), and std are saved as the auto-contrasted
   ``median.png``, ``max_activity.png`` and ``std.png`` preview images, and a
   small half-resolution (2x2-averaged), compressed ``summary_video.mp4`` preview
   is written for quick visual inspection. Video size (x, y), length (n), and fps
   are saved to ``metadata.txt`` and added to ``mkv_paths.json`` as ``size``,
   ``num_frames``, and ``fps``. Recordings that lack the ``std`` dataset (from
   before it was added) cannot be backfilled and are simply reprocessed.
4. Align the recordings to each other, in case the sample moved between them:
   an integer-pixel shift is estimated (high-pass phase correlation) that aligns
   each recording's median to the first recording's median. The same shift is
   applied to all of that recording's projection images before grouping, and is
   stored in ``mkv_paths.json`` as ``group_shift`` ([dy, dx]).
5. Across all selected recordings, compute:
       - ``median``      : pixel-wise median of every recording's median image
       - ``max_proj``    : pixel-wise maximum of every recording's max projection
       - ``std``         : std image rebuilt from the combined mean/mean_sq/count
   These are saved to ``projection_images_grouped.h5`` in the analysis folder,
   and its ``median.png``, ``max_activity.png`` and ``std.png`` preview images
   are written next to it.
6. Save ``activity.jpg`` (max_proj - median) in the analysis folder,
   auto-contrasted between its minimum and 95th percentile.

Example usages from the terminal:
        python3 preprocess_flytrack.py
        python3 preprocess_flytrack.py --input_folder path/to/input/folder
"""

import os
import sys
import json
import datetime
import argparse

import numpy as np
import cv2
import h5py
import tqdm
from scipy import ndimage
from skimage.registration import phase_cross_correlation

# put this file's folder on sys.path so `import flytrack_core` works no matter
# how this module is launched (as a script, via `-m`, or imported as
# FlyTrack.<module> from a notebook) or what the working directory is
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from flytrack_core import (MAIN_PATH, MKV_PATHS_NAME, PROJ_H5_NAME,
                           GROUPED_H5_NAME, ACTIVITY_NAME, _highpass,
                           _process_mkv, _outputs_exist, _read_existing_metadata,
                           _select_mkv_paths, _write_pngs,
                           _autocontrast_8bit)


def preprocess_flytrack(input_folder: str = '') -> None:
    """
    Run the full FlyTrack preprocessing workflow (see module docstring).

    Parameters
    ----------
    input_folder : str, optional
        Path to the input folder. If empty, a file dialog is opened for the
        user to select it interactively.
    """

    # ask for the input folder if not provided
    if not input_folder:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        input_folder = filedialog.askdirectory(initialdir=MAIN_PATH,
                                                title='Select input folder')

    assert os.path.exists(input_folder), f'Input folder {input_folder} does not exist'
    assert os.path.isdir(input_folder), f'Input folder {input_folder} must be a directory'

    # if the selected folder already contains a mkv_paths.json it is an existing
    # analysis folder: reuse it and its recorded mkv paths (recordings that are
    # already processed are skipped below, missing ones are redone). Otherwise
    # create a new analysis folder (YYYY-MM-DD_analysis) and select recordings.
    json_path = os.path.join(input_folder, MKV_PATHS_NAME)
    if os.path.exists(json_path):
        analysis_folder = input_folder
        with open(json_path, 'r') as f:
            entries = json.load(f)
        print(f'Existing analysis folder detected: {analysis_folder}')
        print(f'Loaded {len(entries)} .mkv entries from {MKV_PATHS_NAME}.')
    else:
        analysis_name = datetime.date.today().strftime('%Y-%m-%d') + '_analysis'
        analysis_folder = os.path.join(input_folder, analysis_name)
        os.makedirs(analysis_folder, exist_ok=True)
        print(f'Analysis folder: {analysis_folder}')

        # let the user pick which subfolders of the input folder to include
        mkv_paths = _select_mkv_paths(input_folder)
        if len(mkv_paths) == 0:
            print('No .mkv files selected. Exiting.')
            return
        print(f'Selected {len(mkv_paths)} .mkv files.')
        entries = [{'mkv_path': p} for p in mkv_paths]

    if len(entries) == 0:
        print('No .mkv files to process. Exiting.')
        return

    # write the selected paths to mkv_paths.json (metadata added below)
    json_path = os.path.join(analysis_folder, MKV_PATHS_NAME)
    with open(json_path, 'w') as f:
        json.dump(entries, f, indent=2)

    # process each recording (skipping any that already have outputs) and
    # collect its metadata back into the json entries
    for i, entry in enumerate(entries):
        mkv_file = entry['mkv_path']
        name = os.path.basename(mkv_file)
        dir = os.path.dirname(mkv_file)

        if _outputs_exist(dir):
            print(f'{i+1}/{len(entries)}: {name} already processed, skip.')
            size, num_frames, fps = _read_existing_metadata(mkv_file)
        else:
            print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - '
                  f'{i+1}/{len(entries)}: processing {name}')
            size, num_frames, fps = _process_mkv(mkv_file)

        entry['size'] = size
        entry['num_frames'] = num_frames
        entry['fps'] = fps

        # rewrite the json after each recording so progress is not lost
        with open(json_path, 'w') as f:
            json.dump(entries, f, indent=2)

    # align the recordings to each other, then build the grouped projections
    _align_group(entries, analysis_folder)
    _build_grouped(entries, analysis_folder)


def _align_group(entries: list[dict],
                 analysis_folder: str) -> None:
    """
    Estimate a per-recording shift that aligns each median to the first's.

    In case the sample moved between recordings, an integer-pixel rigid shift is
    estimated (high-pass phase correlation) that aligns every recording's median
    image to the first recording's median image. The shift is stored in each
    entry as `group_shift` ([dy, dx]) and `mkv_paths.json` is rewritten. The same
    shift is later applied to all of that recording's projection images in
    `_build_grouped()`.

    Parameters
    ----------
    entries : list of dict
        The mkv_paths.json entries, each with an 'mkv_path' key.
    analysis_folder : str
        Folder holding `mkv_paths.json`.
    """

    # high-pass of the first recording's median is the alignment reference
    ref_h5 = os.path.join(os.path.dirname(entries[0]['mkv_path']), PROJ_H5_NAME)
    with h5py.File(ref_h5, 'r') as f:
        ref_hp = _highpass(f['median'][:])  # type: ignore[index]

    print('Aligning recordings to the first recording...')
    for entry in entries:
        h5_path = os.path.join(os.path.dirname(entry['mkv_path']), PROJ_H5_NAME)
        with h5py.File(h5_path, 'r') as f:
            median = f['median'][:]  # type: ignore[index]
        # phase_cross_correlation returns the (dy, dx) that aligns median to ref;
        # with upsample_factor=1 it is already whole-pixel (rounded to be safe).
        # normalization=None is plain cross-correlation; the default 'phase'
        # whitening collapses to (0, 0) on these noisy high-pass images.
        shift, _, _ = phase_cross_correlation(
            ref_hp, _highpass(median), upsample_factor=1, normalization=None)
        entry['group_shift'] = [int(round(shift[0])), int(round(shift[1]))]

    with open(os.path.join(analysis_folder, MKV_PATHS_NAME), 'w') as f:
        json.dump(entries, f, indent=2)


def _shift_int(img: np.ndarray, shift: list[int]) -> np.ndarray:
    """
    Shift `img` by a whole-pixel `shift` without interpolation.

    Uses nearest-neighbour (order 0) so pixel values are preserved exactly and
    edges are filled by replication. A zero shift returns the input unchanged.

    Parameters
    ----------
    img : np.ndarray
        Image to shift.
    shift : list of int
        Whole-pixel shift as [dy, dx].

    Returns
    -------
    np.ndarray
        The shifted image, same dtype as `img`.
    """

    if shift[0] == 0 and shift[1] == 0:
        return img
    return ndimage.shift(img, shift, order=0, mode='nearest')


def _build_grouped(entries: list[dict],
                   analysis_folder: str) -> None:
    """
    Combine per-recording projections into grouped outputs.

    Reads every recording's `projection_images.h5`, applies that recording's
    `group_shift` (from `_align_group()`) to each of its projection images, and
    computes the pixel-wise median of all median images (`median`), the
    pixel-wise maximum of all max projections (`max_proj`), and the std image
    rebuilt from the combined cumulative mean, mean of squares, and frame counts.
    These are written to `projection_images_grouped.h5`, its preview PNGs are
    saved with `_write_pngs()`, and an auto-contrasted `max_proj - median`
    activity image is saved as a jpg.

    Parameters
    ----------
    entries : list of dict
        The mkv_paths.json entries, each with an 'mkv_path' and 'group_shift'.
    analysis_folder : str
        Folder the grouped outputs are written to.
    """

    medians: list[np.ndarray] = []
    max_proj: np.ndarray | None = None
    weighted_mean: np.ndarray | None = None
    weighted_mean_sq: np.ndarray | None = None
    total_count = 0

    print('Building grouped projection images...')
    for entry in tqdm.tqdm(entries):
        shift = entry.get('group_shift', [0, 0])
        h5_path = os.path.join(os.path.dirname(entry['mkv_path']), PROJ_H5_NAME)
        with h5py.File(h5_path, 'r') as f:
            median = _shift_int(f['median'][:], shift)     # type: ignore[index]
            mx = _shift_int(f['max_proj'][:], shift)       # type: ignore[index]
            mean = _shift_int(f['mean'][:], shift)         # type: ignore[index]
            mean_sq = _shift_int(f['mean_sq'][:], shift)   # type: ignore[index]
            count = int(f['frame_count'][()])              # type: ignore[index]

        # collect medians (combined below by pixel-wise median) and running max
        medians.append(median)
        max_proj = mx if max_proj is None else np.maximum(max_proj, mx)

        # accumulate count-weighted mean and mean_sq to combine the std image
        if weighted_mean is None:
            weighted_mean = mean * count
            weighted_mean_sq = mean_sq * count
        else:
            weighted_mean += mean * count
            weighted_mean_sq += mean_sq * count
        total_count += count

    # pixel-wise median of every recording's median image
    median = np.median(np.stack(medians), axis=0)

    # combine into a single std image: std = sqrt(E[x^2] - E[x]^2)
    grand_mean = weighted_mean / total_count
    grand_mean_sq = weighted_mean_sq / total_count
    std = np.sqrt(np.clip(grand_mean_sq - grand_mean ** 2, 0, None))

    # save grouped projection images
    grouped_path = os.path.join(analysis_folder, GROUPED_H5_NAME)
    with h5py.File(grouped_path, 'w') as f:
        f.create_dataset('median', data=median)
        f.create_dataset('max_proj', data=max_proj)
        f.create_dataset('std', data=std)
    print(f'Saved grouped projections to {grouped_path}')

    # save the grouped median / max-activity / std preview PNGs
    _write_pngs(grouped_path)

    # activity image: max_proj - median, auto-contrasted min..95th percentile
    activity = np.clip(max_proj.astype(np.float64) - median.astype(np.float64),
                       0, None)
    activity_8bit = _autocontrast_8bit(activity)
    activity_path = os.path.join(analysis_folder, ACTIVITY_NAME)
    cv2.imwrite(activity_path, activity_8bit)
    print(f'Saved activity image to {activity_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess FlyTrack .mkv recordings and build grouped '
                    'projection images.')

    parser.add_argument('--input_folder', type=str, default='',
                        help='Path to the input folder (opens a dialog if omitted)')

    args = parser.parse_args()

    preprocess_flytrack(input_folder=args.input_folder)
