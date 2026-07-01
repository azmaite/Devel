#!/usr/bin/env python3
"""
Motion-correct individual FlyTrack .mkv recordings (rigid-translation drift).

This is fully independent of `preprocess_flytrack.py`: every selected recording
is decoded and corrected from scratch and all of its outputs are overwritten, so
this can be run before or after preprocessing. When run before, preprocessing
later reuses the corrected outputs; when run after, re-run preprocessing so the
grouped projections are rebuilt from the corrected images.

Workflow
--------
1. Ask for a folder. If it contains a ``mkv_paths.json`` (an existing analysis
   folder) its recorded recordings are listed. Otherwise, if the folder contains
   .mkv files directly (a single recording folder) those are the candidates.
   Otherwise the folder's immediate subfolders are listed (same picker
   `preprocess_flytrack.py` uses) and their top-level .mkv files are used.
2. If there is a single candidate recording it is used automatically; otherwise
   the candidates are listed (marking any that are already motion-corrected) and
   the user picks which to correct by number and/or range (e.g. ``1,3,5-7``) or
   ``all``.
3. For each chosen recording, remove rigid-translation drift and overwrite its
   ``projection_images.h5`` (including the per-frame ``shifts``),
   ``metadata.txt``, ``activity.png``, and ``drift.png``, and write a
   half-resolution, compressed ``summary_video_motioncorrected.mp4`` preview of
   the corrected video (see `flytrack_core._process_mkv()`).

Example usages from the terminal:
        python3 motion_correct_flytrack.py
        python3 motion_correct_flytrack.py --folder path/to/analysis/folder
"""

import os
import sys
import glob
import json
import argparse

import h5py

# put this file's folder on sys.path so `import flytrack_core` works no matter
# how this module is launched (as a script, via `-m`, or imported as
# FlyTrack.<module> from a notebook) or what the working directory is
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from flytrack_core import (MAIN_PATH, MKV_PATHS_NAME, PROJ_H5_NAME,
                           _process_mkv, _select_mkv_paths, _parse_selection)


def motion_correct_flytrack(folder: str = '') -> None:
    """
    Interactively motion-correct selected recordings in a folder.

    Parameters
    ----------
    folder : str, optional
        An analysis folder (with a `mkv_paths.json`) or a plain input folder of
        recording subfolders. If empty, a file dialog is opened to select it.
    """

    # ask for the folder if not provided
    if not folder:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(
            initialdir=MAIN_PATH,
            title='Select an analysis folder or an input folder')

    assert os.path.exists(folder), f'Folder {folder} does not exist'
    assert os.path.isdir(folder), f'Folder {folder} must be a directory'

    # gather the candidate recordings. In order of preference: the recordings
    # listed in a mkv_paths.json (an analysis folder); .mkv files directly in the
    # selected folder (a single recording folder); otherwise the subfolder picker
    # preprocessing uses (a parent input folder of recording subfolders).
    json_path = os.path.join(folder, MKV_PATHS_NAME)
    direct_mkvs = sorted(glob.glob(os.path.join(folder, '*.mkv')))
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            entries = json.load(f)
        mkv_paths = [entry['mkv_path'] for entry in entries]
        print(f'Existing analysis folder: {len(mkv_paths)} recordings from '
              f'{MKV_PATHS_NAME}.')
    elif len(direct_mkvs) > 0:
        mkv_paths = direct_mkvs
        print(f'Found {len(mkv_paths)} .mkv file(s) directly in {folder}.')
    else:
        mkv_paths = _select_mkv_paths(folder)

    if len(mkv_paths) == 0:
        print('No .mkv files found. Exiting.')
        return

    # with a single recording just use it; otherwise ask which to correct
    if len(mkv_paths) == 1:
        print(f'Using the only recording found: '
              f'{os.path.basename(mkv_paths[0])}')
        chosen = [0]
    else:
        chosen = _select_recordings(mkv_paths)
        if len(chosen) == 0:
            print('No recordings selected. Exiting.')
            return

    for i, idx in enumerate(chosen):
        mkv_file = mkv_paths[idx]
        print(f'{i+1}/{len(chosen)}: motion-correcting '
              f'{os.path.basename(mkv_file)}')
        _process_mkv(mkv_file, motion_correct=True)

    print('Done. Re-run preprocess_flytrack.py on the analysis folder to '
          'rebuild the grouped projections from the corrected images.')


def _select_recordings(mkv_paths: list[str]) -> list[int]:
    """
    List candidate recordings and let the user pick which to motion-correct.

    Each recording is shown with a number and, if it already has a `shifts`
    dataset in its `projection_images.h5`, a ``[motion-corrected]`` tag. The user
    picks with a comma/space-separated list of numbers and ranges (e.g.
    ``1,3,5-7``), ``all``, or ``q`` to cancel.

    Parameters
    ----------
    mkv_paths : list of str
        Candidate .mkv file paths.

    Returns
    -------
    list of int
        Sorted, de-duplicated 0-based indices into `mkv_paths`.
    """

    print('\nRecordings:')
    for i, mkv_file in enumerate(mkv_paths):
        tag = '  [motion-corrected]' if _is_corrected(mkv_file) else ''
        print(f'  [{i + 1}] {os.path.basename(mkv_file)}{tag}')

    selection = input("\nWhich recordings to motion-correct? "
                      "e.g. '1,3,5-7', 'all', or 'q' to cancel: ").strip().lower()
    if selection in ('q', 'quit', ''):
        return []
    if selection == 'all':
        return list(range(len(mkv_paths)))
    return _parse_selection(selection, len(mkv_paths))


def _is_corrected(mkv_file: str) -> bool:
    """
    Return whether a recording already has motion-correction outputs.

    True if its `projection_images.h5` exists and contains a `shifts` dataset.

    Parameters
    ----------
    mkv_file : str
        Absolute path to the source .mkv video file.

    Returns
    -------
    bool
        Whether the recording has already been motion-corrected.
    """

    h5_path = os.path.join(os.path.dirname(mkv_file), PROJ_H5_NAME)
    if not os.path.exists(h5_path):
        return False
    try:
        with h5py.File(h5_path, 'r') as f:
            return 'shifts' in f
    except OSError:
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Motion-correct individual FlyTrack .mkv recordings.')

    parser.add_argument('--folder', type=str, default='',
                        help='Analysis folder or input folder '
                             '(opens a dialog if omitted)')

    args = parser.parse_args()

    motion_correct_flytrack(folder=args.folder)
