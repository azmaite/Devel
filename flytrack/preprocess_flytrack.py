#!/usr/bin/env python3
"""
Preprocess FlyTrack .mkv muscle recordings (and optionally motion-correct them).

Folder structure
----------------
The structure inside the selected folder decides how recordings are handled:

- **Single long recording**: .mkv files sit directly inside subfolders of the
  selected folder (one subfolder per recording, named ``YYYYMMDD_HHMMSS``), e.g.
  ``<exp>_MM-DD_HHhMM/20260730_141937/*.mkv``. The selected folder is the main
  folder. Its recordings are all from one fly, so they are also aligned to each
  other and summarized together in ``00_preanalysis`` (see below).
- **Multiple recordings**: .mkv files sit two levels down, inside first-level
  subfolders named ``FlyNNN``, e.g. ``<exp>/Fly_001/001/*.mkv``. Every
  first-level subfolder is then its own main folder, and each of its recordings
  is processed independently (no ``00_preanalysis``, alignment, or grids).

A .mkv directly inside the selected folder (e.g. a single recording folder was
selected) is also processed independently. .mkv files deeper than two levels
are ignored with a warning. Selecting a ``00_preanalysis`` folder (or an older
``YYYY-MM-DD_analysis`` one, holding a ``mkv_paths.json``) is the same as
selecting its parent. Each of the
first-level subfolders is classified on its own, so a folder mixing both
layouts is handled too.

Per recording (next to each .mkv)
---------------------------------
Recordings whose ``projection_images.h5`` already holds every required dataset
are skipped. Otherwise the video is decoded once (4 real frames packed per
container frame, 16-bit) to compute, saved in ``projection_images.h5``:

    - ``median``   : approximate, from 3000 frames evenly spaced across the video
    - ``min_proj`` / ``max_proj`` : min and max projections
    - ``mean`` / ``mean_sq`` / ``std`` : whole-video mean, mean of squares, std
    - ``act``      : mean fluorescence of every frame
    - ``frame_count``

plus ``metadata.txt``, the preview PNGs ``median.png``, ``max_proj.png``,
``min_proj.png`` and ``std.png`` (each contrasted on its own between its 0.2th
and 99.8th percentiles), and ``summary_video.mp4``: a half-resolution preview at
the acquisition frame rate from ``config_concatenated.yaml``, with one fixed
contrast (0.2th-99.8th percentiles of frames sampled across the video).

The activity plot (mean fluorescence over time) is saved as ``activity.png``
next to the .mkv for independent recordings, but not for a single long
recording, which is summarized as a whole instead (see below).

Single long recording only (in ``<main>/00_preanalysis``)
---------------------------------------------------------
After processing, ``<main>/00_preanalysis/`` (see `common.preanalysis`) gets:

1. ``activity_shared.png`` / ``activity_own.png``: every processed recording's
   activity (re-read from the saved ``act``, no video decoding) in 3-column
   grids in time order, each subplot titled with its hours after puparium
   formation (hAPF). In 'shared' all subplots share the x and y limits and
   touch horizontally, so each row reads as one continuous trace; in 'own'
   each keeps its own y limits.
2. ``mkv_paths.json``: every recording with its ``size``, ``num_frames``,
   ``fps``, and ``group_shift``. The recordings are aligned to each other in
   case the sample moved between them: an integer-pixel shift (high-pass phase
   correlation of the medians, relative to the first recording) is stored as
   ``group_shift`` ([dy, dx]). Used by ``get_segmentation_data.py``.
3. ``projection_images_grouped.h5``: after applying those shifts, the
   pixel-wise median of the medians (``median``), the overall max projection
   (``max_proj``), and the combined ``std`` (used for the segmentation, see
   ``get_segmentation_data.py`` and ``select_manual_roi.py``).
4. ``max_projection_grid.png`` and ``median_grid.png``: every recording's max
   projection / median, labelled with its time and hAPF, at most 4 rows x 12
   columns per figure (several figures, ``..._grid_1.png``, ..., if needed),
   all with one shared contrast (see `common.preanalysis.save_image_grid()`).

The puparium-formation time is read from the main folder name
(``..._MM-DD_HHhMM``, optionally followed by a suffix such as ``_FAIL``) and the
recording time from the recording folder or .mkv name; if either is missing the
hAPF is omitted. To redo only the activity grids, e.g. for a folder processed
before they existed: ``from preprocess_flytrack import save_activity_summary;
save_activity_summary('path/to/main')``. The frequency analysis of the activity
is in ``analysis/flytrack/``.

Options
-------
``--pick`` / ``-p``: list the .mkv files found and ask which to analyze (by
number and/or range, e.g. ``1,3,5-7``, or ``all``). By default all are analyzed.

``--motion-correct`` / ``-m``: motion correction is meant as a second step,
after looking at the summary videos of a first, uncorrected run. With this flag
the .mkv files are listed (as with ``--pick``) and the chosen ones are processed
again with rigid-translation drift correction (see
`flytrack_core._process_mkv()`). Their ``projection_images.h5`` then holds the
corrected images under the usual names, the uncorrected ones with a ``_raw``
suffix (``median_raw``, ``max_proj_raw``, ...), and the per-frame drift as
``shifts`` (plotted in ``drift.png``); the PNGs (and activity plot, for
independent recordings) are redone from the corrected frames, and the corrected ``summary_video.mp4`` is written (the
uncorrected one from the first run is renamed ``summary_video_raw.mp4``).
For a single long recording, the ``00_preanalysis`` outputs are then rebuilt
from the recordings in the existing ``mkv_paths.json`` (or every processed one).

Example usages from the terminal:
        python3 preprocess_flytrack.py
        python3 preprocess_flytrack.py --folder path/to/folder --pick
        python3 preprocess_flytrack.py --folder path/to/folder -m
From python:
        from preprocess_flytrack import preprocess_flytrack
        preanalysis_folder = preprocess_flytrack('path/to/folder')
        preprocess_flytrack('path/to/folder', motion_correct=True)
"""

import os
import re
import sys
import json
import datetime
import argparse
from pathlib import Path

import numpy as np
import h5py
import tqdm
from scipy import ndimage
from skimage.registration import phase_cross_correlation

import matplotlib
matplotlib.use('Agg')  # force a non-interactive backend

# put this file's folder on sys.path so `import flytrack_core` works no matter
# how this module is launched (as a script, via `-m`, or imported from a
# notebook) or what the working directory is
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from flytrack_core import (MAIN_PATH, MKV_PATHS_NAME, PROJ_H5_NAME,
                           GROUPED_H5_NAME, ACT_SAVE_NAME, _highpass, _process_mkv, _outputs_exist,
                           _is_corrected, _read_existing_metadata,
                           _parse_selection, _plot_activity)
import flytrack_io
from common import preanalysis
from common.dialogs import ask_directory
from common.preanalysis import PREANALYSIS_FOLDER, get_hours_apf, get_time_0

# image grids in 00_preanalysis: dataset in projection_images.h5 -> file name
GRID_NAMES = {'max_proj': 'max_projection_grid.png', 'median': 'median_grid.png'}


def preprocess_flytrack(folder: str = '',
                        pick: bool = False,
                        motion_correct: bool = False) -> str | None:
    """
    Run the full FlyTrack preprocessing workflow (see module docstring).

    Parameters
    ----------
    folder : str, optional
        Folder to process (see the module docstring for the expected
        structures). If empty, a file dialog is opened to select it.
    pick : bool, optional
        If True, list the .mkv files found and ask which to analyze.
    motion_correct : bool, optional
        If True, list the .mkv files found and re-process the chosen ones with
        motion correction.

    Returns
    -------
    str or None
        The ``00_preanalysis`` folder of the (last) single long recording
        processed, or None if there was none.
    """

    # ask for the folder if not provided
    if not folder:
        folder = ask_directory(MAIN_PATH, 'Select folder to preprocess')

    assert os.path.exists(folder), f'Folder {folder} does not exist'
    assert os.path.isdir(folder), f'Folder {folder} must be a directory'
    folder = os.path.abspath(folder)

    # a 00_preanalysis (or older analysis) folder stands for the main folder it sits in
    if os.path.exists(os.path.join(folder, MKV_PATHS_NAME)):
        folder = os.path.dirname(folder)
        print(f'Pre-analysis folder selected, using its parent: {folder}')

    # find the recordings and group them by main folder
    groups = _find_recordings(folder)
    all_mkvs = [mkv for group in groups for mkv in group['mkvs']]
    if len(all_mkvs) == 0:
        print(f'No .mkv files found in {folder}. Exiting.')
        return None
    for group in groups:
        kind = 'single long recording' if group['long'] else 'independent recordings'
        print(f'{group["main"]}: {len(group["mkvs"])} .mkv ({kind})')

    # choose which recordings to (re-)process
    if motion_correct or pick:
        action = 'motion-correct' if motion_correct else 'analyze'
        chosen = set(_pick_recordings(all_mkvs, folder, action))
        if len(chosen) == 0:
            print('No recordings selected. Exiting.')
            return None
    else:
        chosen = set(all_mkvs)

    preanalysis_folder = None
    for group in groups:
        todo = [mkv for mkv in group['mkvs'] if mkv in chosen]
        if group['long']:
            if len(todo) == 0:
                continue
            preanalysis_folder = _process_long(group['main'], group['mkvs'], todo,
                                               motion_correct)
        else:
            for i, mkv in enumerate(todo):
                print(f'{i+1}/{len(todo)}: ', end='')
                _process_one(mkv, os.path.join(os.path.dirname(mkv), ACT_SAVE_NAME),
                             motion_correct)

    return preanalysis_folder


def _is_ignored(path: Path) -> bool:
    """
    Whether a subfolder is one of the pipeline's own outputs (not a recording).

    Parameters
    ----------
    path : Path
        A subfolder.

    Returns
    -------
    bool
        True for hidden folders, ``00_*`` folders (e.g. ``00_preanalysis``), and
        older analysis folders (holding a ``mkv_paths.json``).
    """

    return (path.name.startswith('.') or path.name.startswith('00_')
            or (path / MKV_PATHS_NAME).exists())


def _find_recordings(folder: str) -> list[dict]:
    """
    Find the .mkv files in `folder` and group them by main folder.

    Each first-level subfolder holding .mkv files directly is a recording of a
    single long recording whose main folder is `folder`. Each first-level
    subfolder without direct .mkv files but with .mkv files one level further
    down (e.g. ``Fly_001/001/*.mkv``) is a main folder of independent
    recordings. .mkv files directly in `folder` are independent recordings of
    `folder`. Any other .mkv (deeper) is ignored with a warning.

    Parameters
    ----------
    folder : str
        The selected folder.

    Returns
    -------
    list of dict
        One dict per main folder, with keys 'main' (str), 'long' (bool, single
        long recording), and 'mkvs' (sorted list of .mkv paths).
    """

    root = Path(folder)
    groups = []

    # .mkv directly in the selected folder: independent recordings
    direct = sorted(str(p) for p in root.glob('*.mkv'))
    if direct:
        groups.append({'main': folder, 'long': False, 'mkvs': direct})

    long_mkvs: list[str] = []
    subdirs = sorted(p for p in root.iterdir() if p.is_dir() and not _is_ignored(p))
    for sub in subdirs:
        sub_mkvs = sorted(str(p) for p in sub.glob('*.mkv'))
        if sub_mkvs:
            # one level down: a recording of the single long recording
            long_mkvs.extend(sub_mkvs)
            continue
        fly_mkvs = sorted(str(p) for rec in sorted(sub.iterdir())
                          if rec.is_dir() and not _is_ignored(rec)
                          for p in rec.glob('*.mkv'))
        if fly_mkvs:
            # two levels down: the subfolder (FlyNNN) is a main folder
            groups.append({'main': str(sub), 'long': False, 'mkvs': fly_mkvs})

    if long_mkvs:
        groups.insert(0, {'main': folder, 'long': True, 'mkvs': long_mkvs})

    # warn about any .mkv that doesn't fit (e.g. deeper than two levels)
    used = {mkv for group in groups for mkv in group['mkvs']}
    for p in sorted(root.rglob('*.mkv')):
        if str(p) not in used:
            print(f'Warning: ignoring {p.relative_to(root)} (deeper than 2 '
                  f'levels, or not matching the folder structure).')

    return groups


def _pick_recordings(mkv_paths: list[str],
                     folder: str,
                     action: str) -> list[str]:
    """
    List .mkv files in the terminal and let the user pick some.

    Each file is shown with a number, its path relative to `folder`, and its
    state (not processed, processed, or motion-corrected).

    Parameters
    ----------
    mkv_paths : list of str
        The .mkv files to choose from.
    folder : str
        The selected folder (paths are shown relative to it).
    action : str
        Verb shown in the prompt, e.g. 'analyze' or 'motion-correct'.

    Returns
    -------
    list of str
        The chosen .mkv paths (empty if cancelled).
    """

    print(f'\n.mkv files in {folder}:')
    print('  [0] all')
    for i, mkv in enumerate(mkv_paths):
        dir = os.path.dirname(mkv)
        if _is_corrected(dir):
            state = 'motion-corrected'
        elif _outputs_exist(dir):
            state = 'processed'
        else:
            state = 'not processed'
        print(f'  [{i + 1}] {os.path.relpath(mkv, folder)}  ({state})')

    selection = input(f"\nWhich recordings to {action}? "
                      "e.g. '1,3,5-7', '0' or 'all', or 'q' to cancel: ").strip().lower()
    if selection in ('q', 'quit', ''):
        return []
    if selection == 'all' or '0' in selection.replace(',', ' ').split():
        return list(mkv_paths)
    return [mkv_paths[i] for i in _parse_selection(selection, len(mkv_paths))]


def _process_one(mkv_file: str,
                 act_path: str | None,
                 motion_correct: bool) -> None:
    """
    Process one recording, unless it's already processed (and not correcting).

    If skipped and its activity plot is missing (e.g. it was first processed in
    a different folder layout), the plot is redrawn from the saved `act`.

    Parameters
    ----------
    mkv_file : str
        Path to the .mkv video file.
    act_path : str or None
        Save path of its activity plot, or None for no plot (recordings of a
        long recording, summarized in a grid instead).
    motion_correct : bool
        Whether to (re-)process it with motion correction.
    """

    name = os.path.basename(mkv_file)
    dir = os.path.dirname(mkv_file)

    if not motion_correct and _outputs_exist(dir):
        print(f'{name} already processed, skip.')
        if act_path is not None and not os.path.exists(act_path):
            with h5py.File(os.path.join(dir, PROJ_H5_NAME), 'r') as f:
                act = f['act'][:]  # type: ignore[index]
            _plot_activity(mkv_file, act, act_path)
        return

    verb = 'motion-correcting' if motion_correct else 'processing'
    print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - {verb} {name}')
    _process_mkv(mkv_file, act_path=act_path, motion_correct=motion_correct,
                 plot_activity=act_path is not None)


def _process_long(main: str,
                  mkv_paths: list[str],
                  todo: list[str],
                  motion_correct: bool) -> str:
    """
    Process the recordings of a single long recording and its group outputs.

    Processes the recordings in `todo` (without activity plots), then, in
    ``<main>/00_preanalysis``, saves the activity grids of every processed
    recording (see `save_activity_summary()`), writes ``mkv_paths.json``,
    aligns the recordings, and builds the grouped projections and the max
    projection and median grids. Normally the group is the recordings in
    `todo`; when motion-correcting it is the recordings already in
    ``mkv_paths.json`` (or every processed one, if there is none), so the group
    outputs are rebuilt from the corrected images.

    Parameters
    ----------
    main : str
        The main folder.
    mkv_paths : list of str
        Every .mkv of the long recording.
    todo : list of str
        The .mkv files to (re-)process.
    motion_correct : bool
        Whether to process `todo` with motion correction.

    Returns
    -------
    str
        The ``00_preanalysis`` folder.
    """

    time_0 = get_time_0(main)
    if time_0 is None:
        print(f'Could not get the puparium formation time from the name of '
              f'{main} (expected to end in MM-DD_HHhMM); hAPF will be omitted.')

    for i, mkv in enumerate(todo):
        print(f'{i+1}/{len(todo)}: ', end='')
        _process_one(mkv, None, motion_correct)

    # activity grids of every processed recording (not only `todo`)
    save_activity_summary(main, mkv_paths)

    folder = os.path.join(main, PREANALYSIS_FOLDER)
    os.makedirs(folder, exist_ok=True)
    json_path = os.path.join(folder, MKV_PATHS_NAME)

    # recordings that make up the group
    if not motion_correct:
        group = todo
    elif os.path.exists(json_path):
        with open(json_path, 'r') as f:
            group = [entry['mkv_path'] for entry in json.load(f)]
    else:
        group = mkv_paths
    missing = [mkv for mkv in group if not _outputs_exist(os.path.dirname(mkv))]
    for mkv in missing:
        print(f'Warning: {os.path.basename(mkv)} is not processed, left out of '
              f'the group outputs.')
    group = [mkv for mkv in group if mkv not in missing]
    if len(group) == 0:
        return folder

    # write mkv_paths.json with each recording's metadata
    entries = []
    for mkv in group:
        size, num_frames, fps = _read_existing_metadata(mkv)
        entries.append({'mkv_path': mkv, 'size': size,
                        'num_frames': num_frames, 'fps': fps})
    with open(json_path, 'w') as f:
        json.dump(entries, f, indent=2)

    # align the recordings to each other, then build the group outputs
    _align_group(entries, folder)
    _build_grouped(entries, folder)
    _save_grids(entries, folder, main, time_0)

    return folder


def _get_recording_time(mkv_file: str) -> datetime.datetime | None:
    """
    Get the start time of a recording from its folder or .mkv name.

    Looks, in the recording folder name and then the .mkv name, for
    ``YYYYMMDD_HHMMSS`` or ``YYYYMMDD_HHhMM``; failing that, for the older
    ``MMDD_HHMM`` at a fixed position at the end of the .mkv name (year 1900).

    Parameters
    ----------
    mkv_file : str
        Path to the .mkv video file.

    Returns
    -------
    datetime.datetime or None
        Recording time, or None if it can't be found.
    """

    for name in (os.path.basename(os.path.dirname(mkv_file)),
                 os.path.basename(mkv_file)):
        m = re.search(r'(\d{8})_(\d{6})', name)
        if m:
            return datetime.datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')
        m = re.search(r'(\d{8})_(\d{2})h(\d{2})', name)
        if m:
            return datetime.datetime.strptime(''.join(m.groups()), '%Y%m%d%H%M')
    try:
        return datetime.datetime.strptime(os.path.basename(mkv_file)[-24:-15],
                                          '%m%d_%H%M')
    except ValueError:
        return None


def load_activity_traces(main: str,
                         mkv_paths: list[str] | None = None) -> list[dict]:
    """
    Load the activity traces of every processed recording of a long recording.

    Reads each recording's saved `act` (no video decoding), in time order,
    labelled with its hours APF (see `common.preanalysis`).

    Parameters
    ----------
    main : str
        The main folder of the long recording.
    mkv_paths : list of str, optional
        The .mkv files of the long recording. If None, they are found with
        `_find_recordings()`. Unprocessed ones are left out.

    Returns
    -------
    list of dict
        One trace per recording, with keys 'name' (recording folder, relative
        to `main`), 'title' (hours APF, or the recording folder name if
        unknown), 'hours' (int or None), 'activity' and 'fs' (see
        `common.preanalysis`).
    """

    if mkv_paths is None:
        mkv_paths = [mkv for group in _find_recordings(main) if group['long']
                     for mkv in group['mkvs']]
    time_0 = get_time_0(main)

    traces = []
    for mkv in sorted(mkv_paths, key=lambda p: os.path.dirname(p)):
        dir = os.path.dirname(mkv)
        if not _outputs_exist(dir):
            continue
        with h5py.File(os.path.join(dir, PROJ_H5_NAME), 'r') as f:
            act = f['act'][:]  # type: ignore[index]
        timestamps, fps = flytrack_io.load_timestamps(mkv)
        hours = get_hours_apf(time_0, _get_recording_time(mkv))
        traces.append({'name': os.path.relpath(dir, main),
                       'title': os.path.basename(dir) if hours is None else f'{hours} hAPF',
                       'hours': hours, 'activity': act[:len(timestamps)], 'fs': fps})
    return traces


def save_activity_summary(main: str,
                          mkv_paths: list[str] | None = None) -> None:
    """
    Save the activity grids of a long recording in ``<main>/00_preanalysis``.

    The traces of every processed recording (see `load_activity_traces()`) are
    plotted with `common.preanalysis.save_activity_summary()`.

    Parameters
    ----------
    main : str
        The main folder of the long recording.
    mkv_paths : list of str, optional
        The .mkv files of the long recording. If None, they are found with
        `_find_recordings()`. Unprocessed ones are left out.

    Returns
    -------
    None
    """

    preanalysis.save_activity_summary(load_activity_traces(main, mkv_paths),
                                      os.path.join(main, PREANALYSIS_FOLDER),
                                      os.path.basename(os.path.normpath(main)),
                                      ylabel='Mean fluorescence', linewidth=0.3)


def _align_group(entries: list[dict],
                 folder: str) -> None:
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
    folder : str
        Folder holding `mkv_paths.json` (``00_preanalysis``).
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

    with open(os.path.join(folder, MKV_PATHS_NAME), 'w') as f:
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
                   folder: str) -> None:
    """
    Combine per-recording projections into grouped outputs.

    Reads every recording's `projection_images.h5`, applies that recording's
    `group_shift` (from `_align_group()`) to each of its projection images, and
    computes the pixel-wise median of all median images (`median`), the
    pixel-wise maximum of all max projections (`max_proj`), and the std image
    rebuilt from the combined cumulative mean, mean of squares, and frame counts.
    These are written to `projection_images_grouped.h5` (no preview PNGs: it is
    used by `get_segmentation_data.py` and `select_manual_roi.py`, not viewed).

    Parameters
    ----------
    entries : list of dict
        The mkv_paths.json entries, each with an 'mkv_path' and 'group_shift'.
    folder : str
        Folder the grouped outputs are written to (``00_preanalysis``).
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
    grouped_path = os.path.join(folder, GROUPED_H5_NAME)
    with h5py.File(grouped_path, 'w') as f:
        f.create_dataset('median', data=median)
        f.create_dataset('max_proj', data=max_proj)
        f.create_dataset('std', data=std)
    print(f'Saved grouped projections to {grouped_path}')


def _save_grids(entries: list[dict],
                folder: str,
                main: str,
                time_0: datetime.datetime | None) -> None:
    """
    Save grids of the recordings' max projections and medians (see
    `GRID_NAMES`), labelled with their time and hours APF.

    Horizontal images are rotated 90 degrees clockwise. Each subplot is titled
    with the recording time and hAPF if known, or else the recording folder
    name. See `common.preanalysis.save_image_grid()` for the layout and the
    shared contrast.

    Parameters
    ----------
    entries : list of dict
        The mkv_paths.json entries, one per subplot.
    folder : str
        Folder the figures are saved to (``00_preanalysis``).
    main : str
        The main folder (recording folder names are relative to it).
    time_0 : datetime.datetime or None
        Puparium formation time, see `common.preanalysis.get_time_0()`.

    Returns
    -------
    None
    """

    titles = []
    for entry in entries:
        mkv = entry['mkv_path']
        time_t = _get_recording_time(mkv)
        hours = get_hours_apf(time_0, time_t)
        title = (time_t.strftime('%m%d_%H%M') if time_t is not None
                 else os.path.relpath(os.path.dirname(mkv), main))
        titles.append(title if hours is None else f'{title} - {hours}h APF')

    for key, name in GRID_NAMES.items():
        images = []
        for entry in entries:
            h5_path = os.path.join(os.path.dirname(entry['mkv_path']), PROJ_H5_NAME)
            with h5py.File(h5_path, 'r') as f:
                img = f[key][:]  # type: ignore[index]
            if img.shape[1] > img.shape[0]:
                img = np.rot90(img, k=-1)
            images.append(img)
        preanalysis.save_image_grid(images, titles, os.path.join(folder, name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess FlyTrack .mkv recordings (see module docstring).')

    parser.add_argument('--folder', '--input_folder', dest='folder', type=str,
                        default='',
                        help='Folder to preprocess (opens a dialog if omitted)')
    parser.add_argument('-p', '--pick', action='store_true',
                        help='List the .mkv files found and ask which to analyze')
    parser.add_argument('-m', '--motion-correct', dest='motion_correct',
                        action='store_true',
                        help='Ask which .mkv files to re-process with motion '
                             'correction')

    args = parser.parse_args()

    preprocess_flytrack(folder=args.folder, pick=args.pick,
                        motion_correct=args.motion_correct)
