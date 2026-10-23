"""
Load FlyTrack recordings and their analysis files: lists of .mkv videos, the segment ROIs, masks
and bounding boxes, timestamps, projections and median frames, and ROI pixels of the videos.

sort_dialog_descending: make a tkinter file dialog list files newest first.
get_mkv_list: list the .mkv videos of a folder (or pick one with a dialog).
get_mkv_file: pick one .mkv video with a dialog.
get_hAPF: hours after puparium formation of a recording, from its name.
load_segment_rois: segment ROIs, masks, bounding boxes and names of a recording.
load_timestamps: frame timestamps and frame rate of a recording.
load_min_max_proj: min and max projections of a recording.
load_median_frame: median frame of a recording.
load_mkv_roi: flattened ROI pixels of the frames of a recording.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import av
import h5py
import numpy as np
import tqdm

from common.masks import mask_image
from common.paths import FLYTRACK_PATH

FRAMES_PER_RECORDING = 24_000  # 10 min x 40 fps x 4 packed frames per file frame


def sort_dialog_descending(tk_root: Any) -> None:
    """
    Make Tk file dialogs list folders and files in descending (Z-A) order.

    Our recording folders start with a date (``YYYYMMDD_...``), so this shows
    the newest ones first. Tk's built-in Linux dialog (used by
    `tkinter.filedialog.askdirectory()` and `askopenfilename()`) has no sort
    option, so this wraps its internal listing proc
    ``::tk::dialog::file::GlobFiltered`` to reverse the list it returns. Call it
    once on the Tk root, before opening the dialog; calling it again does nothing.

    Parameters
    ----------
    tk_root : tkinter.Tk
        The Tk root window the dialog will be opened from.

    Returns
    -------
    None
    """
    tk_root.tk.eval('''
        if {[info commands ::tk::dialog::file::GlobFilteredAscending] eq ""} {
            auto_load ::tk::dialog::file::
            rename ::tk::dialog::file::GlobFiltered ::tk::dialog::file::GlobFilteredAscending
            proc ::tk::dialog::file::GlobFiltered {args} {
                lreverse [uplevel 1 [list ::tk::dialog::file::GlobFilteredAscending {*}$args]]
            }
        }
    ''')


def get_mkv_list(folder_path: str | Path | None = None) -> list[str]:
    """Open a file dialog to select a folder, returning a list
    of all mkv files within it and its subfolders."""

    if folder_path is None:
        import tkinter as tk
        from tkinter import filedialog

        default_path = str(FLYTRACK_PATH / 'Pupa_muscle_long_recordings')

        tk_root = tk.Tk()
        tk_root.withdraw()
        folder_path = filedialog.askdirectory(initialdir=default_path, title='Select folder containing recordings')

    mkv_list = sorted(str(p) for p in Path(folder_path).rglob('*.mkv'))
    print(f'Found {len(mkv_list)} .mkv files')

    return mkv_list


def get_mkv_file() -> str:
    """Open a file dialog to select a single mkv file, returning its path."""

    import tkinter as tk
    from tkinter import filedialog

    default_path = str(FLYTRACK_PATH / 'Pupa_muscle_long_recordings')

    tk_root = tk.Tk()
    tk_root.withdraw()
    file_path = filedialog.askopenfilename(
        initialdir=default_path,
        title='Select an .mkv file',
        filetypes=[('MKV files', '*.mkv')]
    )

    return file_path


def get_hAPF(mkv_file: str) -> int:
    """
    Given an mkv file, calculate the hAPF from the pre-puparium time and recording
    times saved in the file and folder names.
    """

    time_t_str = mkv_file[-24:-15]
    time_t = datetime.datetime.strptime(time_t_str, '%m%d_%H%M')

    time_0 = datetime.datetime.strptime(mkv_file.split('/')[-3][-11:], '%m-%d_%Hh%M')

    hoursAPF = np.round((time_t - time_0).total_seconds() / 3600).astype(int)

    return hoursAPF


def _find_segment_rois_h5(path: str) -> Path:
    """
    Locate the `segment_rois.h5` written by `select_manual_roi.py`.

    `select_manual_roi` saves `segment_rois.h5` in the folder that holds the
    projection-images h5 it drew on: the recording folder (per-recording ROIs) or
    the analysis folder (ROIs drawn on the grouped projection). This resolves that
    file from either:

    - a folder path that directly contains `segment_rois.h5` (e.g. an analysis
      folder for grouped ROIs), or
    - an .mkv file path, in which case its own folder (the current, per-recording
      location) is checked first, then the folder above it (the legacy location
      used by the old `select_manual_roi`).

    Parameters
    ----------
    path : str
        A folder containing `segment_rois.h5`, or an .mkv file next to (or one
        level below) it.

    Returns
    -------
    Path
        Path to the located `segment_rois.h5`.

    Raises
    ------
    FileNotFoundError
        If no `segment_rois.h5` is found in the searched locations.
    """

    p = Path(path)
    if p.is_dir():
        candidates = [p / 'segment_rois.h5']
    else:
        # .mkv file: its own folder (current location), then one above (legacy)
        candidates = [p.parent / 'segment_rois.h5',
                      p.parent.parent / 'segment_rois.h5']

    h5_file = next((c for c in candidates if c.exists()), None)
    if h5_file is None:
        searched = ', '.join(str(c) for c in candidates)
        raise FileNotFoundError(f'No segment_rois.h5 found. Searched: {searched}')
    return h5_file


def load_segment_rois(mkv_file: str | list[str]) -> tuple[dict, dict, dict, list[str]]:
    """
    Load the segment rois, masks, and bounding boxes from `segment_rois.h5`.

    `mkv_file` may be an .mkv file path (the h5 is looked up next to it, then one
    folder above, see `_find_segment_rois_h5()`) or a folder that directly
    contains `segment_rois.h5` (e.g. an analysis folder for grouped ROIs). A list
    is accepted, in which case its first entry is used.

    For 3D (SCAPE) ROIs the returned `rois` hold the X-Y polygon (the `{name}_roi`
    alias saved alongside the 3D mask).
    """

    if isinstance(mkv_file, list):
        mkv_file = mkv_file[0]

    rois: dict = {}
    masks: dict = {}
    bounding_boxes: dict = {}
    h5_file = _find_segment_rois_h5(mkv_file)
    with h5py.File(h5_file, 'r') as f:
        segment_names : list = f['segment_names'][:]  # type: ignore[index]
        segment_names = [name.decode('utf-8') for name in segment_names]
        for segment_name in segment_names:
            rois[segment_name] = f[f'{segment_name}_roi'][:]  # type: ignore[index]
            masks[segment_name] = f[f'{segment_name}_mask'][:]  # type: ignore[index]
            bounding_boxes[segment_name] = f[f'{segment_name}_bounding_box'][:]  # type: ignore[index]

    return rois, masks, bounding_boxes, segment_names


def load_timestamps(mkv_file: str) -> tuple[np.ndarray, float]:
    """
    Given an mkv file, load the timestamps in SECONDS from the corresponding h5
    metadata file. Also returns the framerate (fps).
    """

    metadata = Path(mkv_file.replace('_raw_tiff.mkv', '_tif_metadata.h5'))
    with h5py.File(metadata, 'r') as f:
        timestamps : np.ndarray = f['timestamps'][:]  # type: ignore[index]  # in MICROSECONDS
    timestamps = timestamps.astype(np.int64)
    timestamps = timestamps - timestamps[0]
    timestamps[timestamps < 0] = timestamps[timestamps < 0] + 24 * 3600 * 1e6  # wrap midnight overflow
    timestamps = timestamps / 1e6  # convert to seconds

    fps = 1 / np.median(np.diff(timestamps))

    return timestamps, float(fps)


def load_min_max_proj(mkv_file: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Given an mkv file, load the min and max projection across all frames from the
    corresponding h5 file, generated by preprocess_pupa_muscle script.
    """

    h5_path = Path(mkv_file).parent / 'max_min_proj.h5'
    with h5py.File(h5_path, 'r') as f:
        min_proj : np.ndarray = f['min_proj'][:]  # type: ignore[index]
        max_proj : np.ndarray = f['max_proj'][:]  # type: ignore[index]

    return min_proj, max_proj


def load_median_frame(mkv_file: str) -> np.ndarray:
    """
    Given an mkv file, load the median frame from the corresponding h5 file,
    generated by preprocess_pupa_muscle script.
    """

    h5_path = Path(mkv_file).parent / 'max_min_proj.h5'
    with h5py.File(h5_path, 'r') as f:
        median_frame : np.ndarray = f['median'][:]  # type: ignore[index]

    return median_frame


def load_mkv_roi(mkv_file: str,
                 roi_names: list[str],
                 masks: dict,
                 bounding_boxes: dict | None = None) -> tuple:
    """
    Given an mkv file, load:

    - the video frames for each roi in `roi_names`, applying the corresponding mask
      to each frame so that only pixels within the mask are included and flattened
    - the median frame (from h5 file), flattened to match the masked frames, to use
      for NMF frame selection and normalization

    If `bounding_boxes` is provided, also returns the frames cropped to the bounding
    box but without removing pixels outside the mask. Bounding box format: (x, y, w, h).
    """

    for roi_name in roi_names:
        assert roi_name in masks, f'ROI for {roi_name} not found in masks'

    median_frame = load_median_frame(mkv_file)

    all_frames_flat: dict = {}
    median_flat: dict = {}
    all_frames_square: dict = {}

    if bounding_boxes is not None:
        for roi_name in roi_names:
            assert roi_name in bounding_boxes, f'Bounding box for {roi_name} not found in bounding_boxes'

    container = av.open(mkv_file)

    print('Loading video frames ...')
    with tqdm.tqdm(total=FRAMES_PER_RECORDING, unit='frame') as pbar:
        i = 0
        for frame in container.decode(video=0):
            arr = frame.to_ndarray(format='rgba64le')
            for j in range(arr.shape[2]):
                img = arr[..., j]

                for roi_name in roi_names:
                    mask = masks[roi_name]
                    pixels = np.sum(mask)

                    if roi_name not in all_frames_flat:
                        all_frames_flat[roi_name] = np.zeros((FRAMES_PER_RECORDING, pixels), dtype=np.float32)
                        if bounding_boxes is not None:
                            all_frames_square[roi_name] = np.zeros(
                                (FRAMES_PER_RECORDING, bounding_boxes[roi_name][3], bounding_boxes[roi_name][2]),
                                dtype=np.float32
                            )
                        median_flat[roi_name] = mask_image(median_frame, mask)

                    all_frames_flat[roi_name][i] = mask_image(img, mask)

                    if bounding_boxes is not None:
                        x, y, w, h = bounding_boxes[roi_name]
                        all_frames_square[roi_name][i] = img[y:y+h, x:x+w]

                i += 1
                pbar.update(1)

    container.close()

    if bounding_boxes is not None:
        return all_frames_flat, median_flat, all_frames_square
    else:
        return all_frames_flat, median_flat

