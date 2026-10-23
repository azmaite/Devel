#!/usr/bin/env python3
"""
Build ROI-masked, activity-subsampled flattened video data for segmentation
(e.g. NMF) from a FlyTrack analysis folder.

`get_segmentation_data(analysis_folder)` is the entry point. For every segment
in `segment_rois.h5` (or, if that file is missing, a single 'all' segment
covering the whole frame) it:

1. Decodes every recording's raw .mkv listed in `mkv_paths.json`, applying each
   recording's `group_shift` (see `preprocess_flytrack._align_group`) to map
   the ROI mask back into that recording's own coordinate frame, and
   concatenates the masked pixels of every frame across all recordings into a
   flat ``(total_frames, n_roi_pixels)`` array.
2. Otsu-thresholds ``max_proj - median`` (grouped) within the ROI to keep only
   the 'active' pixels, and saves a diagnostic PNG of the threshold per segment.
3. Subtracts the grouped median from the active-pixel frames and normalizes
   the result to 0-1 (`nmf.diff_norm_frames`), then selects the `n_select`
   most different frames (`nmf.select_most_different_frames`).

Results are saved to `segmentation_data.h5` in `analysis_folder` and also
returned as a dict.
"""

from __future__ import annotations

import os
import sys
import json

import numpy as np
import h5py
import matplotlib.pyplot as plt
import av
import tqdm
from skimage.filters import threshold_otsu

# put this file's folder on sys.path so the other scripts of this folder can be imported no
# matter how this module is launched (as a script, via `-m`, or imported from a notebook) or what
# the working directory is
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import flytrack_io
from common import masks, nmf
from common.paths import FLYTRACK_PATH
from flytrack_core import MKV_PATHS_NAME, GROUPED_H5_NAME
from preprocess_flytrack import _shift_int


MAIN_PATH = str(FLYTRACK_PATH)

SEGMENT_ROIS_NAME = 'segment_rois.h5'
SEGMENTATION_DATA_NAME = 'segmentation_data.h5'
ACTIVE_PIXELS_PLOT_SUFFIX = '_active_pixels.png'

# Otsu's threshold is a bit conservative for dim/sparse muscle signal, so it is
# lowered to include this fraction more active pixels (see `_active_threshold`)
ACTIVE_PIXEL_EXTRA_FRACTION = 0.1


def get_segmentation_data(analysis_folder: str = '',
                          n_select: int = 200) -> dict:
    """
    Load, mask, and activity-subsample video data for segmentation.

    See the module docstring for the full processing steps.

    Parameters
    ----------
    analysis_folder : str, optional
        Path to a FlyTrack analysis folder (see `preprocess_flytrack`),
        containing `projection_images_grouped.h5` and `mkv_paths.json`, and
        (recommended) `segment_rois.h5` from `select_manual_roi`. If empty, a
        directory-chooser dialog is opened.
    n_select : int, optional
        Number of most-different frames to select per segment.

    Returns
    -------
    dict
        Keyed by segment name (or 'all' if no `segment_rois.h5` was found).
        Each entry holds:

        - 'mask' : full-frame uint8 array, the segment's ROI mask.
        - 'bounding_box' : [x, y, w, h] of `mask`.
        - 'active_mask' : 1D bool array over the ROI's flat pixels, True where
          `(max_proj - median) > threshold`.
        - 'threshold' : float, the threshold used (Otsu widened to include
          `ACTIVE_PIXEL_EXTRA_FRACTION` more pixels; see `_active_threshold`).
        - 'otsu_threshold' : float, the raw (unwidened) Otsu threshold.
        - 'frames_active' : (total_frames, n_active_pixels) uint16 array of raw
          values at the active pixels, concatenated across all recordings.
        - 'frames_dif_norm' : same shape, difference from the grouped median,
          normalized to 0-1.
        - 'selected_idx' : (n_select,) int array, indices (into `frames_active`
          / `frames_dif_norm`) of the most different frames.

    Raises
    ------
    FileNotFoundError
        If `analysis_folder` is missing `projection_images_grouped.h5` or
        `mkv_paths.json`.
    """

    if not analysis_folder:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        analysis_folder = filedialog.askdirectory(
            initialdir=MAIN_PATH, title='Select analysis folder')
        root.destroy()

    assert analysis_folder, 'No analysis folder selected.'

    grouped_path = os.path.join(analysis_folder, GROUPED_H5_NAME)
    mkv_paths_path = os.path.join(analysis_folder, MKV_PATHS_NAME)
    if not (os.path.exists(grouped_path) and os.path.exists(mkv_paths_path)):
        raise FileNotFoundError(
            f'{analysis_folder} does not contain {GROUPED_H5_NAME} and/or '
            f'{MKV_PATHS_NAME}. Run preprocess_flytrack.py on this '
            f'folder first.')

    with open(mkv_paths_path, 'r') as f:
        entries = json.load(f)
    assert len(entries) > 0, f'{MKV_PATHS_NAME} in {analysis_folder} has no entries'

    with h5py.File(grouped_path, 'r') as f:
        median_img = f['median'][:]      # type: ignore[index]
        max_proj_img = f['max_proj'][:]  # type: ignore[index]

    # difference image the active pixels are picked from: brighter where a
    # pixel's peak fluorescence rises furthest above its resting (median) level
    activity_img = max_proj_img.astype(np.float64) - median_img.astype(np.float64)

    masks, bounding_boxes, segment_names = _load_or_default_rois(
        analysis_folder, median_img.shape)

    n_frames_per_recording = [int(e['num_frames']) for e in entries]
    all_frames_flat = _load_raw_roi_frames(entries, masks, segment_names,
                                           n_frames_per_recording)

    result: dict = {}
    for name in segment_names:
        result[name] = _process_segment(analysis_folder, name, masks[name],
                                        bounding_boxes[name], activity_img,
                                        median_img, all_frames_flat[name],
                                        n_select)

    _save_segmentation_data(analysis_folder, result, n_frames_per_recording)
    return result


def _load_or_default_rois(analysis_folder: str,
                          image_shape: tuple) -> tuple[dict, dict, list[str]]:
    """
    Load segment ROIs, or fall back to a single 'all' whole-frame segment.

    Parameters
    ----------
    analysis_folder : str
        Folder to look for `segment_rois.h5` in.
    image_shape : tuple
        Shape of the grouped median/std image, used to build the whole-frame
        fallback mask.

    Returns
    -------
    tuple of (dict, dict, list of str)
        (masks, bounding_boxes, segment_names), keyed/ordered as in
        `flytrack_io.load_segment_rois`.
    """

    rois_path = os.path.join(analysis_folder, SEGMENT_ROIS_NAME)
    if os.path.exists(rois_path):
        _, masks, bounding_boxes, segment_names = flytrack_io.load_segment_rois(analysis_folder)
        return masks, bounding_boxes, segment_names

    print(f'No {SEGMENT_ROIS_NAME} found in {analysis_folder}. Either run '
         f'select_manual_roi(analysis_folder) to draw ROIs first (recommended), '
         f'or continue now with every pixel in the image as a single "all" '
         f'segment -- WARNING: this may run out of memory / crash, especially '
         f'with several recordings grouped together.')
    height, width = image_shape[:2]
    segment_names = ['all']
    masks = {'all': np.ones((height, width), dtype=np.uint8)}
    bounding_boxes = {'all': [0, 0, width, height]}
    return masks, bounding_boxes, segment_names


def _load_raw_roi_frames(entries: list[dict],
                         masks: dict,
                         segment_names: list[str],
                         n_frames_per_recording: list[int]) -> dict:
    """
    Decode every recording once and extract each segment's masked pixels.

    Each recording's segment masks are shifted by ``-group_shift`` into that
    recording's own (unaligned) coordinate frame before masking its raw frames;
    this is equivalent to (and much cheaper than) shifting every decoded frame
    into the aligned frame and masking with the original mask.

    Parameters
    ----------
    entries : list of dict
        `mkv_paths.json` entries, each with 'mkv_path', 'num_frames', and
        'group_shift'.
    masks : dict
        Segment name -> full-frame uint8 mask, in the aligned (grouped)
        coordinate frame.
    segment_names : list of str
        Segments to extract.
    n_frames_per_recording : list of int
        Expected real-frame count of each entry, in the same order.

    Returns
    -------
    dict
        Segment name -> (total_frames, n_roi_pixels) uint16 array of masked
        pixel values, concatenated across all recordings.
    """

    total_frames = int(sum(n_frames_per_recording))
    n_pixels = {name: int(np.sum(masks[name])) for name in segment_names}
    all_frames_flat = {name: np.zeros((total_frames, n_pixels[name]), dtype=np.uint16)
                       for name in segment_names}

    print(f'Loading raw video from {len(entries)} recording(s) for '
         f'{len(segment_names)} ROI(s)...')
    frame_offset = 0
    for entry, n_frames in zip(entries, n_frames_per_recording):
        mkv_file = entry['mkv_path']
        shift = entry.get('group_shift', [0, 0])
        inv_shift = [-shift[0], -shift[1]]
        rec_masks = {name: _shift_int(masks[name], inv_shift).astype(bool)
                    for name in segment_names}

        container = av.open(mkv_file)
        i = 0
        with tqdm.tqdm(total=n_frames, unit='frame',
                       desc=os.path.basename(mkv_file)) as pbar:
            for frame in container.decode(video=0):
                arr = frame.to_ndarray(format='rgba64le')
                for j in range(arr.shape[2]):
                    if i >= n_frames:
                        break
                    img = arr[..., j]
                    for name in segment_names:
                        all_frames_flat[name][frame_offset + i] = \
                            masks.mask_image(img, rec_masks[name])
                    i += 1
                    pbar.update(1)
        container.close()

        if i != n_frames:
            print(f'Warning: expected {n_frames} frames from '
                 f'{os.path.basename(mkv_file)} but decoded {i}.')
        frame_offset += n_frames

    return all_frames_flat


def _process_segment(analysis_folder: str,
                     name: str,
                     mask: np.ndarray,
                     bounding_box: list,
                     activity_img: np.ndarray,
                     median_img: np.ndarray,
                     frames_flat: np.ndarray,
                     n_select: int) -> dict:
    """
    Threshold, subsample, and select the most different frames for one segment.

    Parameters
    ----------
    analysis_folder : str
        Folder the diagnostic plot is saved into.
    name : str
        Segment name.
    mask : np.ndarray
        Full-frame uint8 ROI mask.
    bounding_box : list
        [x, y, w, h] of `mask`.
    activity_img : np.ndarray
        Grouped ``max_proj - median`` image, used to pick active pixels (same
        coordinate frame as `mask`).
    median_img : np.ndarray
        Grouped median image (same coordinate frame as `mask`).
    frames_flat : np.ndarray
        (total_frames, n_roi_pixels) raw masked pixel values for this segment.
    n_select : int
        Number of most-different frames to select.

    Returns
    -------
    dict
        See `get_segmentation_data`'s Returns section for the per-segment keys.
    """

    activity_flat = masks.mask_image(activity_img, mask)
    median_flat = masks.mask_image(median_img, mask)

    otsu_threshold, threshold = _active_threshold(activity_flat)
    active_mask = activity_flat > threshold
    print(f'{name}: {int(np.sum(active_mask))}/{len(activity_flat)} ROI pixels '
         f'active (otsu={otsu_threshold:.2f}, widened threshold={threshold:.2f})')
    _plot_active_pixels(analysis_folder, name, activity_img, mask, bounding_box,
                        activity_flat, active_mask, threshold, otsu_threshold)

    frames_active = frames_flat[:, active_mask]
    median_active = median_flat[active_mask]

    frames_dif_norm = nmf.diff_norm_frames(frames_active,
                                             background_img=median_active,
                                             rolling_win=None)

    print(f'Selecting {n_select} most different frames for {name}...')
    selected_idx = np.array(
        nmf.select_most_different_frames(frames_dif_norm, n_select=n_select))

    return {
        'mask': mask,
        'bounding_box': bounding_box,
        'active_mask': active_mask,
        'threshold': threshold,
        'otsu_threshold': otsu_threshold,
        'frames_active': frames_active,
        'frames_dif_norm': frames_dif_norm.astype(np.float32),
        'selected_idx': selected_idx,
    }


def _active_threshold(values: np.ndarray,
                      extra_fraction: float = ACTIVE_PIXEL_EXTRA_FRACTION) -> tuple[float, float]:
    """
    Return an Otsu threshold widened to include `extra_fraction` more pixels.

    Otsu's threshold is often a bit conservative for dim/sparse muscle signal;
    this lowers the raw threshold to the value that keeps
    ``round(n_otsu * (1 + extra_fraction))`` pixels above it instead of
    `n_otsu` (the count Otsu's own threshold keeps).

    Parameters
    ----------
    values : np.ndarray
        1D array of pixel values to threshold.
    extra_fraction : float, optional
        Fraction more pixels to include relative to the raw Otsu cut.

    Returns
    -------
    tuple of (float, float)
        (otsu_threshold, widened_threshold).
    """

    otsu_threshold = float(threshold_otsu(values))
    n_otsu = int(np.sum(values > otsu_threshold))
    target_n = min(int(round(n_otsu * (1 + extra_fraction))), len(values))
    if target_n <= 0:
        return otsu_threshold, otsu_threshold

    percentile = 100 * (1 - target_n / len(values))
    widened_threshold = float(np.percentile(values, percentile))
    return otsu_threshold, widened_threshold


def _plot_active_pixels(save_folder: str,
                        name: str,
                        activity_img: np.ndarray,
                        mask: np.ndarray,
                        bounding_box: list,
                        activity_flat: np.ndarray,
                        active_mask: np.ndarray,
                        threshold: float,
                        otsu_threshold: float | None = None) -> None:
    """
    Save a diagnostic PNG of the thresholded active pixels for one segment.

    Shows the ``max_proj - median`` image cropped to the segment's bounding
    box, the same crop with active pixels tinted red, and a histogram of
    in-ROI values with the threshold used marked (and the raw Otsu threshold,
    if given, marked for reference), so the automatic threshold can be
    checked visually.

    Parameters
    ----------
    save_folder : str
        Folder the PNG is saved into.
    name : str
        Segment name (used in the filename and titles).
    activity_img : np.ndarray
        Full-frame grouped ``max_proj - median`` image.
    mask : np.ndarray
        Full-frame uint8 ROI mask.
    bounding_box : list
        [x, y, w, h] of `mask`, used to crop the display.
    activity_flat : np.ndarray
        1D array of `activity_img` values within `mask` (for the histogram).
    active_mask : np.ndarray
        1D bool array over the ROI's flat pixels (True = active).
    threshold : float
        The threshold used to build `active_mask`, marked on the histogram.
    otsu_threshold : float or None, optional
        The unwidened Otsu threshold, marked for reference if given.

    Returns
    -------
    None
    """

    x, y, w, h = bounding_box
    activity_crop = activity_img[y:y+h, x:x+w]
    active_img = masks.unmask_image(active_mask.astype(np.uint8), mask,
                                    bounding_box).astype(bool)

    norm = ((activity_crop - activity_crop.min())
           / (activity_crop.max() - activity_crop.min() + 1e-9))
    overlay = np.stack([norm, norm, norm], axis=-1)
    overlay[active_img, 0] = 1
    overlay[active_img, 1] *= 0.3
    overlay[active_img, 2] *= 0.3

    _, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(activity_crop, cmap='gray')
    axes[0].set_title(f'{name}: max_proj - median')
    axes[0].axis('off')

    axes[1].imshow(overlay)
    axes[1].set_title(f'{name}: active pixels (red)')
    axes[1].axis('off')

    axes[2].hist(activity_flat, bins=100)
    axes[2].axvline(threshold, color='r', linestyle='--', label=f'used = {threshold:.1f}')
    if otsu_threshold is not None:
        axes[2].axvline(otsu_threshold, color='gray', linestyle=':',
                        label=f'otsu = {otsu_threshold:.1f}')
    axes[2].legend()
    axes[2].set_title(f'threshold = {threshold:.1f}')
    axes[2].set_xlabel('max_proj - median')

    plt.tight_layout()
    save_path = os.path.join(save_folder, f'{name}{ACTIVE_PIXELS_PLOT_SUFFIX}')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Saved active-pixel diagnostic plot to {save_path}')


def _save_segmentation_data(analysis_folder: str,
                            result: dict,
                            n_frames_per_recording: list[int]) -> str:
    """
    Save `get_segmentation_data`'s per-segment results to `segmentation_data.h5`.

    Parameters
    ----------
    analysis_folder : str
        Folder to save into.
    result : dict
        Per-segment results, as returned by `get_segmentation_data`.
    n_frames_per_recording : list of int
        Real-frame count of each recording, in `mkv_paths.json` order.

    Returns
    -------
    str
        Path to the written `segmentation_data.h5`.
    """

    h5_path = os.path.join(analysis_folder, SEGMENTATION_DATA_NAME)
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('segment_names',
                         data=np.array(list(result.keys()), dtype='S'))
        f.create_dataset('n_frames_per_recording',
                         data=np.array(n_frames_per_recording))
        for name, data in result.items():
            f.create_dataset(f'{name}_mask', data=data['mask'])
            f.create_dataset(f'{name}_bounding_box',
                             data=np.asarray(data['bounding_box']))
            f.create_dataset(f'{name}_active_mask', data=data['active_mask'])
            f.create_dataset(f'{name}_threshold', data=data['threshold'])
            f.create_dataset(f'{name}_otsu_threshold', data=data['otsu_threshold'])
            f.create_dataset(f'{name}_frames_active', data=data['frames_active'])
            f.create_dataset(f'{name}_frames_dif_norm', data=data['frames_dif_norm'])
            f.create_dataset(f'{name}_selected_idx', data=data['selected_idx'])

    print(f'Saved segmentation data to {h5_path}')
    return h5_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Build ROI-masked, activity-subsampled segmentation data '
                    'for a FlyTrack analysis folder.')
    parser.add_argument('--analysis_folder', type=str, default='',
                        help='Path to the analysis folder (opens a dialog if omitted)')
    parser.add_argument('--n_select', type=int, default=200,
                        help='Number of most-different frames to select per segment')
    args = parser.parse_args()

    get_segmentation_data(args.analysis_folder, args.n_select)
