#!/usr/bin/env python3
"""
Utility functions for NMF muscle roi detection. Includes functions for:

- masking and unmasking functions to apply masks to images and videos
    masked_image = mask_image(image, mask)
    unmasked_image = unmask_image(values, mask, bounding_box)
    masked_video = mask_video(video, mask)
    unmasked_video = unmask_video(masked_frames, mask, bounding_box)
    merged_mask = merge_masks(mask, submask)
    intersection_mask = intersect_masks(mask1, mask2)

- running NMF and selecting/processing frames for NMF
    frames_flat_dif_norm = diff_norm_frames(frames, background_img=None)
    W_temp = nmf_get_temporal(frames, H, **nmf_kwargs)
    W_temp, H_spat, W_temp_orig = fit_nmf(frames, n_components, selected_idx=None, **nmf_kwargs)
    var_explained = nmf_variance_explained(V, W, H)
    selected_idx = select_most_different_frames(frames_flat_dif_norm, n_select=200)

- plotting temporal and spatial components
    plot_temporal_components(W, timestamps=None, order=None, color=None, norm=False, fig=None)
    plot_spatial_components(H, mask, bounding_box, roi=None, order=None)

- loading data from mkv files and corresponding h5 files with rois, masks, bounding boxes, and timestamps
    mkv_list = get_mkv_list(folder_path=None)
    rois, masks, bounding_boxes, segment_names = load_segment_rois(mkv_file)
    timestamps, fps = load_timestamps(mkv_file)
    hoursAPF = get_hAPF(mkv_file)
    min_proj, max_proj = load_min_max_proj(mkv_file)
    median_frame = load_median_frame(mkv_file)
    frames_flat, median_flat, (frames_square) = load_mkv_roi(mkv_file, roi_names, masks, bounding_boxes=None)
"""

from __future__ import annotations

import os
import datetime
from pathlib import Path
import signal
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy import ndimage
from sklearn.decomposition import NMF
import h5py
import av
import tqdm


FRAMES_PER_RECORDING = 24_000  # 10 min × 40 fps × 4 packed frames per file frame


### MASKING AND UNMASKING FUNCTIONS ###

def mask_image(image: np.ndarray,
               mask: np.ndarray,
               bounding_box: np.ndarray | None = None) -> np.ndarray:
    """Extract 1D array of image values under the mask."""

    # check that the dimensions of the mask and image are compatible
    assert mask.shape == image.shape, f'Mask shape {mask.shape} does not match image shape {image.shape}'

    if bounding_box is not None:
        x, y, w, h = bounding_box
        mask = mask[y:y+h, x:x+w]
    return image[mask.astype(bool)]


def unmask_image(values: np.ndarray,
                 mask: np.ndarray,
                 bounding_box: np.ndarray | None = None) -> np.ndarray:
    """Reconstruct 2D image from masked values, zeros elsewhere
    and crop to bounding box (x, y, w, h) if provided."""

    # check that the number of values matches the number of pixels in the mask
    assert np.sum(mask) == len(values), f'Number of values {len(values)} does not match number of pixels in mask {np.sum(mask)}'

    result = np.zeros(mask.shape)
    result[mask.astype(bool)] = values

    if bounding_box is not None:
        x, y, w, h = bounding_box
        result = result[y:y+h, x:x+w]

    return result


def mask_video(video: np.ndarray,
               mask: np.ndarray,
               bounding_box: np.ndarray | None = None) -> np.ndarray:
    """Apply mask to each frame of video, returning 2D array of masked values."""
    return np.array([mask_image(frame, mask, bounding_box) for frame in video])


def unmask_video(masked_frames: np.ndarray,
                 mask: np.ndarray,
                 bounding_box: np.ndarray | None = None) -> np.ndarray:
    """Reconstruct video from masked frames, using unmask_image for each frame."""
    return np.array([unmask_image(frame, mask, bounding_box) for frame in masked_frames])

def merge_masks(mask: np.ndarray,
                submask: np.ndarray) -> np.ndarray:
    """
    Combine a submask with the original mask, returning a new mask with the shape
    of the original but with the submask selected pixels only.
    """
    # check that the submask is a 1d array 
    # with the same number of elements as positive values in the original mask
    assert submask.ndim == 1, f'Submask must be a 1D array, but has shape {submask.shape}'
    assert len(mask) == np.sum(mask), f'Number of pixels in mask {len(mask)} does not match number of pixels in submask {np.sum(submask)}'

    # merge masks
    new_mask = np.zeros_like(mask)
    new_mask[submask.astype(bool)] = 1

    return new_mask

def intersect_masks(mask1: np.ndarray,
                    mask2: np.ndarray) -> np.ndarray:
    """ Combine two masks by taking the intersection of their positive pixels."""
    # check that the masks have the same shape
    assert mask1.shape == mask2.shape, f'Mask shapes {mask1.shape} and {mask2.shape} do not match'

    new_mask = np.zeros_like(mask1)
    new_mask[(mask1.astype(bool)) & (mask2.astype(bool))] = 1
    return new_mask



### RUNNING NMF FUNCTIONS ###
# W=temporal components, H=spatial components

def diff_norm_frames(frames: np.ndarray,
                     background_img: np.ndarray | None = None,
                     rolling_win: None | int = 8) -> np.ndarray:
    """
    For each frame, get the difference from the background image (median frame or
    min_proj) across all frames, and normalize the difference from 0 to 1 (ignoring
    nans). If rolling_win is provided, use a rolling window to compute the background
    image and subtract it to remove bleaching and other slow changes. 
    All This can help to enhance the signal for NMF or other methods. 
    Frames can be flat or not.
    """

    if background_img is None:
        background_img = np.nanmin(frames, axis=0)
        print('Using min projection as background image for frame differencing')

    frames_dif = frames - background_img
    frames_dif[frames_dif < 0] = 0

    if rolling_win is not None:
        rolling_min = ndimage.minimum_filter1d(signal, size=rolling_win, axis=0, mode='nearest')
        frames_dif = frames_dif - rolling_min

    frames_dif_norm = (frames_dif - np.nanmin(frames_dif)) / (np.nanmax(frames_dif) - np.nanmin(frames_dif))

    return frames_dif_norm


def nmf_get_temporal(frames: np.ndarray,
                     H: np.ndarray) -> np.ndarray:
    """Given spatial components H, solve for temporal components W using frames."""

    # check that frames are non-negative and go from 0 to 1
    assert np.all(frames >= 0), "Frames must be non-negative"
    assert np.all(frames <= 1), "Frames must be normalized to [0, 1]"

    model = NMF(n_components=H.shape[0], init='custom')
    frames_type = type(frames[0, 0])
    model.components_ = H.astype(frames_type)
    W = model.transform(frames)

    return W


def fit_nmf(frames: np.ndarray,
            n_components: int,
            selected_idx: np.ndarray | None = None,
            **nmf_kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit NMF to the given frames, returning temporal and spatial components.
    If `selected_idx` is provided, only use those frames to learn the spatial
    components, but then solve for temporal components using all frames.
    This allows us to get spatial components from a subset of frames (e.g. those
    with high activity) while still getting temporal components for the whole video.

    Parameters
    ----------
    frames : np.ndarray
        2D array of shape (n_frames, n_pixels).
    n_components : int
        Number of NMF components.
    selected_idx : np.ndarray or None
        Indices of frames to use for fitting spatial components. If None, all
        frames are used.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        W (temporal components for all frames), 
        H (spatial components),
        W_orig (temporal components for selected frames only).
    """

    # check that frames are non-negative
    assert np.all(frames >= 0), "Frames must be non-negative"

    if selected_idx is None:
        selected_idx = np.arange(frames.shape[0])

    kwargs: dict[str, Any] = dict(n_components=n_components, max_iter=4000, init='nndsvda')
    kwargs.update(nmf_kwargs)
    model = NMF(**kwargs)
    W_orig = model.fit_transform(frames[selected_idx])
    H = model.components_

    if len(selected_idx) < frames.shape[0]:
        W = nmf_get_temporal(frames, H)
    else:
        W = W_orig

    return W, H, W_orig


def nmf_variance_explained(V: np.ndarray,
                            W: np.ndarray,
                            H: np.ndarray) -> float:
    """Calculate the variance explained (R^2) of the NMF reconstruction."""

    V_hat = W @ H
    ss_res = np.linalg.norm(V - V_hat, 'fro')**2
    ss_tot = np.linalg.norm(V - V.mean(), 'fro')**2

    return float(1 - ss_res / ss_tot)


def select_most_different_frames(frames_flat_dif_norm: np.ndarray,
                                  n_select: int = 200) -> list[int]:
    """Given a set of frames, select the `n_select` most different frames."""

    scores = frames_flat_dif_norm.mean(axis=1)
    selected_idx = [scores.argmax()]

    dists = np.linalg.norm(frames_flat_dif_norm - frames_flat_dif_norm[selected_idx[0]], axis=1)
    dists[selected_idx] = -np.inf

    print('Selecting most different frames...')
    for _ in tqdm.tqdm(range(n_select - 1)):
        new_selected_idx = np.argmax(dists)
        selected_idx.append(new_selected_idx)

        dists_new = np.linalg.norm(frames_flat_dif_norm - frames_flat_dif_norm[new_selected_idx], axis=1)
        dists = np.minimum(dists, dists_new)
        dists[selected_idx] = -np.inf

    return selected_idx



### PLOTTING FUNCTIONS ###

def _check_order(order: np.ndarray | list,
                 n_components: int) -> None:
    """
    Check that `order` is valid for `n_components` components.

    Parameters
    ----------
    order : np.ndarray or list[float]
        Ordering array. May contain NaN values for blank slots. Non-NaN values
        must be unique integers in [0, n_components).
    n_components : int
        Expected number of non-NaN entries in `order`.
    """

    assert isinstance(order, (list, np.ndarray)), 'Order must be a list or numpy array'
    assert len(order) >= n_components, 'Order must be at least as long as the number of components'
    assert np.sum(~np.isnan(order)) == n_components, 'Order must have as many non-nan elements as components'

    order = np.array(order)
    non_nan = order[~np.isnan(order)].astype(int)
    assert np.array_equal(np.sort(non_nan), np.arange(0, n_components)), 'All non-nan elements must be unique and go from 0 to n_components'


def plot_temporal_components(W: np.ndarray,
                              timestamps: np.ndarray | None = None,
                              order: np.ndarray | list | None = None,
                              color: list | np.ndarray | None = None,
                              norm: bool = False,
                              fig: Figure | None = None,
                              xlims: tuple | None = None) -> None:
    """Plot the temporal components as line plots, with options for ordering, coloring, and normalization."""

    if timestamps is None:
        timestamps = np.arange(W.shape[0])
    if order is None:
        order = np.arange(W.shape[1])
    else:
        _check_order(order, W.shape[1])

    if color is not None and len(color) == 1:
        color = color * len(order)

    blank = np.zeros(len(timestamps))
    if fig is None:
        plt.figure(figsize=(15, 5))

    for i, comp_i in enumerate(order):
        if np.isnan(comp_i):
            plt.plot(timestamps, blank - 1)
        else:
            temp = W[:, int(comp_i)]
            if norm:
                temp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
            if color is not None:
                plt.plot(timestamps, temp + i, label=f'Comp {int(comp_i)}', linewidth=1, color=color[i])
            else:
                plt.plot(timestamps, temp + i, label=f'Comp {int(comp_i)}', linewidth=1)

    plt.ylim(-0.5, len(order) + 0.5)
    plt.yticks(np.arange(len(order)), [f'Comp {int(comp_i)}' if not np.isnan(comp_i) else '' for comp_i in order])
    plt.xlabel('Time (s)')
    plt.ylabel('Component Activation')
    plt.title('Temporal Components from NMF')
    plt.legend()

    if xlims is not None:
        plt.xlim(xlims)
    else:
        plt.xlim(timestamps[0], timestamps[-1])

    if fig is None:
        plt.show()


def plot_spatial_components(H: np.ndarray,
                             mask: np.ndarray,
                             bounding_box: np.ndarray,
                             roi: np.ndarray | None = None,
                             order: np.ndarray | list | None = None,
                             cols: int = 6,
                             aspect_ratio : float = 1.0) -> None:
    """Plot the spatial components as images, with the roi contour overlaid (if present).
    If the spatial component (and mask) are 3d, they will be averaged across the z dimension before plotting."""

    if order is None:
        order = np.arange(H.shape[0])
    else:
        _check_order(order, H.shape[0])

    rows = int(np.ceil(len(order) / cols))
    _, axes = plt.subplots(rows, cols, figsize=(15, 2 * rows))
    axes = axes.flatten()

    for n, comp_i in enumerate(order):
        if np.isnan(comp_i):
            axes[n].axis('off')
            continue

        img = unmask_image(H[int(comp_i)], mask, bounding_box)

        # If the spatial component are 3D, average across z dimension
        if img.ndim == 3:
            img = img.mean(axis=0)

        axes[n].imshow(img, cmap='viridis', aspect=aspect_ratio)
        if roi is not None:
            roi_cropped = roi - bounding_box[:2]
            axes[n].plot(np.array(roi_cropped)[:, 0], np.array(roi_cropped)[:, 1], 'r--', linewidth=0.4)
        axes[n].axis('off')
        axes[n].set_title(f'Component {int(comp_i)}', fontsize=8)

    for j in range(len(order), len(axes)):
        axes[j].axis('off')
    plt.show()



### LOADING DATA FUNCTIONS ###

def get_mkv_list(folder_path: str | Path | None = None) -> list[str]:
    """Open a file dialog to select a folder, returning a list
    of all mkv files within it and its subfolders."""

    if folder_path is None:
        import tkinter as tk
        from tkinter import filedialog

        default_path = '/mnt/labserver/data/MA/Development_project/Pupa_muscle_long_recordings'

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

    default_path = '/mnt/labserver/data/MA/Development_project/Pupa_muscle_long_recordings'

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


def load_segment_rois(mkv_file: str | list[str]) -> tuple[dict, dict, dict, list[str]]:
    """
    Given an mkv file, load the segment rois, masks, and bounding boxes
    from the corresponding h5 file.
    """

    if isinstance(mkv_file, list):
        mkv_file = mkv_file[0]

    rois: dict = {}
    masks: dict = {}
    bounding_boxes: dict = {}
    h5_file = Path(mkv_file).parent.parent / 'segment_rois.h5'
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
