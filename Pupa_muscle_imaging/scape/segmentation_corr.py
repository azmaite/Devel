"""
Correlation-based seeded segmentation of 3D calcium imaging data, optionally
combining several sibling recordings of the same sample into one segmentation.

Pipeline:
  1. Reorganize each recording's raw `.dat` files into a subfolder (see
     `_setup_raw_data_folder`) and compute/cache its per-voxel std image
     (`get_std_image`); `process_sample_std_images` does this ahead of time
     for every non-HR sample recording inside an experiment folder.
  2. Pick a recording plus the sibling recordings (same `sample<N>` prefix,
     non-HR) to combine, combine their std-image accumulators
     (`combine_std_images`), threshold the combined std image to select
     active voxels, and reload the raw video of every selected recording
     restricted to those voxels (concatenated in time) to compute the
     per-voxel median reference image (`select_active_pixels`).
  3. Subtract baseline and correct for bleaching (rolling minimum filter)
  4. Compute local correlation image — each active voxel vs. its 6-connected neighbors
  5. Find seed voxels as local maxima of the correlation image
  6. Grow each seed region via correlation-based flood fill
  7. Save labeled volume, temporal traces, and summary figures

Usage:
  python segmentation_corr.py /path/to/recording [options]

  Or import and call directly:
  from segmentation_corr import segmentation_corr
  results = segmentation_corr(rec_path, ...)

  To precompute std images for a whole experiment folder ahead of time:
  from segmentation_corr import process_sample_std_images
  process_sample_std_images('/path/to/experiment_folder')

Parameters:
    rec_path : str
        Path to a SCAPE recording folder, used to find its sibling
        recordings to combine. Omit to pick one via file dialog, starting
        from `main_path`. Ignored if `--folders` is given.
    --folders : list of str, default=None
        Explicit list of recording folders to combine (all must share the
        same `sample<N>` prefix); skips `rec_path` and the sibling-selection
        prompt.
    --main-path : str, default='/mnt/labserver/data/MA/SCAPE'
        Root folder to browse from when `rec_path` is not provided.
    --std-threshold : float, default=30.0
        Std-image threshold for selecting active voxels.
    --test-threshold : flag
        If set, show the threshold-selection figure and prompt to confirm
        or edit the threshold before proceeding.
    --batch-size : int, default=100
        Number of volumes to load per batch when reading raw video.
    --rolling-window : int, default=8
        Rolling window size (in frames) for bleaching correction.
    --active-percentile : float, default=50.0
        Percentile threshold for selecting active frames based on total activity.
    --corr-seed : float, default=0.5
        Minimum local correlation for seed selection (0 to 1).
    --min-distance : float, default=15.0
        Minimum physical separation between seeds in microns.
        Converted to a per-axis voxel count using the voxel size from scapeio,
        so it correctly handles anisotropic voxels.
    --corr-grow : float, default=0.75
        Minimum correlation with seed trace to join a region.
    --min-size : int, default=20
        Discard regions smaller than this many voxels.
    --out-dir : str, default=None
        Output directory for results. If not provided, saves to a
        `{sample<N>}_combined/` folder next to the selected recordings.
"""

import os
import re
import sys
import shutil
import argparse

import h5py
import numpy as np
import scapeio
import tqdm
from scipy import ndimage
from skimage.feature import peak_local_max
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tifffile import imwrite

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import utils

# ---------------------------------------------------------------------------
# Raw video processing (std image, active-pixel selection)
# ---------------------------------------------------------------------------

def _is_valid_sample_folder(name: str) -> bool:
    """
    True if `name` looks like a non-high-resolution sample recording
    folder: starts with 'sample' and does not end with 'HR'.

    Parameters
    ----------
    name : str
        Folder name (not a full path).

    Returns
    -------
    bool
    """
    return name.startswith('sample') and not name.endswith('HR')


def _sample_prefix(folder_name: str) -> str:
    """
    Extract the `sample<N>` prefix from a recording folder name such as
    `sample5_run3` or `sample12_run1_HR`.

    Parameters
    ----------
    folder_name : str

    Returns
    -------
    str
    """
    match = re.match(r'^(sample\d+)_', folder_name)
    if not match:
        raise ValueError(f"'{folder_name}' does not match the "
                         f"'sample<N>_...' naming convention.")
    return match.group(1)


def _setup_raw_data_folder(rec_path: str) -> str:
    """
    Reorganize raw SCAPE files into a data subfolder so the main recording
    folder isn't cluttered with `.dat` spool files.

    scapeio matches a scan folder to its `{folder_name}_info.mat` and
    `{folder_name}_stim_data.bin` metadata files one level above it, so the
    data subfolder is named after `rec_path` and those two files must sit
    directly inside `rec_path` (not the subfolder).

    If `rec_path` already contains a `{folder_name}/` subfolder plus both
    metadata files directly inside `rec_path`, it is used as-is. Otherwise:
      - creates `{folder_name}/` inside `rec_path` and moves the `.dat`
        spool files, `acquisitionmetadata.ini`, `Spooled files.sifx`, and
        `ModifiedData/` into it, leaving any other files in `rec_path` untouched
      - moves `{folder_name}_info.mat` and `{folder_name}_stim_data.bin`
        from the parent folder into `rec_path`

    Parameters
    ----------
    rec_path : str
        Path to the main recording folder.

    Returns
    -------
    str
        Path to the data subfolder containing the `.dat` files.
    """
    folder_name = os.path.basename(rec_path)
    parent_dir = os.path.dirname(rec_path)
    dat_subfolder = os.path.join(rec_path, folder_name)
    info_path = os.path.join(rec_path, folder_name + '_info.mat')
    stim_path = os.path.join(rec_path, folder_name + '_stim_data.bin')

    if os.path.isdir(dat_subfolder) and os.path.isfile(info_path) and os.path.isfile(stim_path):
        print(f'Using existing data subfolder: {dat_subfolder}')
        return dat_subfolder

    dat_files = sorted(f for f in os.listdir(rec_path) if f.lower().endswith('.dat'))
    if not dat_files:
        if os.path.isdir(dat_subfolder):
            print(f'Using existing data subfolder: {dat_subfolder} '
                  f'(info/stim metadata may still be missing)')
            return dat_subfolder
        raise FileNotFoundError(
            f'No .dat files found in {rec_path} and no {folder_name}/ '
            f'subfolder exists.')

    print(f'Found {len(dat_files)} .dat files in {rec_path}. '
          f'Reorganizing into {folder_name}/')
    os.makedirs(dat_subfolder, exist_ok=True)

    to_move = dat_files + ['acquisitionmetadata.ini', 'Spooled files.sifx', 'ModifiedData']
    for fname in to_move:
        src = os.path.join(rec_path, fname)
        if os.path.exists(src):
            shutil.move(src, os.path.join(dat_subfolder, fname))
    print(f'Moved raw acquisition files to {dat_subfolder}')

    for suffix in ('_info.mat', '_stim_data.bin'):
        dst = os.path.join(rec_path, folder_name + suffix)
        if os.path.isfile(dst):
            continue
        src = os.path.join(parent_dir, folder_name + suffix)
        if os.path.isfile(src):
            shutil.move(src, dst)
            print(f'Moved {folder_name + suffix} to {rec_path}')
        else:
            print(f'Warning: {src} not found, skipping.')

    return dat_subfolder


def _get_num_volumes(rec_path: str) -> int:
    """
    Get the number of volumes in a SCAPE recording.

    Parameters
    ----------
    rec_path : str
        Path to the recording folder.

    Returns
    -------
    int
    """
    scan_name = scapeio.get_scan_name(rec_path)
    scan_metadata = scapeio.load_matlab_metadata(scan_name)
    return scan_metadata['daq']['numberOfScans']


def get_std_image(rec_path: str,
                   batch_size: int = 100,
                   h5_path: str | None = None) -> dict:
    """
    Reorganize `rec_path`'s raw SCAPE files (see `_setup_raw_data_folder`)
    and compute its per-voxel std image by streaming the video in batches
    (mean and mean-of-squares accumulators), then cache `sum_x`, `sum_x2`,
    `frame_count`, and `std_image` in `h5_path`.

    If those are already cached in `h5_path`, they are loaded directly
    instead of being recomputed.

    Parameters
    ----------
    rec_path : str
        Path to the recording folder.
    batch_size : int
        Number of volumes to load per batch.
    h5_path : str or None
        Path to the h5 file used to cache/save results. Defaults to
        `corr_output/processed_active_pixels.h5` inside `rec_path`.

    Returns
    -------
    dict with keys: sum_x, sum_x2, frame_count, std_image
    """
    h5_path = h5_path or os.path.join(rec_path, 'corr_output', 'processed_active_pixels.h5')

    if os.path.isfile(h5_path):
        with h5py.File(h5_path, 'r') as f:
            if 'std_image' in f and 'sum_x' in f and 'sum_x2' in f:
                print(f'Loaded existing std_image from {h5_path}')
                return dict(sum_x=f['sum_x'][:], sum_x2=f['sum_x2'][:],
                           frame_count=int(f.attrs['frame_count']),
                           std_image=f['std_image'][:])

    dat_folder = _setup_raw_data_folder(rec_path)
    num_volumes = _get_num_volumes(dat_folder)

    sum_x: np.ndarray | None = None
    sum_x2: np.ndarray | None = None
    frame_count = 0

    for i in tqdm.tqdm(range(0, num_volumes, batch_size),
                       desc=f'Computing std image ({os.path.basename(rec_path)})'):
        start_volume = i
        end_volume = min(i + batch_size, num_volumes)
        print(f'Loading volumes {start_volume} to {end_volume}...')
        video_batch, _ = scapeio.load_scan(dat_folder, colors=1, start_volume=start_volume,
                                           end_volume=end_volume - 1, return_axis_order=True)

        deskewed_batch = scapeio.operations.deskew_scan(dat_folder, images=video_batch)
        deskewed_batch = deskewed_batch.astype(np.float32)
        del video_batch

        if sum_x is None:
            sum_x = np.sum(deskewed_batch, axis=0)
            sum_x2 = np.sum(deskewed_batch ** 2, axis=0)
        else:
            sum_x += np.sum(deskewed_batch, axis=0)
            sum_x2 += np.sum(deskewed_batch ** 2, axis=0)
        frame_count += deskewed_batch.shape[0]
        del deskewed_batch

    mean = sum_x / frame_count
    mean_sq = sum_x2 / frame_count
    variance = np.maximum(0, mean_sq - np.square(mean))
    std_image = np.sqrt(variance)

    os.makedirs(os.path.dirname(h5_path) or '.', exist_ok=True)
    with h5py.File(h5_path, 'a') as f:
        for name, data in (('sum_x', sum_x), ('sum_x2', sum_x2), ('std_image', std_image)):
            if name in f:
                del f[name]
            f.create_dataset(name, data=data)
        f.attrs['frame_count'] = frame_count
    print(f'Saved sum_x, sum_x2, frame_count, std_image to {h5_path}')

    return dict(sum_x=sum_x, sum_x2=sum_x2, frame_count=frame_count, std_image=std_image)


def process_sample_std_images(exp_path: str | None = None,
                              main_path: str = '/mnt/labserver/data/MA/SCAPE',
                              batch_size: int = 100) -> None:
    """
    Reorganize raw files and precompute the std-image checkpoint for every
    non-HR sample recording directly inside `exp_path`.

    For each subfolder of `exp_path` whose name starts with `sample` and
    does not end in `HR` (see `_is_valid_sample_folder`), reorganizes its
    raw `.dat` files (see `_setup_raw_data_folder`) and computes/caches its
    std image via `get_std_image`. Recordings already cached are loaded,
    not recomputed. Errors for individual recordings are printed and do not
    stop the loop.

    Parameters
    ----------
    exp_path : str or None
        Path to the experiment folder containing sample recording
        subfolders. Omit to pick one via file dialog, starting from
        `main_path`.
    main_path : str
        Root folder to browse from when `exp_path` is not provided.
    batch_size : int
        Number of volumes to load per batch.
    """
    exp_path = exp_path or _pick_rec_path_via_dialog(main_path, title='Select experiment folder')

    names = sorted(n for n in os.listdir(exp_path)
                   if os.path.isdir(os.path.join(exp_path, n)) and _is_valid_sample_folder(n))
    print(f'Found {len(names)} recordings to process in {exp_path}')
    for name in names:
        rec_path = os.path.join(exp_path, name)
        print(f'--- {name} ---')
        try:
            get_std_image(rec_path, batch_size=batch_size)
        except Exception as e:
            print(f'Error processing {rec_path}: {e}')


def combine_std_images(rec_paths: list[str], batch_size: int = 100) -> np.ndarray:
    """
    Combine the std-image accumulators of multiple recordings into a single
    std image, as if all their frames belonged to one recording.

    Parameters
    ----------
    rec_paths : list of str
        Paths to recording folders. Each is processed (or loaded from
        cache) via `get_std_image`.
    batch_size : int
        Number of volumes to load per batch for any recording not yet cached.

    Returns
    -------
    std_image : ndarray, shape (Z, X, Y), float32
    """
    total_sum_x: np.ndarray | None = None
    total_sum_x2: np.ndarray | None = None
    total_frame_count = 0

    for rec_path in rec_paths:
        stats = get_std_image(rec_path, batch_size=batch_size)
        if total_sum_x is None:
            total_sum_x = stats['sum_x'].astype(np.float64)
            total_sum_x2 = stats['sum_x2'].astype(np.float64)
        else:
            total_sum_x += stats['sum_x']
            total_sum_x2 += stats['sum_x2']
        total_frame_count += stats['frame_count']

    mean = total_sum_x / total_frame_count
    mean_sq = total_sum_x2 / total_frame_count
    variance = np.maximum(0, mean_sq - np.square(mean))
    print(f'Combined std image from {len(rec_paths)} recording(s), '
          f'{total_frame_count} total frames')
    return np.sqrt(variance).astype(np.float32)


def _plot_active_threshold(std_image: np.ndarray,
                           flat_std_image: np.ndarray,
                           threshold: float,
                           aspect_ratio: float) -> tuple[plt.Figure, np.ndarray, float]:
    """
    Build the threshold-selection figure: histogram of std values, the flat
    std image, and the flat std image restricted to active pixels.

    Parameters
    ----------
    std_image : ndarray, shape (Z, X, Y)
    flat_std_image : ndarray, shape (X, Y)
        `std_image` averaged over Z.
    threshold : float
    aspect_ratio : float

    Returns
    -------
    fig : matplotlib.figure.Figure
    active_pixels : ndarray, shape (Z, X, Y), bool
    percentage_active : float
    """
    active_pixels = std_image > threshold
    num_active_pixels = int(np.sum(active_pixels))
    total_pixels = active_pixels.size
    percentage_active = 100 * num_active_pixels / total_pixels
    print(f'Percentage of active pixels: {percentage_active:.2f}% - '
          f'{num_active_pixels} out of {total_pixels}')

    flat_active = np.any(active_pixels, axis=0)
    std_vmin, std_vmax = np.nanpercentile(flat_std_image, [5, 95])

    fig, axs = plt.subplots(1, 3, figsize=(15, 3))
    axs[0].hist(flat_std_image.flatten(), bins=100)
    axs[0].set_title('Histogram of Std Values')
    axs[0].vlines(threshold, 0, axs[0].get_ylim()[1], color='r', linestyle='dashed', linewidth=1)
    axs[1].imshow(flat_std_image, aspect=aspect_ratio, vmin=std_vmin, vmax=std_vmax)
    axs[1].set_title('Std Flat Image')
    axs[2].imshow(flat_active, aspect=aspect_ratio, cmap='gray', vmin=0, vmax=1)
    axs[2].set_title(f'Active Pixels = {percentage_active:.2f}% (std > {threshold})')
    fig.tight_layout()

    return fig, active_pixels, percentage_active


def select_active_pixels(rec_paths: str | list[str],
                          std_image: np.ndarray,
                          threshold: float = 30.0,
                          test_threshold: bool = False,
                          aspect_ratio: float = 1.0,
                          batch_size: int = 100,
                          h5_path: str | None = None,
                          out_dir: str | None = None) -> dict:
    """
    Threshold `std_image` to select active voxels, then reload the raw
    video of every recording in `rec_paths` restricted to those voxels
    (concatenated in the given order) and compute the per-voxel median
    reference image. Saves `median_flat`, `std_image`, `active_pixel_mask`,
    and `active_video_flat` to `h5_path`.

    Parameters
    ----------
    rec_paths : str or list of str
        Path(s) to the recording folder(s) contributing frames, in the
        order they should be concatenated.
    std_image : ndarray, shape (Z, X, Y)
    threshold : float
        Std-image threshold used to select active voxels.
    test_threshold : bool
        If `True`, show the threshold-selection figure and prompt to
        confirm the threshold or type a new one, repeating until confirmed.
        Pressing Escape (or typing `esc`/`q`) raises `KeyboardInterrupt`,
        aborting the whole pipeline instead of continuing.
        If `False`, use `threshold` as-is and save the figure without
        displaying it.
    aspect_ratio : float
        width / height ratio used when plotting.
    batch_size : int
        Number of volumes to load per batch.
    h5_path : str or None
        Path to the h5 file used to save results. Defaults to
        `processed_active_pixels.h5` inside `out_dir`.
    out_dir : str or None
        Directory to save the threshold-selection figure and (by default)
        the h5 results to. Defaults to the first entry of `rec_paths`.

    Returns
    -------
    dict with keys: active_video_flat, active_mask, median_flat, std_image,
                    threshold
    """
    if isinstance(rec_paths, str):
        rec_paths = [rec_paths]

    out_dir = out_dir or rec_paths[0]
    h5_path = h5_path or os.path.join(out_dir, 'processed_active_pixels.h5')
    flat_std_image = np.nanmean(std_image, axis=0)

    if test_threshold:
        while True:
            fig, active_pixels, _ = _plot_active_threshold(
                std_image, flat_std_image, threshold, aspect_ratio)
            plt.show()
            try:
                response = input(f'Active-pixel threshold = {threshold}. Press Enter to '
                                 f'confirm, type a new threshold, or press Escape to '
                                 f'abort: ').strip()
            except KeyboardInterrupt:
                plt.close(fig)
                print('Aborted threshold selection.')
                raise
            if response.lower() in ('esc', 'escape', 'q', 'quit'):
                plt.close(fig)
                print('Aborted threshold selection.')
                raise KeyboardInterrupt('Threshold selection aborted by user.')
            if not response:
                break
            plt.close(fig)
            threshold = float(response)
    else:
        fig, active_pixels, _ = _plot_active_threshold(
            std_image, flat_std_image, threshold, aspect_ratio)

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, 'active_pixels_threshold.png'), dpi=150,
               bbox_inches='tight')
    plt.close(fig)
    print(f'Threshold-selection figure saved to {out_dir}')

    active_video_parts = []
    for rec_path in rec_paths:
        dat_folder = _setup_raw_data_folder(rec_path)
        num_volumes = _get_num_volumes(dat_folder)
        video_flat = np.zeros((num_volumes, int(np.sum(active_pixels))), dtype=np.float32)

        for i in tqdm.tqdm(range(0, num_volumes, batch_size),
                           desc=f'Loading active pixels ({os.path.basename(rec_path)})'):
            start_volume = i
            end_volume = min(i + batch_size, num_volumes)
            print(f'Loading volumes {start_volume} to {end_volume}...')
            video_batch, _ = scapeio.load_scan(dat_folder, colors=1, start_volume=start_volume,
                                               end_volume=end_volume - 1, return_axis_order=True)

            deskewed_batch = scapeio.operations.deskew_scan(dat_folder, images=video_batch)
            deskewed_batch = deskewed_batch.astype(np.float32)
            del video_batch

            video_flat[start_volume:end_volume, :] = deskewed_batch[:, active_pixels]
            del deskewed_batch

        active_video_parts.append(video_flat)

    active_video_flat = np.concatenate(active_video_parts, axis=0)
    median_flat = np.median(active_video_flat, axis=0)

    os.makedirs(os.path.dirname(h5_path) or '.', exist_ok=True)
    with h5py.File(h5_path, 'a') as f:
        for name, data in (('median_flat', median_flat), ('std_image', std_image),
                          ('active_pixel_mask', active_pixels),
                          ('active_video_flat', active_video_flat)):
            if name in f:
                del f[name]
            f.create_dataset(name, data=data)
        f.attrs['active_threshold'] = threshold
    print(f'Saved median_flat, std_image, active_pixel_mask, active_video_flat to {h5_path}')

    return dict(active_video_flat=active_video_flat, active_mask=active_pixels,
               median_flat=median_flat, std_image=std_image, threshold=threshold)


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

def load_data(h5_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load preprocessed data from h5 file.

    Parameters
    ----------
    h5_path : str
        Path to processed_active_pixels.h5.

    Returns
    -------
    active_video_flat : ndarray, shape (T, N_active), float32
    active_mask : ndarray, shape (Z, X, Y), bool
    std_image : ndarray, shape (Z, X, Y), float32
    """
    with h5py.File(h5_path, 'r') as f:
        active_video_flat = f['active_video_flat'][:]
        active_mask = f['active_pixel_mask'][:].astype(bool)
        std_image = f['std_image'][:]
    print(f'Loaded: {active_video_flat.shape[0]} frames, '
          f'{active_video_flat.shape[1]} active voxels, '
          f'volume shape {active_mask.shape}')
    return active_video_flat, active_mask, std_image


def preprocess(active_video_flat: np.ndarray,
               rolling_window: int = 8,
               active_frame_percentile: float = 50.0
               ) -> tuple[np.ndarray, np.ndarray]:
    """
    Background subtraction, bleaching correction, and active frame selection.

    Parameters
    ----------
    active_video_flat : ndarray, shape (T, N)
    rolling_window : int
        Window size for rolling minimum bleaching correction.
    active_frame_percentile : float
        Keep frames with total activity above this percentile.

    Returns
    -------
    signal_active : ndarray, shape (T_active, N)
        Preprocessed signal for active frames only.
    active_idx : ndarray, shape (T_active,)
        Frame indices of active frames in the original video.
    """
    signal = active_video_flat - np.median(active_video_flat, axis=0)
    rolling_min = ndimage.minimum_filter1d(signal, size=rolling_window,
                                           axis=0, mode='nearest')
    signal = signal - rolling_min
    activity = np.sum(signal, axis=1)
    thresh = np.percentile(activity, active_frame_percentile)
    active_idx = np.where(activity > thresh)[0]
    print(f'Active frames: {len(active_idx)}/{len(active_video_flat)} '
          f'({100 * len(active_idx) / len(active_video_flat):.1f}%)')
    return signal[active_idx], active_idx


def preprocess_full(active_video_flat: np.ndarray,
                    rolling_window: int = 8) -> np.ndarray:
    """
    Background subtraction and bleaching correction on all frames.

    Parameters
    ----------
    active_video_flat : ndarray, shape (T, N)
    rolling_window : int

    Returns
    -------
    ndarray, shape (T, N)
    """
    signal = active_video_flat - np.median(active_video_flat, axis=0)
    rolling_min = ndimage.minimum_filter1d(signal, size=rolling_window,
                                           axis=0, mode='nearest')
    return signal - rolling_min


def normalize_traces(traces: np.ndarray) -> np.ndarray:
    """
    Z-score normalize each voxel's time trace.

    Parameters
    ----------
    traces : ndarray, shape (T, N)

    Returns
    -------
    ndarray, shape (T, N)
    """
    mean = traces.mean(axis=0)
    std = traces.std(axis=0)
    std[std < 1e-10] = 1.0
    return (traces - mean) / std


# ---------------------------------------------------------------------------
# Local correlation image
# ---------------------------------------------------------------------------

def compute_local_correlation(traces_norm: np.ndarray,
                               active_mask: np.ndarray,
                               chunk_size: int = 10_000) -> np.ndarray:
    """
    Compute mean Pearson correlation of each active voxel with its
    6-connected active neighbors.

    Parameters
    ----------
    traces_norm : ndarray, shape (T, N_active)
        Z-score normalized traces.
    active_mask : ndarray, shape (Z, X, Y)
    chunk_size : int
        Neighbor pairs per batch (memory/speed tradeoff).

    Returns
    -------
    corr_image : ndarray, shape (Z, X, Y), float32
        Mean local correlation at each active voxel (0 for inactive).
    """
    T, N = traces_norm.shape
    Z, X, Y = active_mask.shape

    idx_vol = np.full((Z, X, Y), -1, dtype=np.int32)
    idx_vol[active_mask] = np.arange(N, dtype=np.int32)

    corr_sum = np.zeros(N, dtype=np.float64)
    n_neighbors = np.zeros(N, dtype=np.int32)

    offsets = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]

    for dz, dx, dy in tqdm.tqdm(offsets, desc='Computing local correlations'):
        shifted = np.roll(idx_vol, (-dz, -dx, -dy), axis=(0, 1, 2))
        # zero out boundary wrapping artifacts
        if dz == 1:    shifted[-1, :, :] = -1
        elif dz == -1: shifted[0,  :, :] = -1
        if dx == 1:    shifted[:, -1, :] = -1
        elif dx == -1: shifted[:, 0,  :] = -1
        if dy == 1:    shifted[:, :, -1] = -1
        elif dy == -1: shifted[:, :, 0]  = -1

        i_flat = idx_vol[active_mask]   # shape (N,)
        j_flat = shifted[active_mask]   # shape (N,), -1 where no neighbor
        valid = j_flat >= 0
        ii, jj = i_flat[valid], j_flat[valid]

        for start in range(0, len(ii), chunk_size):
            sl = slice(start, min(start + chunk_size, len(ii)))
            c = np.einsum('ti,ti->i',
                          traces_norm[:, ii[sl]],
                          traces_norm[:, jj[sl]]) / T
            np.add.at(corr_sum, ii[sl], c)

        np.add.at(n_neighbors, ii, 1)

    with np.errstate(invalid='ignore'):
        corr_avg = np.where(n_neighbors > 0,
                            corr_sum / np.maximum(n_neighbors, 1), 0.0)

    corr_image = np.zeros((Z, X, Y), dtype=np.float32)
    corr_image[active_mask] = corr_avg.astype(np.float32)
    return corr_image


# ---------------------------------------------------------------------------
# Seed detection
# ---------------------------------------------------------------------------

def find_seeds(corr_image: np.ndarray,
               active_mask: np.ndarray,
               corr_threshold: float,
               min_distance_um: float,
               voxel_size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find seed voxels as local maxima of the local correlation image.

    Seeds are sorted by descending correlation so strongest seeds
    claim territory first during region growing.

    Parameters
    ----------
    corr_image : ndarray, shape (Z, X, Y)
    active_mask : ndarray, shape (Z, X, Y)
    corr_threshold : float
        Minimum local correlation for a seed.
    min_distance_um : float
        Minimum physical separation between seeds in the same units as
        `voxel_size` (typically microns).
    voxel_size : array-like, shape (3,)
        Physical size of each voxel along (axis-0, axis-1, axis-2), in the
        same units as `min_distance_um`. Used to build an anisotropic
        ellipsoidal exclusion footprint.

    Returns
    -------
    seeds : ndarray, shape (N_seeds, 3)
        (z, x, y) of each seed, sorted by descending local correlation.
    seed_corr : ndarray, shape (N_seeds,)
        Local correlation value at each seed.
    """
    masked = np.where(active_mask, corr_image, 0.0).astype(np.float32)
    radii = np.array([min_distance_um / vs for vs in voxel_size], dtype=float)
    half = [int(np.ceil(r)) for r in radii]
    gz, gx, gy = np.ogrid[-half[0]:half[0] + 1,
                           -half[1]:half[1] + 1,
                           -half[2]:half[2] + 1]
    safe_radii = np.maximum(radii, 1e-9)
    footprint = ((gz / safe_radii[0])**2 +
                 (gx / safe_radii[1])**2 +
                 (gy / safe_radii[2])**2) <= 1.0
    seeds = peak_local_max(masked, footprint=footprint,
                           threshold_abs=corr_threshold)
    seed_corr = corr_image[seeds[:, 0], seeds[:, 1], seeds[:, 2]]
    order = np.argsort(-seed_corr)
    print(f'Found {len(seeds)} seed voxels')
    return seeds[order], seed_corr[order]


# ---------------------------------------------------------------------------
# Region growing
# ---------------------------------------------------------------------------

def grow_regions(seeds: np.ndarray,
                 traces_norm: np.ndarray,
                 active_mask: np.ndarray,
                 idx_vol: np.ndarray,
                 corr_threshold: float,
                 min_region_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Grow a region from each seed via correlation-based flood fill.

    Each voxel is assigned to the first (highest-priority) seed whose
    expanding region reaches it above `corr_threshold`. Seeds are
    processed in descending order of local correlation.

    Parameters
    ----------
    seeds : ndarray, shape (N_seeds, 3)
        (z, x, y) of each seed, sorted strongest first.
    traces_norm : ndarray, shape (T, N_active)
        Z-score normalized traces (active frames only).
    active_mask : ndarray, shape (Z, X, Y)
    idx_vol : ndarray, shape (Z, X, Y), int32
        Maps each voxel to its flat index (-1 if inactive).
    corr_threshold : float
        Minimum correlation with seed trace to join the region.
    min_region_size : int
        Regions smaller than this are discarded.

    Returns
    -------
    labels : ndarray, shape (N_active,), int32
        Region label per active voxel (0 = unassigned).
    labels_3d : ndarray, shape (Z, X, Y), int32
    """
    T, N = traces_norm.shape
    Z, X, Y = active_mask.shape
    coords = np.argwhere(active_mask)  # coords[i] = (z, x, y) for flat index i

    labels = np.zeros(N, dtype=np.int32)
    offsets = np.array([[-1, 0, 0], [1, 0, 0],
                        [0, -1, 0], [0, 1, 0],
                        [0, 0, -1], [0, 0, 1]], dtype=np.int32)
    bounds = np.array([Z, X, Y], dtype=np.int32)

    current_label = 0
    discarded = 0

    for sz, sx, sy in tqdm.tqdm(seeds, desc='Growing regions'):
        seed_flat = int(idx_vol[sz, sx, sy])
        if seed_flat < 0 or labels[seed_flat] != 0:
            continue

        current_label += 1
        seed_trace = traces_norm[:, seed_flat]  # shape (T,)

        labels[seed_flat] = current_label
        region_voxels = [seed_flat]

        visited = np.zeros(N, dtype=bool)
        visited[seed_flat] = True
        frontier = np.array([seed_flat], dtype=np.int32)

        while len(frontier) > 0:
            frontier_coords = coords[frontier]  # (F, 3)

            # Collect all unvisited, unassigned active neighbors
            all_cands = []
            for off in offsets:
                nc = frontier_coords + off          # (F, 3)
                in_bounds = np.all((nc >= 0) & (nc < bounds), axis=1)
                nc_ok = nc[in_bounds]
                if nc_ok.shape[0] == 0:
                    continue
                nb = idx_vol[nc_ok[:, 0], nc_ok[:, 1], nc_ok[:, 2]]
                active = nb >= 0
                nb = nb[active]
                if nb.shape[0] == 0:
                    continue
                eligible = (labels[nb] == 0) & ~visited[nb]
                all_cands.append(nb[eligible])

            if not all_cands:
                break

            cand_arr = np.unique(np.concatenate(all_cands))
            visited[cand_arr] = True

            # Batch correlation with seed trace
            corrs = traces_norm[:, cand_arr].T @ seed_trace / T
            accepted = cand_arr[corrs >= corr_threshold]

            if len(accepted) == 0:
                break

            labels[accepted] = current_label
            region_voxels.extend(accepted.tolist())
            frontier = accepted

        if len(region_voxels) < min_region_size:
            labels[np.array(region_voxels)] = 0
            current_label -= 1
            discarded += 1

    # Re-label sequentially to close gaps from discarded regions
    unique = np.unique(labels[labels > 0])
    new_labels = np.zeros_like(labels)
    for new_l, old_l in enumerate(unique, 1):
        new_labels[labels == old_l] = new_l

    labels_3d = np.zeros((Z, X, Y), dtype=np.int32)
    labels_3d[active_mask] = new_labels

    n_muscles = len(unique)
    n_assigned = int(np.sum(new_labels > 0))
    print(f'Segmentation: {n_muscles} muscles found '
          f'({discarded} discarded, '
          f'{n_assigned}/{N} voxels assigned = {100 * n_assigned / N:.1f}%)')
    return new_labels, labels_3d


# ---------------------------------------------------------------------------
# Temporal traces
# ---------------------------------------------------------------------------

def compute_temporal_traces(labels: np.ndarray,
                             signal_full: np.ndarray) -> np.ndarray:
    """
    Compute mean temporal trace per region over all frames.

    Parameters
    ----------
    labels : ndarray, shape (N_active,)
    signal_full : ndarray, shape (T_full, N_active)
        Full preprocessed signal (all frames, not just active ones).

    Returns
    -------
    traces : ndarray, shape (T_full, N_muscles), float32
    """
    n_muscles = int(labels.max())
    T = signal_full.shape[0]
    traces = np.zeros((T, n_muscles), dtype=np.float32)
    for m in range(1, n_muscles + 1):
        voxels = np.where(labels == m)[0]
        if len(voxels) > 0:
            traces[:, m - 1] = signal_full[:, voxels].mean(axis=1)
    return traces


# ---------------------------------------------------------------------------
# Save and visualize
# ---------------------------------------------------------------------------

def save_results(out_path: str,
                 labels: np.ndarray,
                 labels_3d: np.ndarray,
                 temporal_traces: np.ndarray,
                 corr_image: np.ndarray,
                 seeds: np.ndarray,
                 active_mask: np.ndarray,
                 params: dict) -> None:
    """
    Save segmentation results to HDF5.

    Parameters
    ----------
    out_path : str
    labels : ndarray, shape (N_active,)
    labels_3d : ndarray, shape (Z, X, Y)
    temporal_traces : ndarray, shape (T, N_muscles)
    corr_image : ndarray, shape (Z, X, Y)
    seeds : ndarray, shape (N_seeds, 3)
    active_mask : ndarray, shape (Z, X, Y)
    params : dict
        Algorithm parameters stored as HDF5 attributes.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('labels',          data=labels,          compression='gzip')
        f.create_dataset('labels_3d',       data=labels_3d,       compression='gzip')
        f.create_dataset('temporal_traces', data=temporal_traces, compression='gzip')
        f.create_dataset('corr_image',      data=corr_image,      compression='gzip')
        f.create_dataset('seeds',           data=seeds)
        f.create_dataset('active_mask',     data=active_mask,     compression='gzip')
        for k, v in params.items():
            f.attrs[k] = v
    print(f'Results saved to {out_path}')


def plot_results(corr_image: np.ndarray,
                 labels_3d: np.ndarray,
                 seeds: np.ndarray,
                 temporal_traces: np.ndarray,
                 active_mask: np.ndarray,
                 aspect_ratio: float = 1.0,
                 voxel_size: np.ndarray | None = None,
                 out_dir: str = '.') -> None:
    """
    Save summary figures and TIFFs: correlation image, segmentation map,
    temporal traces, and (when `voxel_size` is provided) OME-TIFFs with
    correct voxel spacing for ImageJ / napari.

    Parameters
    ----------
    corr_image : ndarray, shape (Z, X, Y)
    labels_3d : ndarray, shape (Z, X, Y)
    seeds : ndarray, shape (N_seeds, 3)
    temporal_traces : ndarray, shape (T, N_muscles)
    active_mask : ndarray, shape (Z, X, Y)
    aspect_ratio : float
        width / height pixel ratio used for imshow (default: 1.0).
    voxel_size : array-like of shape (3,) or None
        Physical voxel size (z_um, x_um, y_um) in micrometres.  When
        provided, saves corr_image.tif and segmentation_3d.tif with
        embedded OME voxel-size metadata.
    out_dir : str
    """
    os.makedirs(out_dir, exist_ok=True)
    n_muscles = int(labels_3d.max())
    cmap = plt.colormaps['tab20'].resampled(max(n_muscles, 1))

    # 1. Local correlation image (max-Z projection)
    #corr_proj = np.where(active_mask, corr_image, np.nan).max(axis=0)
    corr_proj = np.mean(corr_image, axis=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(corr_proj, aspect=aspect_ratio, cmap='hot')#, vmin=0, vmax=1)
    ax.scatter(seeds[:, 2], seeds[:, 1], c='cyan', s=15, marker='+',
               linewidths=0.8, label=f'{len(seeds)} seeds')
    plt.colorbar(im, ax=ax, label='Mean local correlation')
    ax.set_title('Local correlation image (max-Z projection)')
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'corr_image.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # 2. Segmentation map (max-Z projection, colored by label)
    label_proj = labels_3d.max(axis=0)
    label_rgba = cmap(label_proj / max(n_muscles, 1))
    label_rgba[label_proj == 0] = [0.1, 0.1, 0.1, 1.0]  # unassigned → dark grey
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(label_rgba, aspect=aspect_ratio)
    ax.set_title(f'Segmented muscles (max-Z projection): {n_muscles} found')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'segmentation.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # 3. Temporal traces (first 20 muscles)
    n_show = min(n_muscles, 20)
    if n_show > 0:
        fig, axes = plt.subplots(n_show, 1,
                                 figsize=(14, max(4, n_show * 1.2)),
                                 sharex=True)
        if n_show == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax.plot(temporal_traces[:, i], lw=0.7,
                    color=cmap(i / n_show))
            ax.set_ylabel(f'M{i + 1}', fontsize=7, rotation=0, labelpad=22)
            ax.tick_params(labelsize=6)
            ax.spines[['top', 'right']].set_visible(False)
        axes[-1].set_xlabel('Frame')
        fig.suptitle(f'Temporal traces (first {n_show} muscles)')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'temporal_traces.png'), dpi=150,
                    bbox_inches='tight')
        plt.close(fig)

    # 4. OME-TIFFs with voxel-size metadata (skipped when voxel_size is unknown)
    if voxel_size is not None:
        z_um, x_um, y_um = voxel_size
        # Array axes are (Z, X_physical, Y_physical); in OME ZYX notation the
        # last dim is OME-X (columns) and the second-to-last is OME-Y (rows).
        # Assigning PhysicalSizeY=x_um and PhysicalSizeX=y_um keeps the image
        # orientation consistent with the notebook plots (no transpose needed).
        tif_meta = {
            'axes': 'ZYX',
            'PhysicalSizeZ': float(z_um), 'PhysicalSizeZUnit': 'µm',
            'PhysicalSizeY': float(x_um), 'PhysicalSizeYUnit': 'µm',
            'PhysicalSizeX': float(y_um), 'PhysicalSizeXUnit': 'µm',
        }
        imwrite(os.path.join(out_dir, 'corr_image.tif'),
                corr_image.astype(np.float32),
                ome=True, metadata=tif_meta)

        # RGB segmentation volume: each label gets a distinct tab20 colour
        lut = (np.array([cmap(m / max(n_muscles, 1))[:3]
                         for m in range(n_muscles + 1)]) * 255).astype(np.uint8)
        lut[0] = [26, 26, 26]  # background → dark grey
        seg_rgb = lut[labels_3d]  # (Z, X, Y, 3)
        imwrite(os.path.join(out_dir, 'segmentation_3d.tif'),
                seg_rgb, ome=True, photometric='rgb',
                metadata={**tif_meta, 'axes': 'ZYXS'})

    print(f'Figures saved to {out_dir}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _pick_rec_path_via_dialog(main_path: str, title: str = 'Select recording folder') -> str:
    """
    Open a folder-chooser dialog rooted at `main_path` and return the
    selected path. Exits with an error message if the folder is not found.

    Parameters
    ----------
    main_path : str
        Root folder to browse from.
    title : str
        Dialog window title.
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(
        title=title,
        initialdir=main_path)
    root.destroy()

    if not folder:
        print('No folder selected. Exiting.')
        sys.exit(0)

    if not os.path.isdir(folder):
        print(f'Folder not found: {folder}')
        sys.exit(1)

    return folder


def segmentation_corr(rec_path: str | None = None,
                       main_path: str = '/mnt/labserver/data/MA/SCAPE',
                       folders: list[str] | None = None,
                       std_threshold: float = 30.0,
                       test_threshold: bool = False,
                       batch_size: int = 100,
                       rolling_window: int = 8,
                       active_percentile: float = 50.0,
                       corr_seed: float = 0.5,
                       min_distance: float = 15.0,
                       corr_grow: float = 0.75,
                       min_size: int = 20,
                       out_dir: str | None = None) -> dict:
    """
    Run the full correlation-based seeded segmentation pipeline, optionally
    combining several sibling recordings of the same sample into one
    segmentation.

    If `folders` is not given, `rec_path` is used to find its sibling
    recordings — other subfolders in the same parent directory sharing its
    `sample<N>` prefix and not ending in `HR` (see `_is_valid_sample_folder`)
    — and the user is prompted to confirm using all of them or pick a
    subset. If `folders` is given directly, all of them must share the same
    `sample<N>` prefix and the prompt is skipped.

    Each selected recording's raw `.dat` files are reorganized into a
    subfolder (see `_setup_raw_data_folder`); their std-image accumulators
    are combined (see `combine_std_images`) into one std image, which is
    thresholded to select active voxels shared across all of them. Their
    raw video is then reloaded and concatenated (in selection order)
    restricted to those voxels.

    Parameters
    ----------
    rec_path : str or None
        Path to a recording folder used to find its sibling recordings.
        Omit to pick one via file dialog, starting from `main_path`.
        Ignored if `folders` is given.
    main_path : str
        Root folder to browse from when `rec_path` is not provided.
    folders : list of str or None
        Explicit list of recording folders to combine, all sharing the same
        `sample<N>` prefix. If given, `rec_path` and the sibling-selection
        prompt are skipped.
    std_threshold : float
        Std-image threshold for selecting active voxels. Only used when
        active-pixel data has not already been processed and cached.
    test_threshold : bool
        If `True`, show the threshold-selection figure and prompt to
        confirm or edit `std_threshold` before proceeding.
    batch_size : int
        Number of volumes to load per batch when reading raw video.
    rolling_window : int
        Rolling-minimum window for bleaching correction (frames).
    active_percentile : float
        Use frames above this activity percentile for correlation computation.
    corr_seed : float
        Min local correlation to be a seed candidate (0 to 1).
    min_distance : float
        Min physical separation between seeds in microns.
    corr_grow : float
        Min correlation with seed trace to join a region.
    min_size : int
        Discard regions smaller than this many voxels.
    out_dir : str or None
        Output directory. Defaults to a `{sample<N>}_combined/` folder next
        to the selected recordings. All outputs — the combined
        std/active-pixel checkpoint (processed_active_pixels.h5), the
        threshold-selection figure, and the final segmentation results —
        are saved inside its `corr_output/` subfolder.

    Returns
    -------
    dict with keys: labels, labels_3d, temporal_traces, corr_image, seeds,
                    active_mask, voxel_size, aspect_ratio
    """
    if folders is not None:
        folders = [os.path.abspath(f) for f in folders]
        prefixes = {_sample_prefix(os.path.basename(f)) for f in folders}
        if len(prefixes) != 1:
            raise ValueError(f'All folders must share the same sample<N> '
                             f'prefix, got: {sorted(prefixes)}')
        sample_prefix = prefixes.pop()
        parent_dir = os.path.dirname(folders[0])
    else:
        rec_path = rec_path or _pick_rec_path_via_dialog(main_path)
        rec_path = os.path.abspath(rec_path)
        parent_dir = os.path.dirname(rec_path)
        sample_prefix = _sample_prefix(os.path.basename(rec_path))

        sibling_names = sorted(
            name for name in os.listdir(parent_dir)
            if os.path.isdir(os.path.join(parent_dir, name))
            and name.startswith(sample_prefix + '_')
            and _is_valid_sample_folder(name))

        print(f"Sibling recordings for '{sample_prefix}':")
        for i, name in enumerate(sibling_names, 1):
            print(f'  {i}. {name}')
        response = input('Press Enter to use all of them, or type comma-separated '
                         'numbers to select a subset: ').strip()
        if response:
            indices = [int(x.strip()) - 1 for x in response.split(',')]
            sibling_names = [sibling_names[i] for i in indices]
        folders = [os.path.join(parent_dir, name) for name in sibling_names]

    print(f"Using {len(folders)} recording(s): "
          f"{[os.path.basename(f) for f in folders]}")

    out_dir = out_dir or os.path.join(parent_dir, f'{sample_prefix}_combined')
    out_dir = os.path.join(out_dir, 'corr_output')
    h5_path = os.path.join(out_dir, 'processed_active_pixels.h5')

    axes = {'z': 0, 'x': 1, 'y': 2}
    try:
        ref_dat_folder = _setup_raw_data_folder(folders[0])
        voxel_size = np.array(scapeio.get_voxel_size(ref_dat_folder, '50 mm Navitar',
                                                     deskewed=True))
        aspect_ratio = voxel_size[axes['x']] / voxel_size[axes['y']]
        print(f'Voxel size: {voxel_size[0]:.3f} x {voxel_size[1]:.3f} x '
              f'{voxel_size[2]:.3f} µm  (aspect ratio: {aspect_ratio:.4f})')
    except Exception as e:
        print(f'Could not get voxel size from scapeio ({e}), '
              f'assuming isotropic voxels — min-distance treated as voxels')
        voxel_size = np.ones(3)
        aspect_ratio = 1.0

    # 1. Load cached combined active-pixel data, or compute it from the raw recordings
    try:
        active_video_flat, active_mask, std_image = load_data(h5_path)
    except Exception:
        print(f'No valid processed data found at {h5_path}. '
              f'Processing raw video data...')
        std_image = combine_std_images(folders, batch_size=batch_size)
        selected = select_active_pixels(folders, std_image,
                                        threshold=std_threshold,
                                        test_threshold=test_threshold,
                                        aspect_ratio=aspect_ratio,
                                        batch_size=batch_size,
                                        h5_path=h5_path, out_dir=out_dir)
        active_video_flat = selected['active_video_flat']
        active_mask = selected['active_mask']
        std_image = selected['std_image']

    # 2. Preprocess: active frames only for correlation computation
    signal_active, _ = preprocess(active_video_flat,
                                   rolling_window=rolling_window,
                                   active_frame_percentile=active_percentile)
    traces_norm = normalize_traces(signal_active)

    # 3. Local correlation image
    corr_image = compute_local_correlation(traces_norm, active_mask)

    # 4. Find seeds (sorted by descending correlation)
    seeds, seed_corr = find_seeds(corr_image, active_mask,
                                   corr_threshold=corr_seed,
                                   min_distance_um=min_distance,
                                   voxel_size=voxel_size)

    # 5. Region growing
    Z, X, Y = active_mask.shape
    N = int(active_mask.sum())
    idx_vol = np.full((Z, X, Y), -1, dtype=np.int32)
    idx_vol[active_mask] = np.arange(N, dtype=np.int32)

    labels, labels_3d = grow_regions(seeds, traces_norm, active_mask, idx_vol,
                                      corr_threshold=corr_grow,
                                      min_region_size=min_size)

    # 6. Temporal traces on full preprocessed signal (all frames)
    signal_full = preprocess_full(active_video_flat, rolling_window=rolling_window)
    temporal_traces = compute_temporal_traces(labels, signal_full)

    # 7. Save
    params = dict(std_threshold=std_threshold,
                  rolling_window=rolling_window,
                  active_percentile=active_percentile,
                  corr_seed=corr_seed,
                  min_distance_um=min_distance,
                  voxel_size=list(voxel_size),
                  corr_grow=corr_grow,
                  min_size=min_size)
    save_results(os.path.join(out_dir, 'segmentation_corr.h5'),
                 labels, labels_3d, temporal_traces, corr_image,
                 seeds, active_mask, params)

    # 8. Visualize
    plot_results(corr_image, labels_3d, seeds, temporal_traces, active_mask,
                 aspect_ratio=aspect_ratio, voxel_size=voxel_size,
                 out_dir=out_dir)

    return dict(labels=labels, labels_3d=labels_3d,
                temporal_traces=temporal_traces, corr_image=corr_image,
                seeds=seeds, active_mask=active_mask,
                voxel_size=voxel_size, aspect_ratio=aspect_ratio)


def main() -> None:
    """
    Run correlation-based seeded segmentation from the command line.
    """
    parser = argparse.ArgumentParser(
        description='Correlation-based muscle segmentation from SCAPE data')
    parser.add_argument('rec_path', nargs='?', default=None,
                        help='Path to a SCAPE recording folder, used to find '
                             'its sibling recordings to combine (omit to pick '
                             'one via file dialog; ignored if --folders is given)')
    parser.add_argument('--folders', nargs='+', default=None,
                        help='Explicit list of recording folders to combine '
                             '(all must share the same sample<N> prefix); '
                             'skips rec_path and the sibling-selection prompt')
    parser.add_argument('--main-path', type=str, default='/mnt/labserver/data/MA/SCAPE',
                        help='Root folder to browse from when rec_path is not '
                             'provided (default: /mnt/labserver/data/MA/SCAPE)')
    parser.add_argument('--std-threshold', type=float, default=30.0,
                        help='Std-image threshold for selecting active voxels '
                             '(default: 30.0)')
    parser.add_argument('--test-threshold', action='store_true',
                        help='Show the threshold-selection figure and prompt '
                             'to confirm or edit the threshold')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Number of volumes to load per batch when '
                             'reading raw video (default: 100)')
    parser.add_argument('--rolling-window', type=int, default=8,
                        help='Rolling-minimum window for bleaching correction '
                             '(frames, default: 8)')
    parser.add_argument('--active-percentile', type=float, default=50.0,
                        help='Use frames above this activity percentile for '
                             'correlation computation (default: 50)')
    parser.add_argument('--corr-seed', type=float, default=0.5,
                        help='Min local correlation to be a seed candidate '
                             '(default: 0.5)')
    parser.add_argument('--min-distance', type=float, default=15.0,
                        help='Min physical separation between seeds in microns '
                             '(default: 15.0)')
    parser.add_argument('--corr-grow', type=float, default=0.75,
                        help='Min correlation with seed trace to join a region '
                             '(default: 0.75)')
    parser.add_argument('--min-size', type=int, default=20,
                        help='Discard regions smaller than this many voxels '
                             '(default: 20)')
    parser.add_argument('--out-dir', type=str, default=None,
                        help="Output directory (default: '{sample<N>}_combined/' "
                             'folder next to the selected recordings)')
    args = parser.parse_args()

    rec_path = None
    if args.folders is None:
        rec_path = args.rec_path or _pick_rec_path_via_dialog(args.main_path)

    segmentation_corr(rec_path,
                      main_path=args.main_path,
                      folders=args.folders,
                      std_threshold=args.std_threshold,
                      test_threshold=args.test_threshold,
                      batch_size=args.batch_size,
                      rolling_window=args.rolling_window,
                      active_percentile=args.active_percentile,
                      corr_seed=args.corr_seed,
                      min_distance=args.min_distance,
                      corr_grow=args.corr_grow,
                      min_size=args.min_size,
                      out_dir=args.out_dir)



if __name__ == '__main__':
    main()
