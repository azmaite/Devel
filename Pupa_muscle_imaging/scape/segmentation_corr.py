#!/usr/bin/env python3
"""
Correlation-based seeded segmentation of 3D calcium imaging data.

Pipeline:
  1. Load preprocessed data from processed_active_pixels.h5
  2. Subtract baseline and correct for bleaching (rolling minimum filter)
  3. Compute local correlation image — each active voxel vs. its 6-connected neighbors
  4. Find seed voxels as local maxima of the correlation image
  5. Grow each seed region via correlation-based flood fill
  6. Save labeled volume, temporal traces, and summary figures

Usage:
  python segmentation_corr.py /path/to/processed_active_pixels.h5 [options]

Parameters:
    h5_path : str
        Path to processed_active_pixels.h5 file containing selected active pixels and mask.
        Omit to pick a folder via file dialog.
    --rolling_window : int, default=8
        Rolling window size (in frames) for bleaching correction.
    --active_percentile : float, default=50.0
        Percentile threshold for selecting active frames based on total activity.
    --corr-seed : float, default=0.3
        Minimum local correlation for seed selection (0 to 1).
    --min_distance : float, default=5.0
        Minimum physical separation between seeds in microns.
        Converted to a per-axis voxel count using the voxel size from scapeio,
        so it correctly handles anisotropic voxels.
    --out-dir : str, default=None
        Output directory for results. If not provided, saves to same folder as h5_path.
"""

import os
import sys
import argparse

import h5py
import numpy as np
import tqdm
from scipy import ndimage
from skimage.feature import peak_local_max
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tifffile import imwrite

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils


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

def _pick_h5_via_dialog() -> str:
    """
    Open a folder-chooser dialog and return the path to
    processed_active_pixels.h5 inside the selected folder.
    Exits with an error message if the file is not found.
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(
        title='Select recording folder containing processed_active_pixels.h5',
        initialdir='/mnt/labserver/data/MA/')
    root.destroy()

    if not folder:
        print('No folder selected. Exiting.')
        sys.exit(0)

    h5_path = os.path.join(folder, 'processed_active_pixels_noResReduction.h5')
    if not os.path.isfile(h5_path):
        print(f'processed_active_pixels_noResReduction.h5 not found in {folder}')
        sys.exit(1)

    return h5_path


def main() -> None:
    """
    Run correlation-based seeded segmentation from the command line.
    """
    parser = argparse.ArgumentParser(
        description='Correlation-based muscle segmentation from SCAPE data')
    parser.add_argument('h5_path', nargs='?', default=None,
                        help='Path to processed_active_pixels.h5 '
                             '(omit to pick a folder via file dialog)')
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
                             '(default: 10)')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Output directory (default: same folder as h5_path)')
    args = parser.parse_args()

    h5_path = args.h5_path or _pick_h5_via_dialog()
    out_dir = os.path.join(
        args.out_dir or os.path.dirname(os.path.abspath(h5_path)),
        'corr_output')

    try:
        import scapeio
        rec_path = os.path.dirname(os.path.abspath(h5_path))
        voxel_size = np.array(scapeio.get_voxel_size(rec_path, '50 mm Navitar',
                                                     deskewed=True))
        aspect_ratio = voxel_size[1] / voxel_size[2]
        print(f'Voxel size: {voxel_size[0]:.3f} x {voxel_size[1]:.3f} x '
              f'{voxel_size[2]:.3f} µm  (aspect ratio: {aspect_ratio:.4f})')
    except Exception as e:
        print(f'Could not get voxel size from scapeio ({e}), '
              f'assuming isotropic voxels — min-distance treated as voxels')
        voxel_size = np.ones(3)
        aspect_ratio = 1.0

    # 1. Load
    active_video_flat, active_mask, std_image = load_data(h5_path)

    # 2. Preprocess: active frames only for correlation computation
    signal_active, _ = preprocess(active_video_flat,
                                   rolling_window=args.rolling_window,
                                   active_frame_percentile=args.active_percentile)
    traces_norm = normalize_traces(signal_active)

    # 3. Local correlation image
    corr_image = compute_local_correlation(traces_norm, active_mask)

    # 4. Find seeds (sorted by descending correlation)
    seeds, seed_corr = find_seeds(corr_image, active_mask,
                                   corr_threshold=args.corr_seed,
                                   min_distance_um=args.min_distance,
                                   voxel_size=voxel_size)

    # 5. Region growing
    Z, X, Y = active_mask.shape
    N = int(active_mask.sum())
    idx_vol = np.full((Z, X, Y), -1, dtype=np.int32)
    idx_vol[active_mask] = np.arange(N, dtype=np.int32)

    labels, labels_3d = grow_regions(seeds, traces_norm, active_mask, idx_vol,
                                      corr_threshold=args.corr_grow,
                                      min_region_size=args.min_size)

    # 6. Temporal traces on full preprocessed signal (all frames)
    signal_full = preprocess_full(active_video_flat,
                                   rolling_window=args.rolling_window)
    temporal_traces = compute_temporal_traces(labels, signal_full)

    # 7. Save
    params = dict(rolling_window=args.rolling_window,
                  active_percentile=args.active_percentile,
                  corr_seed=args.corr_seed,
                  min_distance_um=args.min_distance,
                  voxel_size=list(voxel_size),
                  corr_grow=args.corr_grow,
                  min_size=args.min_size)
    save_results(os.path.join(out_dir, 'segmentation_corr.h5'),
                 labels, labels_3d, temporal_traces, corr_image,
                 seeds, active_mask, params)

    # 8. Visualize
    plot_results(corr_image, labels_3d, seeds, temporal_traces, active_mask,
                 aspect_ratio=aspect_ratio, voxel_size=voxel_size,
                 out_dir=out_dir)


if __name__ == '__main__':
    main()
