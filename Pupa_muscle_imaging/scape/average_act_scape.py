#!/usr/bin/env python3
"""
Compute per-volume mean and max fluorescence activity from raw SCAPE data.

Pipeline:
  1. Optionally reorganize .dat files into a RawData subfolder.
  2. Load the recording in batches of 100 volumes.
  3. Deskew each batch and convert to float32.
  4. Compute per-volume mean and max fluorescence.
  5. Plot mean and max activity vs volume number and save PNGs to Figures/.

Usage:
  python average_act_scape.py /path/to/recording/folder [options]

  Or import and call directly:
  from average_act_scape import average_act_scape
  average_act_scape('/path/to/recording/folder')

Parameters:
    folder_path : str
        Path to the main recording folder. If it contains .dat files
        directly, they are moved into a RawData subfolder automatically.
    --batch-size : int, default=100
        Number of volumes to load and process at a time.

Notes:
    The data subfolder is named after the main folder (not RawData) so that
    scapeio can resolve the matching `{folder_name}_info.mat` file, which must
    sit directly inside the main folder (one level above the .dat files).
"""

import os
import sys
import shutil
import argparse
import time

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import scapeio


def _setup_raw_data_folder(folder_path: str) -> str:
    """
    Reorganize raw SCAPE files so scapeio can find them.

    The data subfolder is named after the main folder so that scapeio can
    match it to the accompanying `{folder_name}_info.mat` file, which must
    sit one level above the data folder (i.e., inside the main folder).

    When the main folder contains raw data files directly, this function:
      - Creates a `{folder_name}` subfolder and moves into it:
          * all .dat spool files
          * acquisitionmetadata.ini
          * Spooled files.sifx
      - Moves from the parent directory into the main folder:
          * {folder_name}_info.mat
          * {folder_name}_stim_data.bin

    If no .dat files are present, verifies that the `{folder_name}`
    subfolder already exists.

    Parameters
    ----------
    folder_path : str
        Path to the main recording folder.

    Returns
    -------
    str
        Path to the data subfolder containing the .dat files.
    """
    folder_name = os.path.basename(folder_path)
    parent_dir = os.path.dirname(folder_path)
    dat_files = sorted(f for f in os.listdir(folder_path)
                       if f.lower().endswith('.dat'))
    dat_subfolder = os.path.join(folder_path, folder_name)

    if dat_files:
        print(f'Found {len(dat_files)} .dat files in main folder. '
              f'Reorganizing into {folder_name}/')
        os.makedirs(dat_subfolder, exist_ok=True)

        # Files that live inside the data folder alongside the .dat spools
        to_move_into_subfolder = dat_files + [
            'acquisitionmetadata.ini',
            'Spooled files.sifx',
        ]
        for fname in to_move_into_subfolder:
            src = os.path.join(folder_path, fname)
            if os.path.exists(src):
                shutil.move(src, os.path.join(dat_subfolder, fname))
        print(f'Moved data files to {dat_subfolder}')

        # Files that scapeio expects one level above the data folder
        for suffix in ('_info.mat', '_stim_data.bin'):
            src = os.path.join(parent_dir, folder_name + suffix)
            if os.path.exists(src):
                shutil.move(src, os.path.join(folder_path, folder_name + suffix))
                print(f'Moved {folder_name + suffix} to {folder_path}')
            else:
                print(f'Warning: {src} not found, skipping.')
    else:
        if not os.path.isdir(dat_subfolder):
            raise FileNotFoundError(
                f'No .dat files found in {folder_path} and no '
                f'{folder_name}/ subfolder exists.')
        print(f'Using existing data subfolder: {dat_subfolder}')

    return dat_subfolder


def average_act_scape(folder_path: str,
                      batch_size: int = 100) -> None:
    """
    Compute and plot per-volume mean and max fluorescence from SCAPE data.

    Loads the recording in batches to keep memory usage bounded, deskews
    each batch, and computes:
      - per-volume mean and max fluorescence (saved as PNGs in Figures/)
      - a per-voxel std image across all volumes (saved as Figures/std_image.png
        and as mean_std_stats.h5 with mean, mean_sq, and frame_count so the
        std can be updated later with additional data)

    Parameters
    ----------
    folder_path : str
        Path to the main recording folder. If it contains .dat files
        directly, they are moved into a RawData subfolder automatically.
        If not, a RawData subfolder must already exist.
    batch_size : int
        Number of volumes to load and process per batch.
    """
    folder_path = os.path.abspath(folder_path)
    dat_folder = _setup_raw_data_folder(folder_path)

    scan_duration, num_volumes, scan_rate = scapeio.get_scan_timing(dat_folder)
    print(f'Total volumes: {num_volumes}, '
          f'scan rate: {scan_rate:.2f} vol/s, '
          f'duration: {scan_duration:.1f} s')

    all_means = []
    all_maxes = []
    sum_x: np.ndarray | None = None
    sum_x2: np.ndarray | None = None
    frame_count = 0

    n_batches = int(np.ceil(num_volumes / batch_size))
    t0 = time.time()
    for batch_idx in range(n_batches):
        start_volume = batch_idx * batch_size
        end_volume = min(start_volume + batch_size, num_volumes)
        elapsed_min = (time.time() - t0) // 60
        elapsed_sec = (time.time() - t0) % 60
        print(f'Batch {batch_idx + 1}/{n_batches}: '
              f'volumes {start_volume}–{end_volume - 1}  '
              f'(elapsed: {elapsed_min:.0f} min {elapsed_sec:.0f} s)')

        video_batch, _ = scapeio.load_scan(
            dat_folder,
            colors=1,
            return_axis_order=True,
            start_volume=start_volume,
            end_volume=end_volume - 1)
        deskewed_batch = scapeio.operations.deskew_scan(
            dat_folder, images=video_batch)
        deskewed_batch = deskewed_batch.astype(np.float32)
        del video_batch

        # deskewed_batch shape: (n_vols_in_batch, Z, X, Y)
        spatial_axes = tuple(range(1, deskewed_batch.ndim))
        all_means.append(deskewed_batch.mean(axis=spatial_axes))
        all_maxes.append(deskewed_batch.max(axis=spatial_axes))

        if sum_x is None:
            sum_x = np.sum(deskewed_batch, axis=0)
            sum_x2 = np.sum(deskewed_batch ** 2, axis=0)
        else:
            sum_x += np.sum(deskewed_batch, axis=0)
            sum_x2 += np.sum(deskewed_batch ** 2, axis=0)
        frame_count += deskewed_batch.shape[0]
        del deskewed_batch

    means = np.concatenate(all_means)  # shape (num_volumes,)
    maxes = np.concatenate(all_maxes)

    mean_image = sum_x / frame_count
    mean_sq_image = sum_x2 / frame_count
    variance = np.maximum(0, mean_sq_image - np.square(mean_image))
    std_image = np.sqrt(variance)

    figures_dir = os.path.join(folder_path, 'Figures')
    os.makedirs(figures_dir, exist_ok=True)

    for values, label, color, filename in [
            (means, 'Mean fluorescence', 'steelblue', 'mean_activity.png'),
            (maxes, 'Max fluorescence',  'tomato',    'max_activity.png')]:
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(values, lw=0.8, color=color)
        ax.set_xlabel('Volume number')
        ax.set_ylabel(label)
        ax.set_title(f'{label} per volume over time')
        ax.set_xlim(0, num_volumes - 1)
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        out_path = os.path.join(figures_dir, filename)
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {out_path}')

    # std image: max projection across Z for a 2D summary
    std_proj = std_image.max(axis=0)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(std_proj.T, cmap='gray', aspect='auto')
    ax.set_title('Std image (Z max projection)')
    ax.axis('off')
    fig.tight_layout()
    std_png_path = os.path.join(figures_dir, 'std_image.png')
    fig.savefig(std_png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {std_png_path}')

    stats_path = os.path.join(folder_path, 'mean_std_stats.h5')
    with h5py.File(stats_path, 'w') as f:
        f.create_dataset('mean', data=mean_image, compression='gzip')
        f.create_dataset('mean_sq', data=mean_sq_image, compression='gzip')
        f.attrs['frame_count'] = frame_count
    print(f'Saved {stats_path}')


def main() -> None:
    """
    Run average_act_scape from the command line.
    """
    parser = argparse.ArgumentParser(
        description='Compute per-volume mean and max activity from SCAPE data')
    parser.add_argument(
        'folder_path',
        help='Path to the main recording folder')
    parser.add_argument(
        '--batch-size', type=int, default=100,
        help='Number of volumes to load per batch (default: 100)')
    args = parser.parse_args()
    average_act_scape(args.folder_path, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
