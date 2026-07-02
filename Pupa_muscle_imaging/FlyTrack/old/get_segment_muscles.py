#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import h5py
import matplotlib.pyplot as plt
import av
from sklearn.decomposition import NMF
import tqdm

import utils


def get_segment_muscles(mkv_file: str = '') -> None:

    # select the raw .mkv video file (if not provided)
    if mkv_file == '':
        main_path = '/mnt/labserver/data/MA/Development_project/Pupa_muscle_long_recordings'
        import tkinter as tk
        from tkinter import filedialog
        tk_root = tk.Tk()
        tk_root.withdraw()
        mkv_file = filedialog.askopenfilename(initialdir=main_path, title='Select video .mkv file', filetypes=[('MKV files', '*.mkv')])

    print(f'Video file selected: {mkv_file}')
    assert mkv_file.endswith('.mkv'), 'Video path must be in .mkv format'
    assert Path(mkv_file).exists(), f'Video path {mkv_file} does not exist'

    save_folder = Path(mkv_file).parent / 'NMF_segment_figures'
    save_folder.mkdir(exist_ok=True)

    # load timestamps from metadata for plotting
    timestamps, _ = utils.load_timestamps(mkv_file)

    # load manually selected rois from h5 file
    rois, masks, bounding_boxes, segment_names = utils.load_segment_rois(mkv_file)
    segment_names = [name for name in segment_names if name != 'all_legs']
    print(segment_names)

    # load the full video, mask the non-roi pixels per segment, and track max projection
    n = utils.FRAMES_PER_RECORDING
    all_frames_flat: dict = {}
    min_proj_flat: dict = {}
    max_proj: np.ndarray | None = None

    container = av.open(mkv_file)

    print('Loading video frames ...')
    with tqdm.tqdm(total=n, unit='frame') as pbar:
        i = 0
        for frame in container.decode(video=0):
            arr = frame.to_ndarray(format='rgba64le')
            for j in range(arr.shape[2]):
                img = arr[..., j]

                if max_proj is None:
                    max_proj = img.astype(np.float32)
                else:
                    max_proj = np.maximum(max_proj, img.astype(np.float32))

                for segment_name in segment_names:
                    mask = masks[segment_name]
                    pixels = np.sum(mask)

                    if segment_name not in all_frames_flat:
                        all_frames_flat[segment_name] = np.zeros((n, pixels), dtype=np.float32)
                        min_proj_flat[segment_name] = np.full(pixels, np.inf, dtype=np.float32)

                    masked_flat = utils.mask_image(img, mask)
                    all_frames_flat[segment_name][i] = masked_flat
                    min_proj_flat[segment_name] = np.minimum(min_proj_flat[segment_name], masked_flat)

                i += 1
                pbar.update(1)

    container.close()

    # save the max_proj image with the rois overlaid as a png for reference
    assert max_proj is not None
    plt.figure(figsize=(10, 10))
    plt.imshow(max_proj, cmap='gray')
    for segment_name in segment_names:
        roi = rois[segment_name]
        plt.plot(np.array(roi)[:, 0], np.array(roi)[:, 1], 'r-', linewidth=1)
    plt.axis('off')
    plt.savefig(save_folder / 'rois.png')
    plt.close()

    print('')
    for segment_name in segment_names:
        print(f'Processing segment {segment_name} ...')

        mask = masks[segment_name]
        roi = rois[segment_name]
        bounding_box = bounding_boxes[segment_name]

        # plot mean activity across time and save
        act = np.nanmean(all_frames_flat[segment_name], 1)
        plt.figure(figsize=(15, 3))
        plt.plot(timestamps, act)
        plt.xlim(0, timestamps[-1])
        plt.title(f'{segment_name} mean activity across time')
        plt.xlabel('Time (s)')
        plt.ylabel('Mean Activity')
        plt.tight_layout()
        plt.savefig(save_folder / f'{segment_name}_mean_activity.png')
        plt.close()

        # get diff from min projection and normalize 0-1
        all_frames_flat_dif = all_frames_flat[segment_name] - min_proj_flat[segment_name]
        all_frames_flat_dif = (all_frames_flat_dif - np.nanmin(all_frames_flat_dif)) / (np.nanmax(all_frames_flat_dif) - np.nanmin(all_frames_flat_dif))

        # find top 200 most different frames to use for NMF
        selected = utils.select_most_different_frames(all_frames_flat_dif, n_select=200)

        # fit NMF to get spatial components (H) from the selected frames
        print('Fitting NMF to selected frames...')
        n_components = 160 if segment_name == 'all_legs' else 15
        selected_frames = all_frames_flat_dif[selected]

        model = NMF(n_components=n_components, init='random', random_state=0, max_iter=2000)
        _ = model.fit_transform(selected_frames)
        H_spatial_comp = model.components_

        # plot the spatial components as images with roi outline
        if segment_name == 'all_legs':
            _, axes = plt.subplots(np.ceil(n_components / 10).astype(int), 10, figsize=(15, 30))
        else:
            _, axes = plt.subplots(2, np.ceil(n_components / 2).astype(int), figsize=(15, 5))
        axes = axes.flatten()
        for i in range(n_components):
            component = utils.unmask_image(H_spatial_comp[i], mask, bounding_box=bounding_box)
            axes[i].imshow(component, cmap='gray')
            axes[i].plot(np.array(roi)[:, 0] - bounding_box[0], np.array(roi)[:, 1] - bounding_box[1], 'y-', linewidth=0.5)
            axes[i].set_title(f'Component {i}')
            axes[i].axis('off')
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        plt.savefig(save_folder / f'{segment_name}_nmf_components.png')
        plt.close()

        # get temporal components from all frames
        print('Getting temporal components from all frames ...')
        V = all_frames_flat_dif.astype(np.float32)
        model = NMF(n_components=n_components, init='custom', solver='cd', max_iter=2000)
        model.components_ = H_spatial_comp.astype(np.float32)
        model.n_components_ = H_spatial_comp.shape[1]
        W_temporal_comp = model.transform(V)

        # plot each temporal component 0-1 with a vertical offset and save
        plt.figure(figsize=(15, 5))
        for i in range(n_components):
            temp = W_temporal_comp[:, i]
            temp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
            plt.plot(timestamps, temp + i, label=i, linewidth=1)
        plt.xlim(0, timestamps[-1])
        plt.yticks(np.arange(n_components) + 0.1, [f'Comp {i}' for i in range(n_components)])
        plt.xlabel('Time (s)')
        plt.title(f'{segment_name} NMF Temporal Components')
        plt.savefig(save_folder / f'{segment_name}_nmf_temporal_components.png')
        plt.close()

        # save spatial and temporal components to h5
        with h5py.File(save_folder / f'{segment_name}_nmf_output.h5', 'w') as f:
            f.create_dataset('H_spatial_components', data=H_spatial_comp)
            f.create_dataset('W_temporal_components', data=W_temporal_comp)
            f.create_dataset('roi', data=rois[segment_name])
            f.create_dataset('mask', data=masks[segment_name])
            f.create_dataset('bounding_box', data=bounding_boxes[segment_name])
            f.create_dataset('timestamps', data=timestamps)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Select frames from a video and save to H5.')
    parser.add_argument('--video_path', type=str, default='', help='Path to the input video file (e.g., .mkv). If not provided, will open a file dialog to select a video.')

    args = parser.parse_args()
    get_segment_muscles(args.video_path)
