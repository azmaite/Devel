"""
Utility functions for NMF muscle roi detection. Includes functions for:

- masking and unmasking functions to apply masks to images and videos
    masked_image = mask_image (image, mask) 
    unmasked_image = unmask_image (values, mask, bounding_box)
    masked_video = mask_video (video, mask)
    unmasked_video = unmask_video (masked_frames, mask, bounding_box)
    
- running NMF and selecting/processing frames for NMF
    frames_flat_dif_norm = diff_norm_frames (frames_flat, min_proj_flat=None)
    W_temp = nmf_get_temporal (frames, H, **nmf_kwargs):
    W_temp, H_spat = fit_nmf (frames, n_components, selected_idx=None, **nmf_kwargs)
    var_explained = nmf_variance_explained (V, W, H)
    selected_idx = select_most_different_frames (frames_flat_dif_norm, n_select=200)
    
- plotting temporal and spatial components
    plot_temporal_components(W, timestamps=None, order=None, color=None, norm=True, fig=None)
    plot_spatial_components(H, mask, bounding_box, roi, order=None)
    
- loading data from mkv files and corresponding h5 files with rois, masks, bounding boxes, and timestamps
    mkv_list =  get_mkv_list()
    rois, masks, bounding_boxes, segment_names = load_segment_rois(mkv_file)
    timestamps = load_timestamps(mkv_file)
    min_proj, max_proj = load_min_max_proj(mkv_file)
    frames_flat, min_proj_flat, (frames_square) = load_mkv_roi(mkv_file, roi_names, masks, bounding_boxes=None)
    
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import h5py
import av
import tqdm


### MASKING AND UNMASKING FUNCTIONS ###

def mask_image(image, mask):
    """Extract 1D array of image values under the mask."""
    return image[mask.astype(bool)]

def unmask_image(values, mask, bounding_box=None):
    """Reconstruct 2D image from masked values, zeros elsewhere,
    and crop to bounding box (x, y, w, h) if provided."""
    result = np.zeros(mask.shape)
    result[mask.astype(bool)] = values

    # crop if bounding box provided
    if bounding_box is not None:
        x, y, w, h = bounding_box
        result = result[y:y+h, x:x+w]
        
    return result

def mask_video(video, mask):
    """Apply mask to each frame of video, returning 2D array of masked values."""
    return np.array([mask_image(frame, mask) for frame in video])

def unmask_video(masked_frames, mask, bounding_box=None):
    """Reconstruct video from masked frames, using unmask_image for each frame."""
    return np.array([unmask_image(frame, mask, bounding_box) for frame in masked_frames])



### RUNNING NMF FUNCTIONS ###
# W=temporal components, H=spatial components

def diff_norm_frames(frames, min_proj=None):
    """ For each frame, get the difference from the min projection across all frames,
    and normalize the difference from 0 to 1 (ignoring nans). This can help to enhance the signal for NMF.
    Frames can be flat or not."""
    
    if min_proj is None:
        min_proj = np.nanmin(frames, axis=0)

    # get diff from min projection for each frame
    frames_dif = frames - min_proj

    # normalize each frame dif from 0 to 1 (ignore nans)
    frames_dif_norm = (frames_dif - np.nanmin(frames_dif)) / (np.nanmax(frames_dif) - np.nanmin(frames_dif))

    return frames_dif_norm


def nmf_get_temporal(frames, H, **nmf_kwargs):
    """ Given spatial components H, solve for temporal components W using frames."""
    
    # set default NMF parameters, which can be overridden by user input
    kwargs = dict(n_components=H.shape[0], max_iter=2000, init='custom')
    kwargs.update(nmf_kwargs)
    kwargs.update({'init': 'custom'}) # make sure for this second step we use the custom initialization regardless of user input
    
    #run
    model = NMF(**kwargs)
    model.components_ = H.astype(np.float32) # lock the spatial components to those learned from the selected frames
    W = model.transform(frames) #solve for W using all frames
    
    return W


def fit_nmf(frames, n_components, selected_idx=None, **nmf_kwargs):
    """ Fit NMF to the given frames, returning temporal and spatial components.
    If selected_idx is provided, only use those frames to learn the spatial components, 
    but then solve for temporal components using all frames. 
    This allows us to get spatial components from a subset of frames (e.g. those with high activity) 
    while still getting temporal components for the whole video."""
    
    if selected_idx is None:
        selected_idx = np.arange(frames.shape[0])
    
    # set default NMF parameters, which can be overridden by user input
    kwargs = dict(n_components=n_components, max_iter=2000, init='random', random_state=0)
    kwargs.update(nmf_kwargs)
    # run initial NMF on selected frames to learn spatial components
    model = NMF(**nmf_kwargs)
    W = model.fit_transform(frames[selected_idx])
    H = model.components_

    # if not all frames were used for fitting, we need to get temporal components from all frames
    if len(selected_idx) < frames.shape[0]:

        # set default NMF parameters, which can be overridden by user input
        kwargs = dict(n_components=n_components, max_iter=2000, init='custom')
        kwargs.update(nmf_kwargs)
        kwargs.update({'init': 'custom'}) # make sure for this second step we use the custom initialization regardless of user input

        # Get temporal components
        W = nmf_get_temporal(frames, H, **nmf_kwargs)
    
    return W, H # W=temporal components, H=spatial components


def nmf_variance_explained(V, W, H): 
    """Calculate the variance explained (R^2) of the NMF reconstruction."""

    V_hat = W @ H

    ss_res = np.linalg.norm(V - V_hat, 'fro')**2
    ss_tot = np.linalg.norm(V - V.mean(), 'fro')**2

    r2 = 1 - ss_res / ss_tot

    return r2


def select_most_different_frames(frames_flat_dif_norm, n_select=200):
    """ Given a set of frames, select the n_select most different frames to use for NMF."""

    # initalize by getting the first most different frame
    scores = frames_flat_dif_norm.mean(axis=1)
    selected_idx = [scores.argmax()]
    
    # calculate distance from selected frame to all other frames
    dists = np.linalg.norm(frames_flat_dif_norm - frames_flat_dif_norm[selected_idx[0]], axis=1)
    
    # avoid reselecting
    dists[selected_idx] = -np.inf

    # iteratively pick the most different frame
    print('Selecting most different frames for NMF...')
    for _ in tqdm.tqdm(range(n_select-1)):
        
        # select most different frame
        new_selected_idx = np.argmax(dists)
        selected_idx.append(new_selected_idx)

        # calculate distance from new selected 
        dists_new = np.linalg.norm(frames_flat_dif_norm - frames_flat_dif_norm[new_selected_idx], axis=1)
        # get min distance to any selected frame
        dists = np.minimum(dists, dists_new)

        # avoid reselecting any
        dists[selected_idx] = -np.inf

    return selected_idx



### PLOTTING FUNCTIONS ###

def _check_order(order, n_components):
    """Check that order works for the n_components given.
    - order must have as least as many elements as n_components
    - order must have as many non-nan elements as n_components
    - all non-nan elements in order must be unique and between 0 and n_components"""

    # check that order has as least as many elements as n_components
    assert len(order) >= n_components, 'Order must be at least as long as the number of components'
    
    # check that the number of non-nan elements in order matches the number of n_components
    assert np.sum(~np.isnan(order)) == n_components, 'Order must have as many non-nan elements as components'
    
    # check that all non-nan elements are unique and between 1 and n_components 
    order = np.array(order)
    non_nan = order[~np.isnan(order)].astype(int)
    assert np.array_equal(np.sort(non_nan), np.arange(1, n_components+1)), 'All non-nan elements must be unique and go from 1 to n_components'


def plot_temporal_components(W, timestamps=None, order=None, color=None, norm=True, fig=None, xlims=None):
    """ Plot the temporal components as line plots, with options for ordering, coloring, and normalization."""

    if timestamps is None:
        timestamps = np.arange(W.shape[0])
    if order is None:
        order = np.arange(W.shape[1])
    else:
        # check that order is valid
        _check_order(order, W.shape[1])
        order = [ord-1 for ord in order]

    if color is not None and len(color) == 1:
        color = color * len(order)
        
    
    blank = np.zeros(len(timestamps))
    if not fig:
        plt.figure(figsize=(15,5))
        
    for i, ord in enumerate(order):
        # if ord is nan, plot blank line
        if np.isnan(ord):
            plt.plot(timestamps, blank-1)
        else: 
            temp = W[:,ord]
            if norm:
                temp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
            if color is not None:
                plt.plot(timestamps, temp+i, label=f'Comp {ord}', linewidth=1, color=color[i])
            else:
                plt.plot(timestamps, temp+i, label=f'Comp {ord}', linewidth=1)
            
    plt.ylim(-0.5, len(order)+0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Component Activation')
    plt.title('Temporal Components from NMF')
    plt.legend()

    if xlims is not None:
        plt.xlim(xlims)
    else:
        plt.xlims(timestamps[0], timestamps[-1])

    if not fig:
        plt.show()


def plot_spatial_components(H, mask, bounding_box, roi=None, order=None):
    """ Plot the spatial components as images, with the roi contour overlaid (if present)."""
    
    if order is None:
        order = np.arange(H.shape[0])
    else:
        # check that order is valid
        _check_order(order, H.shape[0])
        order = [ord-1 for ord in order]

    
    # set number of rows and image size based on number of components
    cols = 6
    rows = int(np.ceil(len(order) / cols))
    _, axes = plt.subplots(rows,cols, figsize=(15,2*rows))
    axes = axes.flatten()
    
    # plot
    for n, ord in enumerate(order):
        if np.isnan(ord):
            axes[n].axis('off')
            continue
        
        img = unmask_image(H[ord], mask, bounding_box)
        axes[n].imshow(img, cmap='viridis')
        # overlay the roi contour if present
        if roi is not None:
            roi_cropped = roi - bounding_box[:2]  # adjust roi coordinates to cropped image
            axes[n].plot(np.array(roi_cropped)[:,0], np.array(roi_cropped)[:,1], 'r--', linewidth=0.4)
        axes[n].axis('off')
        axes[n].set_title(f'Component {ord}', fontsize=8)
        
    for j in range(n+1, len(axes)):
        axes[j].axis('off')
    plt.show()



### LOADING DATA FUNCITONS ###

def get_mkv_list():
    """Open a file dialog to select a folder, returning a list
    of all mkv files within it and its subfolders."""

    import tkinter as tk
    from tkinter import filedialog

    main_path = '/mnt/labserver/data/MA/Development_project/Pupa_muscle_long_recordings'

    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(initialdir=main_path, title='Select folder containing recordings')

    # get list of all .mkv files in folder and subfolders
    mkv_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.mkv'):
                mkv_list.append(os.path.join(root, file))

    # sort the list of files
    mkv_list.sort()
    # print the number of files
    print(f'Found {len(mkv_list)} .mkv files')

    return mkv_list


def load_segment_rois(mkv_file):
    """ Given an mkv file, load the segment rois, masks, and bounding boxes 
    from the corresponding h5 file."""
    
    # if mkv_file is a list of files, use the first one
    if isinstance(mkv_file, list):
        mkv_file = mkv_file[0]

    # get all rois, masks, and bounding boxes
    rois = {}
    masks = {}
    bounding_boxes = {}
    fly_folder = os.path.dirname(os.path.dirname(mkv_file))
    h5_file = os.path.join(fly_folder, 'segment_rois.h5')
    with h5py.File(h5_file, 'r') as f:
        segment_names = f['segment_names'][:]
        segment_names = [name.decode('utf-8') for name in segment_names]
        for segment_name in segment_names:
            rois[segment_name] = f[f'{segment_name}_roi'][:]
            masks[segment_name] = f[f'{segment_name}_mask'][:]
            bounding_boxes[segment_name] = f[f'{segment_name}_bounding_box'][:]

    return rois, masks, bounding_boxes, segment_names


def load_timestamps(mkv_file):
    """ Given an mkv file, load the timestamps in SECONDS from the corresponding h5 metadata file,"""
    
    metadata = mkv_file.replace('_raw_tiff.mkv', '_tif_metadata.h5')
    with h5py.File(metadata, 'r') as f:
        timestamps = f['timestamps'][:] #in MICROSECONDS
    timestamps = timestamps.astype(np.int64)
    timestamps = timestamps - timestamps[0]    # set time as relative
    timestamps[timestamps < 0] = timestamps[timestamps < 0] + 24*3600*1e6 # add a full day of microseconds if the values are negative
    timestamps = timestamps / 1e6 # convert to seconds

    return timestamps


def load_min_max_proj(mkv_file):
    """ Given an mkv file, load the min and max projection across all frames from the corresponding h5 file,
    generated by preprocess_pupa_muscle script"""
    
    h5_path = os.path.join(os.path.dirname(mkv_file), 'max_min_proj.h5')
    with h5py.File(h5_path, 'r') as f:
        min_proj = f['min_proj'][:]
        max_proj = f['max_proj'][:]

    return min_proj, max_proj


def load_mkv_roi(mkv_file, roi_names, masks, bounding_boxes=None):
    """ Given an mkv file, load:
        - the video frames for each roi in roi_names, applying the corresponding mask to each frame
            so that only pixels within the mask are included and flattened
        - the min projection across all frames from the given rois
        If bounding boxes are provided, it also returns the frames cropped to the bounding box but without removing pixels outside the mask
            *boungind box should be in format (x, y, w, h) and should be provided for each roi in roi_names*
    """
    import av

    # check that all roi_names are in masks
    for roi_name in roi_names:
        assert roi_name in masks, f'ROI for {roi_name} not found in masks'

    n = 24000 # length of video (10 min at 40 fps)
    all_frames_flat = {}
    min_proj_flat = {}
    if bounding_boxes is not None:
        # check that all roi_names are in bounding_boxes
        for roi_name in roi_names:
            assert roi_name in bounding_boxes, f'Bounding box for {roi_name} not found in bounding_boxes'
        all_frames_square = {}

    container = av.open(mkv_file)

    print(f'Loading video frames ...')
    with tqdm.tqdm(total=n, unit="frame") as pbar:
        i = 0
        for frame in container.decode(video=0):
            # Load frames
            arr = frame.to_ndarray(format='rgba64le')
            for j in range(arr.shape[2]):
                img = arr[...,j]

                for roi_name in roi_names:

                    mask = masks[roi_name]
                    pixels = np.sum(mask)

                    # initialize arrays for this roi if not already done
                    if roi_name not in all_frames_flat:
                        all_frames_flat[roi_name] = np.zeros((n, pixels), dtype=np.float32)
                        min_proj_flat[roi_name] = np.full(pixels, np.inf, dtype=np.float32)  
                        if bounding_boxes is not None:
                            all_frames_square[roi_name] = np.zeros((n, bounding_boxes[roi_name][3], bounding_boxes[roi_name][2]), dtype=np.float32) 
                    
                    # Apply mask (pixels outside ROI become nan)
                    masked_flat = mask_image(img, mask)
                    all_frames_flat[roi_name][i] = masked_flat

                    # Update min projection
                    min_proj_flat[roi_name] = np.minimum(min_proj_flat[roi_name], masked_flat)

                    # if bounding box provided, also save the square frames cropped to the bounding box but without masking
                    if bounding_boxes is not None:
                        x, y, w, h = bounding_boxes[roi_name]
                        cropped = img[y:y+h, x:x+w]
                        all_frames_square[roi_name][i] = cropped

                i += 1
                # update progress bar
                pbar.update(1)

    container.close()

    if bounding_boxes is not None:
        return all_frames_flat, min_proj_flat, all_frames_square
    else:
        return all_frames_flat, min_proj_flat

