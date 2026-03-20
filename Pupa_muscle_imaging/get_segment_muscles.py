import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import av
from sklearn.decomposition import NMF
import tqdm


def _mask_image(image, mask):
    """Extract 1D array of image values under the mask."""
    return image[mask.astype(bool)]

def _unmask_image(values, mask, bounding_box=None):
    """Reconstruct 2D image from masked values, zeros elsewhere,
    and crop to bounding box (x, y, w, h) if provided."""
    result = np.zeros(mask.shape)
    result[mask.astype(bool)] = values

    # crop if bounding box provided
    if bounding_box is not None:
        x, y, w, h = bounding_box
        result = result[y:y+h, x:x+w]

    return result


def get_segment_muscles(video_path=''):

    # select the raw .mkv video file (if not provided)
    if video_path == '':
        main_path = '/mnt/labserver/data/MA/Development_project/Pupa_muscle_long_recordings'
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        mkv_file = filedialog.askopenfilename(initialdir=main_path, title='Select video .mkv file', filetypes=[('MKV files', '*.mkv')])

    # check that path provided exists and is a .mkv file
    assert mkv_file.endswith('.mkv'), 'Video path must be in .mkv format'
    assert os.path.exists(mkv_file), f'Video path {mkv_file} does not exist'

    # Folder to save images
    save_folder = os.path.join(os.path.dirname(mkv_file), 'NMF_segment_figures')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


    # load timestamps from metadata for plotting
    metadata = mkv_file.replace('_raw_tiff.mkv', '_tif_metadata.h5')
    with h5py.File(metadata, 'r') as f:
        timestamps = f['timestamps'][:] #in MICROSECONDS
    timestamps = timestamps.astype(np.int64)
    timestamps = timestamps - timestamps[0]    # set time as relative
    timestamps[timestamps < 0] = timestamps[timestamps < 0] + 24*3600*1e6 # add a full day of microseconds if the values are negative
    timestamps = timestamps / 1e6 # convert to seconds


    # load manually selected rois from h5 file
    h5_file = mkv_file.replace('.mkv', '_segment_rois.h5')
    rois = {}
    masks = {}
    bounding_boxes = {}

    with h5py.File(h5_file, 'r') as f:
        segment_names = f['segment_names'][:]
        segment_names = [name.decode('utf-8') for name in segment_names]
        ave_img = f['average_image'][:]
        for segment_name in segment_names:
            rois[segment_name] = f[f'{segment_name}_roi'][:]
            masks[segment_name] = f[f'{segment_name}_mask'][:]
            bounding_boxes[segment_name] = f[f'{segment_name}_bounding_box'][:]
    print(segment_names)

    # save the average image with the rois overlaid as a png for reference
    plt.figure(figsize=(10,10))
    plt.imshow(ave_img, cmap='gray')
    for segment_name in segment_names:
        roi = rois[segment_name]
        plt.plot(np.array(roi)[:,0], np.array(roi)[:,1], 'r-', linewidth=1)
    plt.axis('off')
    plt.savefig(os.path.join(save_folder, 'rois.png'))
    plt.close()


    # for each roi:
    # load the full video, mask the non-roi pixels, and crop to the roi bounding box, and save all frames to a list
    # also get the min_projection to use as baseline

    n = 24000 # length of video (10 min at 40 fps)
    all_frames_flat = {}
    min_proj_flat = {}

    container = av.open(mkv_file)

    print(f'Loading video frames ...')
    with tqdm.tqdm(total=n, unit="frame") as pbar:
        i = 0
        for frame in container.decode(video=0):
            # Load frames
            arr = frame.to_ndarray(format='rgba64le')
            for j in range(arr.shape[2]):
                img = arr[...,j]

                for segment_name in segment_names:

                    mask = masks[segment_name]
                    pixels = np.sum(mask)
                    bounding_box = bounding_boxes[segment_name]   

                    # initialize
                    if segment_name not in all_frames_flat:
                        all_frames_flat[segment_name] = np.zeros((n, pixels), dtype=np.float32)
                        min_proj_flat[segment_name] = np.full(pixels, np.inf, dtype=np.float32)   
                    
                    # Apply mask (pixels outside ROI become nan)
                    masked_flat = _mask_image(img, mask)
                    all_frames_flat[segment_name][i] = masked_flat

                    # Update min projection
                    min_proj_flat[segment_name] = np.minimum(min_proj_flat[segment_name], masked_flat)

                i += 1
                # update progress bar
                pbar.update(1)

    container.close()

    print('')
    for segment_name in segment_names:
        print(f'Processing segment {segment_name} ...')

        mask = masks[segment_name]
        roi = rois[segment_name]
        bounding_box = bounding_boxes[segment_name]  


        # get mean across time
        act = np.nanmean(all_frames_flat[segment_name], 1)
        # plot and save as png
        plt.figure(figsize=(15,3))
        plt.plot(timestamps, act)
        plt.xlim(0, timestamps[-1])
        plt.title(f'{segment_name} mean activity across time')
        plt.xlabel('Time (s)')
        plt.ylabel('Mean Activity')
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f'{segment_name}_mean_activity.png'))
        plt.close()


        # get diff from min projection for each frame
        all_frames_flat_dif = all_frames_flat[segment_name] - min_proj_flat[segment_name]

        # normalize each frame dif from 0 to 1 (ignore nans)
        all_frames_flat_dif = (all_frames_flat_dif - np.nanmin(all_frames_flat_dif)) / (np.nanmax(all_frames_flat_dif) - np.nanmin(all_frames_flat_dif))


        # find top X most different frames to use for NMF
        x = 200 #13 min for 200

        # get most different frame
        scores = all_frames_flat_dif.mean(axis=1)
        selected = [scores.argmax()]
        # calculate distance from selected frame to all other frames
        dists = np.linalg.norm(all_frames_flat_dif - all_frames_flat_dif[selected[0]], axis=1)
        # avoid reselecting
        dists[selected] = -np.inf

        # iteratively pick the most different frame
        print('Selecting most different frames for NMF...')
        for _ in tqdm.tqdm(range(x-1)):
            # select most different frame
            new_selected = np.argmax(dists)
            selected.append(new_selected)

            # calculate distance from new selected 
            dists_new = np.linalg.norm(all_frames_flat_dif - all_frames_flat_dif[new_selected], axis=1)
            # get min distance to any selected frame
            dists = np.minimum(dists, dists_new)

            # avoid reselecting any
            dists[selected] = -np.inf



        # fit NMF to get spatial components (H) from the selected frames
        n_components = 15
        selected_frames = all_frames_flat_dif[selected]

        model = NMF(n_components=n_components, init='random', random_state=0, max_iter=2000)
        _ = model.fit_transform(selected_frames)
        H_spatial_comp = model.components_


        # plot the spatial components as images with roi outline
        _, axes = plt.subplots(2, np.ceil(n_components/2).astype(int), figsize=(15, 5))
        axes = axes.flatten()
        for i in range(n_components):
            component = _unmask_image(H_spatial_comp[i], mask, bounding_box=bounding_box)
            axes[i].imshow(component, cmap='gray')
            roi = rois[segment_name]
            axes[i].plot(np.array(roi)[:,0]-bounding_box[0], np.array(roi)[:,1]-bounding_box[1], 'y-', linewidth=0.5)
            axes[i].set_title(f'Component {i}')
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f'{segment_name}_nmf_components.png'))
        plt.close()


        # get temporal components from all frames
        V = all_frames_flat_dif.astype(np.float32)
        model = NMF(n_components=n_components, init='custom', solver='cd', max_iter=2000)

        # "Lock in" your spatial components (H)
        model.components_ = H_spatial_comp.astype(np.float32) # W must be (n_components, n_pixels) here
        model.n_components_ = H_spatial_comp.shape[1]

        # Solve for W
        # V should be (n_frames, n_pixels)
        W_temporal_comp = model.transform(V) 


        # plot each temporal 0-1 with a vertical offset and save
        plt.figure(figsize=(15,5))
        for i in range(n_components):
            temp = W_temporal_comp[:,i]
            temp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
            plt.plot(timestamps, temp + i, label=i, linewidth=1)
        plt.xlim(0, timestamps[-1])
        plt.yticks(np.arange(n_components)+0.1, [f'Comp {i}' for i in range(n_components)])
        plt.xlabel('Time (s)')
        plt.title(f'{segment_name} NMF Temporal Components')
        plt.savefig(os.path.join(save_folder, f'{segment_name}_nmf_temporal_components.png'))
        plt.close()


        # SAVE THE SPATIAL AND TEMPORAL COMPONENTS IN AN H5 FILE
        with h5py.File(os.path.join(save_folder, f'{segment_name}_nmf_output.h5'), 'w') as f:
            f.create_dataset('H_spatial_components', data=H_spatial_comp)
            f.create_dataset('W_temporal_components', data=W_temporal_comp)
            f.create_dataset('roi', data=rois[segment_name])
            f.create_dataset('mask', data=masks[segment_name])
            f.create_dataset('bounding_box', data=bounding_boxes[segment_name])
            f.create_dataset('timestamps', data=timestamps)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Select frames from a video and save to H5.')
    parser.add_argument('--video_path', type=str, default='', help='Path to the input video file (e.g., .mkv). If not provided, will open a file dialog to select a video.')
    
    args = parser.parse_args()
    get_segment_muscles(args.video_path)