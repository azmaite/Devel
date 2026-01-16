import cv2
import h5py
import numpy as np
import os

## function to compute shift between two images using cross-correlation via FFT
from scipy.signal import fftconvolve
def fast_xcorr_shift(im0, im1):
    # subtract means to reduce global illumination differences
    a = im0 - np.mean(im0)
    b = im1 - np.mean(im1)

    # cross-correlation using FFT convolution
    corr = fftconvolve(a, b[::-1, ::-1], mode='same')

    # peak gives the shift
    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    dy = y - corr.shape[0] // 2
    dx = x - corr.shape[1] // 2
    
    return dx, dy, corr


## function to shift image by (dx, dy)
def shift_image(im, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted_im = cv2.warpAffine(im, M, (im.shape[1], im.shape[0]))
    return shifted_im



# set main path
main_path = '/mnt/upramdya_data/MA/Development_project/LD'


for pupa_num in range(1, 13):
    print(f"Processing pupa {pupa_num}/12")

    # get list of recordings sorted by number 
    # (recoring# is not zero-padded in the folder names)
    pupa_path = os.path.join(main_path, f'pupa_{pupa_num}')
    recording_list = os.listdir(pupa_path)

    # remove any non-recording files
    recording_list = [f for f in recording_list if 'recording' in f]

    def extract_recording_number(filename):
        # 1. Remove the 'recording' prefix
        part_after_prefix = filename.split('recording', 1)[1]
        # 2. Extract the number up to the next underscore
        number_string = part_after_prefix.split('_', 1)[0]
        # 3. Convert the extracted string to an integer
        return int(number_string)

    recording_list.sort(key=extract_recording_number)

    # calculate best shift in x and y for each frame compared to reference (average of first 10 frames)
    # save as h5 file

    for recording_num in range(len(recording_list)):

        print(f"  Processing recording {recording_num + 1}/{len(recording_list)}: {recording_list[recording_num]}")
        # h5 file to save motion-corrected video
        motioncorr_h5_path = os.path.join(pupa_path, recording_list[recording_num], 'video_motioncorr.h5')

        # skip if motioncorr h5 file already exists
        if os.path.exists(motioncorr_h5_path):
            print(f"   SKIP - Motion-corrected H5 file already exists.")
            continue

        # Read the original from the h5 file
        raw_h5_path = os.path.join(pupa_path, recording_list[recording_num], 'video_raw.h5')
        with h5py.File(raw_h5_path, 'r') as f_raw:
            frames_raw = f_raw['frames'][:]

        # Get video properties
        frame_width = frames_raw.shape[2]
        frame_height = frames_raw.shape[1]
        total_frames = frames_raw.shape[0]


        # align first 10 frames and get average as reference
        first_frames = frames_raw[0:10]
        ref_frame = np.mean(first_frames, axis=0)

        # Create motioncorr HDF5 file
        with h5py.File(motioncorr_h5_path, "w") as f_motioncorr:
            frames_motioncorr = f_motioncorr.create_dataset("frames_motioncorr", shape=(total_frames, frame_height, frame_width), dtype=frames_raw.dtype)
            x_shift = f_motioncorr.create_dataset("x_shift", shape=(total_frames,), dtype=frames_raw.dtype)
            y_shift = f_motioncorr.create_dataset("y_shift", shape=(total_frames,), dtype=frames_raw.dtype)

            # iterate through all frames
            for frame_num, frame in enumerate(frames_raw):

                # compute shift compared to reference
                dx, dy, _ = fast_xcorr_shift(ref_frame, frame)

                # shift image
                frame_shifted = shift_image(frame, dx, dy)

                # save to h5
                frames_motioncorr[frame_num, :, :] = frame_shifted
                x_shift[frame_num] = dx
                y_shift[frame_num] = dy

                if frame_num % 1000 == 0 and frame_num > 0:
                    print(f"   Processed {frame_num}/{total_frames} frames")