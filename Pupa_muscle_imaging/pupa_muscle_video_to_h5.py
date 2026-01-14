import cv2
import h5py
import numpy as np
import os


def extract_recording_number(filename):
        # 1. Remove the 'recording' prefix
        part_after_prefix = filename.split('recording', 1)[1]
        # 2. Extract the number up to the next underscore
        number_string = part_after_prefix.split('_', 1)[0]
        # 3. Convert the extracted string to an integer
        return int(number_string)



# set main paths
main_path_leandre = '/mnt/upramdya_data/LD/Data/recording_experiments/imaging/Automated_experiment'
main_path = '/mnt/upramdya_data/MA/Development_project/LD'

for pupa_num in range(1, 13):
    print(f"Processing pupa {pupa_num}/12")

    # get list of recordings sorted by number 
    # (recoring# is not zero-padded in the folder names)
    pupa_path_leandre = os.path.join(main_path_leandre, f'pupa_{pupa_num}')
    recording_list = os.listdir(pupa_path_leandre)

    # set main path to save and processed data outside of Leandre's folder
    pupa_path = os.path.join(main_path, f'pupa_{pupa_num}')
    os.makedirs(pupa_path, exist_ok=True)

    # remove any non-recording files
    recording_list = [f for f in recording_list if 'recording' in f]

    # sort recordings by recording number
    recording_list.sort(key=extract_recording_number)


    # save videos as h5 files
    for recording_num in range(len(recording_list)):
        # get video path (in INVERSE order)
        video_filename = 'output_video.mp4'
        recording_inverse_num = len(recording_list) - recording_num - 1
        raw_video_path = os.path.join(pupa_path_leandre, recording_list[recording_inverse_num], video_filename)
        print(f"  Processing video {recording_num + 1}/{len(recording_list)}: {recording_list[recording_inverse_num]}")

        # set h5 output path
        raw_h5_path = os.path.join(pupa_path, recording_list[recording_inverse_num], 'video_raw.h5')
        os.makedirs(os.path.dirname(raw_h5_path), exist_ok=True)

        # skip if h5 file already exists
        if os.path.exists(raw_h5_path):
            print(f"    H5 file already exists, skipping.")
            continue

        # get video properties
        cap = cv2.VideoCapture(raw_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # create h5 file
        with h5py.File(raw_h5_path, 'w') as f:
            # create dataset
            raw_frames = f.create_dataset('frames', (total_frames, frame_height, frame_width), dtype='uint8')

            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                raw_frames[i] = np.mean(frame, axis=2).astype('uint8')  # convert to grayscale

        cap.release()
