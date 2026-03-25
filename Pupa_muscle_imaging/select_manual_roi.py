"""
Select a free-hand Region of Interest (ROI) on the average of the first 4 frames of a .mkv video.
In the .mkv video, the first 4 frames correspond to 1 packet of data (saved as one frame with 4 channels).

use: 
  import select_manual_roi
  roi, mask, ave_img, bounding_box = select_manual_roi(video_path)

"""


import numpy as np
import os
import av
import cv2
import h5py


def select_manual_roi(video_path):

    # ask for input
    segment_names = input(f"Enter segment names separated by spaces (e.g., FL_femur FR_femur): ")
    segment_names = [name.strip() for name in segment_names.split(' ')]

    # select the raw .mkv video file (if not provided)
    if video_path == '':
        main_path = '/mnt/labserver/data/MA/Development_project/Pupa_muscle_long_recordings'
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        video_path = filedialog.askopenfilename(initialdir=main_path, title='Select video .mkv file', filetypes=[('MKV files', '*.mkv')])

    # check that path provided exists and is a .mkv file
    assert video_path.endswith('.mkv'), 'Video path must be in .mkv format'
    assert os.path.exists(video_path), f'Video path {video_path} does not exist'

    # check that segment_names is a list of strings
    assert isinstance(segment_names, list), 'segment_names must be a list of strings'

    # Load First 4 frames (1 packet) and average to get a clearer image for ROI selection
    container = av.open(video_path)
    first_packet = next(container.decode(video=0))
    first_packet = first_packet.to_ndarray(format='rgba64le')
    ave_frame = np.mean(first_packet, axis=2)  # Average across frames to get a single image
    
    # Normalize to 8-bit for OpenCV display
    ave_frame = cv2.normalize(ave_frame, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')


    # select as many rois as there are segment names provided
    all_rois = []

    drawing = False
    pts = []

    def _draw_roi(event, x, y, flags, param):
        nonlocal drawing, pts
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            pts = [(x, y)]
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            pts.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    cv2.namedWindow("Draw Free-hand ROI")
    cv2.setMouseCallback("Draw Free-hand ROI", _draw_roi)

    n_rois = len(segment_names)
    for i in range(n_rois):
        print(f"Draw ROI {segment_names[i]}, press ENTER when finished")

        pts = []

        while True:
            img_copy = ave_frame.copy()

            # draw previous ROIs (optional but useful)
            for roi in all_rois:
                cv2.polylines(img_copy, [np.array(roi)], True, 150, 1)

            # draw current ROI
            if len(pts) > 1:
                cv2.polylines(img_copy, [np.array(pts)], False, 255, 2)

            cv2.imshow("Draw Free-hand ROI", img_copy)

            key = cv2.waitKey(1)
            if key == 13:  # ENTER
                break

        if len(pts) > 2:
            all_rois.append(pts)

    cv2.destroyAllWindows()



    # Create Binary Mask from ROIs
    roi_masks = []
    for pts in all_rois:
        mask = np.zeros_like(ave_frame, dtype=np.uint8)
        if len(pts) > 0:
            cv2.fillPoly(mask, [np.array(pts)], 1)
        roi_masks.append(mask)

    # get roi bounding boxes
    bounding_boxes = []
    for pts in all_rois:
        x, y, w, h = cv2.boundingRect(np.array(pts))
        bounding_boxes.append([x, y, w, h])

    


    # save in an h5 file - in main folder
    fly_folder = os.path.dirname(os.path.dirname(video_path))
    h5_file = os.path.join(fly_folder, 'segment_rois.h5')
    # if h5 file doesn't exist, create it. If it does exist, add the new rois to it 
    if os.path.exists(h5_file):
        with h5py.File(h5_file, 'a') as f:

            # update segment names
            current_segment_names = f['segment_names'][:]
            current_segment_names = [name.decode('utf-8') for name in current_segment_names]
            for i, segment_name in enumerate(segment_names):
                    if segment_name in current_segment_names:
                        del f[f'{segment_name}_roi']
                        del f[f'{segment_name}_mask']
                        del f[f'{segment_name}_bounding_box']
                        raise Warning(f'Segment name {segment_name} already exists in {h5_file}. Overwriting...')
                    else:
                        current_segment_names.append(segment_name)
            del f['segment_names']
            f.create_dataset('segment_names', data=current_segment_names)

            for i, segment_name in enumerate(segment_names):
                f.create_dataset(f'{segment_name}_roi', data=all_rois[i])
                f.create_dataset(f'{segment_name}_mask', data=roi_masks[i])
                f.create_dataset(f'{segment_name}_bounding_box', data=bounding_boxes[i])
    else:
        with h5py.File(h5_file, 'w') as f:
            f.create_dataset('average_image', data=ave_frame)
            f.create_dataset('segment_names', data=segment_names)
            for i, segment_name in enumerate(segment_names):
                f.create_dataset(f'{segment_name}_roi', data=all_rois[i])
                f.create_dataset(f'{segment_name}_mask', data=roi_masks[i])
                f.create_dataset(f'{segment_name}_bounding_box', data=bounding_boxes[i])





# make this module callable so it can be used in the notebook without importing the function directly
# use: 
#   import select_manual_roi
#   roi, mask, ave_img, bounding_box = select_manual_roi(video_path)

import sys
sys.modules[__name__] = select_manual_roi



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Select frames from a video and save to H5.')
    parser.add_argument('--video_path', type=str, default='', help='Path to the input video file (e.g., .mkv). If not provided, will open a file dialog to select a video.')
    
    args = parser.parse_args()
    select_manual_roi(args.video_path)