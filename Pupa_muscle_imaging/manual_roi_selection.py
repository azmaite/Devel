import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from matplotlib.widgets import LassoSelector


pupa_num = 1  # set pupa number (1, 2, etc.)
recording_num = 59


# get list of recordings sorted by number 
# (recoring# is not zero-padded in the folder names)
main_path = '/mnt/upramdya_data/MA/Development_project/LD'
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

print(f"Selected recording: {recording_list[recording_num]}")
recording_path = os.path.join(pupa_path, recording_list[recording_num])


# get average image of first 100 frames
h5_file = os.path.join(recording_path, 'video_raw.h5')
with h5py.File(h5_file, 'r') as f:
    # Access the data you need from the HDF5 file
    ave = np.mean(f['frames'][:][0:100], axis=0)

print(f"Loaded average image")


# manual ROI selection
rois = []

fig, ax = plt.subplots()
ax.imshow(ave)
ax.set_title("Draw ROI (freehand). Close window when done.")
ax.axis("off")

def on_select(verts):
    verts = np.array(verts)
    rois.append(verts)
    ax.plot(verts[:, 0], verts[:, 1], '-r')
    fig.canvas.draw_idle()

lasso = LassoSelector(ax, on_select)

plt.show()
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()

# rois is a list of (N, 2) arrays of polygon vertices


# save rois to a file
roi_file = os.path.join(recording_path, 'manual_rois.pkl')
with open(roi_file, 'wb') as f:
    pickle.dump(rois, f)

print(f"Saved {len(rois)} ROIs to {roi_file}")