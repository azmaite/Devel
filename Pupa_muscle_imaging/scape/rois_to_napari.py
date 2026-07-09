""" Convert SCAPE ROIs to NRRD format to load in napari.
For the 3d rois to match scape video data, it must be deskewed before loading into napari. 
Run scape_convert . -1 --batches 5 --deskew using conda environment scapeio from
the raw_data_path to get the deskewed data that can be opened in napari.
Then visualize the video using scape_view sample1_run12_deskewed_batch2of5.nrrd -1 --deskewed
(replace the nnrd file with the one you want to visualize).
"""



import h5py
import scapeio
import numpy as np
from pathlib import Path
import nrrd
from typing import OrderedDict
import tkinter as tk
from tkinter import filedialog

# get path to data
main_path = Path('/mnt/labserver/data/MA/SCAPE') 
root = tk.Tk()
root.withdraw()
exp_path = filedialog.askdirectory(
    initialdir=main_path, title='Select experiment folder (contains segmentation_corr.h5)')
root.destroy()

assert exp_path, 'No experiment folder selected.'
exp_path = Path(exp_path)
assert exp_path.exists(), f'Experiment folder {exp_path} does not exist.'

# check that segmentation_corr.h5 exists in the selected folder
segmentation_file = exp_path / 'corr_output/segmentation_corr.h5'
assert segmentation_file.exists(), f'Segmentation file {segmentation_file} does not exist.'

# check that raw data folder exists in the selected folder
raw_data_path = exp_path / exp_path.name
assert raw_data_path.exists(), f'Raw data folder {raw_data_path} does not exist.'



# get rois
with h5py.File(segmentation_file, 'r') as f:
    labels_3d = f['labels_3d'][:]

# get voxel size for deskewed data
tube_lens = '50 mm Navitar' #always!
voxel_size_skewed = scapeio.get_voxel_size(raw_data_path, tube_lens, deskewed=True)

# make a header from scratch - ordered dict!
header = OrderedDict([
    ('type', 'uint8'),
    ('dimension', 3),
    ('space dimension', 3),
    ('sizes', np.array([labels_3d.shape[2], labels_3d.shape[1], labels_3d.shape[0]])),
    ('space directions', np.array([
        [voxel_size_skewed[2], 0, 0],
        [0, voxel_size_skewed[1], 0],
        [0, 0, voxel_size_skewed[0]] 
    ])),
    ('encoding', 'raw'),
    ('space units', ['microns', 'microns', 'microns'])])


# reverse transpose back to NRRD storage order
rois_to_save = np.transpose(labels_3d, (2, 1, 0))
roi_save_path = exp_path / 'rois_not_deskewed.nrrd'
nrrd.write(
    str(roi_save_path),
    rois_to_save,
    header
)