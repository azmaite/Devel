import numpy as np
import os
from pathlib import Path
from skimage.registration import phase_cross_correlation

import scapeio
import scape_utils

MAIN_PATH = '/mnt/labserver/data/MA/SCAPE'



## select list of replicates and make analysis folder

# select single movie to align (or list of replicates)
# (also creates analysis folder and saves the list of selected replicates to scape_paths.json)
analysis_folder, recording_list = scape_utils.get_recording_list(MAIN_PATH, mkv=False)

# check that all folders start with sample and extract the sample numbers
assert all(Path(recording).name.startswith('sample') for recording in recording_list), "Not all folders start with 'sample'"
sample_n = [Path(recording).name.split('sample')[1][0] for recording in recording_list]
assert len(set(sample_n)) == 1, "Not all replicates are from the same sample"
sample_n = sample_n[0]

# rename the analysis folder to include the sample#
analysis_folder_new = analysis_folder + f'_sample{sample_n}'
os.rename(analysis_folder, analysis_folder_new)
analysis_folder = analysis_folder_new



## get a reference volume

# load 50 frames, deskew, and make a median volume from each recording 
n = 50
min_volumes = {}
for recording_folder in recording_list:
    video_batch, _ = scapeio.load_scan(
        recording_folder,
        colors=1,
        return_axis_order=True,
        start_volume=0,
        end_volume=n-1)
    deskewed_batch = scapeio.operations.deskew_scan(
        recording_folder, images=video_batch)
    del video_batch
    deskewed_batch = deskewed_batch.astype(np.float32)
    min_volumes[recording_folder] = np.min(deskewed_batch, axis=0)
    del deskewed_batch

# check that there is no movement between the replicates by calculating the shift between the min volumes of each to the first
# if not, Warning
for n in range(1,len(recording_list)):
    shift, error, phasediff = phase_cross_correlation(
        min_volumes[list(min_volumes.keys())[0]],
        min_volumes[list(min_volumes.keys())[n]],
    )
    if not np.allclose(shift, 0):
        print(f"WARNING: Replicate {n} is not aligned with the first replicate. Shift: {shift}")


## load the reference cuticle meshes and interactively align them to the reference volume
cuticle_mesh_path = '/mnt/labserver/data/MA/CT_scans_PIXE/2026-06-16_Webknossos_muscle_leg_tracing_pupa_annotations/Leg segments in pupa'

meshes = scape_utils.load_leg_meshes(cuticle_mesh_path)

# voxel size is anisotropic (SCAPE voxels are not cubic), needed to display
# the volume projections without stretching them
tube_lens = '50 mm Navitar'  # always!
reference_volume = min_volumes[recording_list[0]]
voxel_size_zxy = scapeio.get_voxel_size(recording_list[0], tube_lens, deskewed=True)

alignment_params = scape_utils.align_meshes_to_volume(
    reference_volume,
    meshes,
    voxel_size_zxy=voxel_size_zxy,
    save_path=Path(analysis_folder) / 'mesh_alignment.h5')

