"""
Open a manually-labeled segmentation TIFF in napari as an editable Labels
layer, with the sample's max projection shown alongside it for reference.

Run from the terminal:
    python edit_labels_napari.py
"""

from pathlib import Path

import napari
import tifffile

from scape_utils import load_voxel_size, load_proj

folder = Path('/mnt/labserver/data/MA/SCAPE/20260716_MA12_pupa-P15/sample2_combined')
tif_list = [
            #folder / 'RLF-RLM_leg_segments.tif',
            folder / 'NMF_segmentation' / 'all_muscles_NMF_components.tif',
            
            # folder / 'NMF_segmentation' / 'LM_coxa_NMF_components_12_joined.tif',
            # folder / 'NMF_segmentation' / 'LM_trochanter_NMF_components_12_joined.tif',
            # folder / 'NMF_segmentation' / 'LM_femur_NMF_components_12_joined.tif',
            # folder / 'NMF_segmentation' / 'LM_tibia_NMF_components_12_joined.tif',
            # folder / 'NMF_segmentation' / 'RM_coxa_NMF_components_12_joined.tif',
            # folder / 'NMF_segmentation' / 'RM_trochanter_NMF_components_12_joined.tif',
            # folder / 'NMF_segmentation' / 'RM_femur_NMF_components_12_joined.tif',
            # folder / 'NMF_segmentation' / 'RM_tibia_NMF_components_12_joined.tif',

            # folder / 'NMF_segmentation' / 'LF_coxa_NMF_components_12_joined.tif',
            # folder / 'NMF_segmentation' / 'LF_trochanter_NMF_components_12_joined.tif',
            # folder / 'NMF_segmentation' / 'LF_femur_NMF_components_12_joined.tif',
            # folder / 'NMF_segmentation' / 'LF_tibia_NMF_components_12_joined.tif',
            # folder / 'NMF_segmentation' / 'RF_coxa_NMF_components_12_joined.tif',
            # folder / 'NMF_segmentation' / 'RF_trochanter_NMF_components_12_joined.tif',
            # folder / 'NMF_segmentation' / 'RF_femur_NMF_components_12_joined.tif',
            # folder / 'NMF_segmentation' / 'RF_tibia_NMF_components_12_joined.tif',
            ]

voxel_sizes = load_voxel_size(folder)
scale = (voxel_sizes['z'], voxel_sizes['x'], voxel_sizes['y'])  # matches these arrays' own (z, x, y) axis order


proj = load_proj(folder, 'max_proj')

viewer = napari.Viewer()
viewer.add_image(proj, name='max_proj', scale=scale)
for tif in tif_list:
    labels = tifffile.imread(tif)
    viewer.add_labels(labels, name=tif.stem, scale=scale)

napari.run()
