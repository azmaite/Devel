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

main_path = Path('/mnt/labserver/data/MA/SCAPE/')
folder = main_path / '20261016_V7xPR_pupa_long_paired' / 'sample1_run3_short' 

tif_list = [
            #folder / 'RLF-RLM_leg_segments.tif',
            folder / 'analysis' / 'leg_muscles.tif',
            folder / 'analysis' / 'leg_segments.tif',
            ]

# labels converted from webknossos (webknossos_labels_to_tif.py) are (z, y, x) like the
# OME-TIFF projections, but load_proj returns (z, x, y), so they are transposed on loading
zyx_tif_list = [folder / 'analysis' / 'leg_muscles.tif',
                folder / 'analysis' / 'leg_segments.tif']

voxel_sizes = load_voxel_size(folder)
scale = (voxel_sizes['z'], voxel_sizes['x'], voxel_sizes['y'])  # matches these arrays' own (z, x, y) axis order


proj1 = load_proj(folder, 'min_proj')
proj2 = load_proj(folder, 'max_proj')

viewer = napari.Viewer()
viewer.add_image(proj1, name='min_proj', scale=scale)
viewer.add_image(proj2, name='max_proj', scale=scale)
for tif in tif_list:
    labels = tifffile.imread(tif)
    if tif in zyx_tif_list:
        labels = labels.transpose(0, 2, 1)  # (z, y, x) -> (z, x, y), as load_proj returns
    viewer.add_labels(labels, name=tif.stem, scale=scale)

napari.run()
