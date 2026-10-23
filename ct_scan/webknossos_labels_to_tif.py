"""
Script to convert manual labels drawn in webknossos on a SCAPE run's projection image (e.g. its
min projection) to a label tif that napari opens as a Labels layer, like
`RLF-RLM_leg_segments.tif` (uint32 ids, 0 = unlabelled).

Expects the webknossos export folder (one `.nml` file and a `data_Volume` folder, see
`webknossos.load_labels`) of labels drawn on a dataset made from the run's min projection
(`analysis/min_proj.ome.tif`), and that run's `analysis/projection_images.h5`, whose projection
images set the frame the labels are placed in. The labels are saved in the same axis order as
the OME-TIFF projections (z, y, x; webknossos indexes them (x, y, z)). This is NOT the axis
order of the h5 projections returned by `scape_utils.load_proj` (z, x, y): transpose the labels
with `.transpose(0, 2, 1)` to overlay them on those (see `edit_labels_napari.py`).

Saves to the run's analysis folder `{out_name}.tif` (a plain uint32 tif, which napari opens as
a Labels layer; the voxel size is stored in its tags but napari ignores it, so set the layer
scale yourself) and `{out_name}_ids.json`, which holds `id_to_name` (the webknossos segment
names), `id_to_group` (the webknossos group/folder each segment was put in), the axis order and
`voxel_size_um_zyx`.

Can be run from the terminal
(`python webknossos_labels_to_tif.py run_folder labels_folder out_name [-n ID=NAME ...] [-i ID ...]`)
or imported and called as a function. Needs numpy, h5py, tifffile and wkw (not scapeio).

webknossos_labels_to_tif: convert one webknossos label export to a label tif and an ids
    json file in a run's analysis folder.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import tifffile

# webknossos.py is in this folder: make it importable however this script is launched
sys.path.insert(0, str(Path(__file__).resolve().parent))
from webknossos import load_labels_in_frame, load_segment_group_names

ANALYSIS_PATH_NAME = 'analysis'                  # same as scape_utils.ANALYSIS_PATH_NAME
PROJECTION_FILE_NAME = 'projection_images.h5'    # same as scape_utils.PROJECTION_FILE_NAME


def webknossos_labels_to_tif(run_path: str | Path,
                             labels_path: str | Path,
                             out_name: str,
                             id_to_name_extra: dict[int, str] | None = None,
                             ignore_ids: tuple[int, ...] = ()) -> Path:
    """
    Convert a webknossos label export drawn on a run's projection image to a label tif (see the
    module docstring), saved with an ids json file in the run's analysis folder.

    Parameters
    ----------
    run_path : str | Path
        Path to the `sample#_run#` folder whose analysis folder holds the projections.
    labels_path : str | Path
        Path to the webknossos export folder (with the `.nml` file and `data_Volume`).
    out_name : str
        Name (without extension) of the output files in the analysis folder.
    id_to_name_extra : dict of int to str, optional
        Names for ids that are not named in webknossos (they are added, and override the
        webknossos names).
    ignore_ids : tuple of int, optional
        Ids to leave out: their voxels are set to 0 and their names dropped.

    Returns
    -------
    Path
        Path to the saved tif.
    """
    analysis_path = Path(run_path) / ANALYSIS_PATH_NAME
    with h5py.File(analysis_path / PROJECTION_FILE_NAME, 'r') as f:
        proj_shape = f['min_proj'].shape                    # (z, x, y)
        voxel_size_xyz = f['voxel_sizes_xyz'][:]            # um
    shape_xyz = (proj_shape[1], proj_shape[2], proj_shape[0])

    labels_xyz, id_to_name = load_labels_in_frame(labels_path, shape_xyz)
    id_to_group = load_segment_group_names(labels_path)
    id_to_name.update(id_to_name_extra or {})

    for ignored in ignore_ids:
        labels_xyz[labels_xyz == ignored] = 0
        id_to_name.pop(ignored, None)
        id_to_group.pop(ignored, None)

    labels = np.ascontiguousarray(np.transpose(labels_xyz, (2, 1, 0)), dtype=np.uint32)

    present = {int(i) for i in np.unique(labels) if i != 0}
    unnamed = sorted(present - set(id_to_name))
    unpainted = sorted(set(id_to_name) - present)
    if unnamed:
        print(f'Warning: ids {unnamed} are painted but have no name.')
    if unpainted:
        print(f'Warning: ids {unpainted} are named but not painted.')
    names = id_to_name
    duplicate_names = sorted({n for n in names.values() if list(names.values()).count(n) > 1})
    if duplicate_names:
        print(f'Warning: several ids share the names {duplicate_names}.')

    # a plain (not OME) uint32 tif is opened by napari as a Labels layer, but napari's tif reader
    # ignores the voxel size, so it is only stored in the tif tags/ids json for reference
    voxel_size_zyx = [float(voxel_size_xyz[2]), float(voxel_size_xyz[1]), float(voxel_size_xyz[0])]
    tif_path = analysis_path / f'{out_name}.tif'
    tifffile.imwrite(tif_path, labels,
                     resolution=(1 / voxel_size_xyz[0], 1 / voxel_size_xyz[1]),
                     metadata={'spacing': voxel_size_zyx[0], 'unit': 'um', 'axes': 'ZYX'})
    with open(analysis_path / f'{out_name}_ids.json', 'w') as f:
        json.dump({'axes': 'zyx',
                   'voxel_size_um_zyx': voxel_size_zyx,
                   'id_to_name': {str(k): v for k, v in sorted(names.items())},
                   'id_to_group': {str(k): v for k, v in sorted(id_to_group.items())},
                   'source': str(labels_path)}, f, indent=2)

    print(f'Saved {tif_path} (shape {labels.shape}, {labels.dtype}, {len(present)} labels: '
          f'{sorted(present)}) and {out_name}_ids.json')
    return tif_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert webknossos labels to a napari label tif.')
    parser.add_argument('run_path', help='sample#_run# folder.')
    parser.add_argument('labels_path', help='webknossos export folder (with .nml and data_Volume).')
    parser.add_argument('out_name', help='Output name, without extension.')
    parser.add_argument('-n', '--name', action='append', default=[], metavar='ID=NAME',
                        help='Name for an id that has none in webknossos (repeatable).')
    parser.add_argument('-i', '--ignore-id', action='append', type=int, default=[],
                        help='Id to leave out (repeatable).')
    args = parser.parse_args()
    extra = {int(k): v for k, v in (item.split('=', 1) for item in args.name)}
    webknossos_labels_to_tif(args.run_path, args.labels_path, args.out_name,
                             id_to_name_extra=extra, ignore_ids=tuple(args.ignore_id))
