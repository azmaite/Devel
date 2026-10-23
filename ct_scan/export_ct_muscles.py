"""
Export the muscle labels of a CT scan traced in WebKnossos as a label volume at the voxel size of
the master reference (1.5 um isotropic), for `master_reference_3d/register_ct_labels.py`.

The WebKnossos export (`labels/` folder with the `.nml` file and `data_Volume`, see
`webknossos.load_labels`) holds one label per traced structure, grouped by leg and segment
(e.g. `RF_femur`, with nested subgroups). The voxel size is read from the `.nml` (`<scale>`, in
nm). The labels are resampled with nearest-neighbour interpolation and transposed to array order
(z, y, x) of the WebKnossos axes, so that SimpleITK's physical (x, y, z) are WebKnossos's (x, y, z).

Saves, in `out_folder`:
- `ct_labels_1p5um.tif`: uint32 label volume (all WebKnossos ids, 0 = unlabelled), (z, y, x),
- `ct_labels_1p5um.json`: `id_to_name`, `id_to_group` (innermost WebKnossos group of each id),
  `id_to_muscle` (the muscle each label is part of, with the name fixes and part merging of
  `ct_muscle_names.py`),
  `voxel_size_um` (1.5), `ct_voxel_size_um` (from the `.nml`), `array_axes`.

Run with the uv environment (it needs `wkw`): `python export_ct_muscles.py labels_folder
out_folder`.

ct_voxel_size_um: voxel size of a WebKnossos export, from its `.nml`.
export_ct_muscles: export the labels of a WebKnossos export at a given voxel size.
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import tifffile

import ct_muscle_names
import webknossos


def ct_voxel_size_um(labels_dir: str | Path) -> float:
    """
    Isotropic voxel size of a WebKnossos export, from the `<scale>` of its `.nml` (nm).

    Parameters
    ----------
    labels_dir : str | Path
        WebKnossos export folder (with the `.nml` file).

    Returns
    -------
    float
        Voxel size, in um.
    """
    root = ET.parse(next(Path(labels_dir).glob('*.nml'))).getroot()
    scale = root.find('parameters/scale').attrib
    sizes = {float(scale[axis]) for axis in ('x', 'y', 'z')}
    if len(sizes) != 1:
        raise ValueError(f'Anisotropic voxel size {scale} is not supported.')
    return sizes.pop() / 1000


def export_ct_muscles(labels_dir: str | Path,
                      out_folder: str | Path,
                      voxel_size_um: float = 1.5) -> Path:
    """
    Export the labels of a WebKnossos export at `voxel_size_um` (see the module docstring).

    Parameters
    ----------
    labels_dir : str | Path
        WebKnossos export folder.
    out_folder : str | Path
        Output folder (created if needed).
    voxel_size_um : float
        Output voxel size, in um.

    Returns
    -------
    Path
        Path of the saved label tif.
    """
    labels_dir, out_folder = Path(labels_dir), Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    ct_voxel = ct_voxel_size_um(labels_dir)
    labels, id_to_name = webknossos.load_labels(labels_dir)  # (x, y, z)
    id_to_group = webknossos.load_segment_group_names(labels_dir)
    root = ET.parse(next(labels_dir.glob('*.nml'))).getroot()
    id_to_name = {int(s.attrib['id']): s.attrib.get('name', '')
                  for s in root.find('volume/segments')} | id_to_name
    dtype = np.uint8 if max(id_to_name) < 256 else np.uint16
    labels = np.ascontiguousarray(labels.astype(dtype).transpose(2, 1, 0))  # (z, y, x)
    resampled = ndi.zoom(labels, ct_voxel / voxel_size_um, order=0).astype(np.uint32)
    path = out_folder / 'ct_labels_1p5um.tif'
    tifffile.imwrite(path, resampled, compression='zlib')
    (out_folder / 'ct_labels_1p5um.json').write_text(json.dumps(
        {'source': str(labels_dir), 'voxel_size_um': voxel_size_um,
         'ct_voxel_size_um': ct_voxel, 'array_axes': 'zyx (of the webknossos x, y, z)',
         'id_to_name': {int(k): v for k, v in sorted(id_to_name.items())},
         'id_to_group': {int(k): v for k, v in sorted(id_to_group.items())},
         'id_to_muscle': ct_muscle_names.muscle_names(dict(sorted(id_to_name.items())))},
        indent=1))
    print(f'Saved {path} {resampled.shape} ({len(np.unique(resampled)) - 1} labels; CT voxel '
          f'{ct_voxel:.4f} um)')
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export WebKnossos CT labels at 1.5 um.')
    parser.add_argument('labels_dir', help='WebKnossos export folder (.nml and data_Volume).')
    parser.add_argument('out_folder', help='Output folder.')
    parser.add_argument('-v', '--voxel_size', type=float, default=1.5,
                        help='Output voxel size, in um.')
    args = parser.parse_args()
    export_ct_muscles(args.labels_dir, args.out_folder, args.voxel_size)
