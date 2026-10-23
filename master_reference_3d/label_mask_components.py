"""
Label the connected components of the automatic muscle mask of hemibody images, so that each
component can be assigned to a leg segment by hand (a quick way to segment the leg segments for
`register_segments.py`).

For each hemibody tif (e.g. from `split_hemibodies.py`), the binary muscle mask is computed with
`muscle_mask.segment_muscles` (default settings, 1.5 um voxels, except that
components of at least `KEEP_VOLUME_UM3` are kept whatever their shape: the default elongation
test drops bulky muscles, which matter here), and its
connected components (face connectivity, so components touching only at an edge or corner stay
separate) are numbered from 1 by decreasing size. The labels are saved as a plain uint32 tif
(opens as a Labels layer in napari, same array shape and order as the image) as
`{out_folder}/{image name}_components.tif`, with a json of the voxel count of each label, and a
plain tif copy of the image (`{image name}_image.tif`, no scale) so both open together at scale 1.

Run from the terminal: `python label_mask_components.py image.ome.tif [image ...] -o out_folder`.

label_components: binary muscle mask of an image, labelled by connected component.
label_hemibody: label the clusters of one hemibody tif and save them.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import tifffile

import muscle_mask

KEEP_VOLUME_UM3 = 30000.0  # mask components at least this large skip the elongation test


def label_components(image: np.ndarray,
                     voxel_size_um: float = 1.5) -> np.ndarray:
    """
    Compute the automatic muscle mask of an image and label its connected components.

    Parameters
    ----------
    image : np.ndarray
        3D image (isotropic voxels).
    voxel_size_um : float
        Voxel size, in um.

    Returns
    -------
    np.ndarray
        uint32 labels, 0 = background, 1..N by decreasing component size.
    """
    # large regions are kept whatever their shape, so that bulky muscles are not dropped
    mask = muscle_mask.segment_muscles(image, voxel_size_um=voxel_size_um,
                                       keep_volume_um3=KEEP_VOLUME_UM3)
    labels, n = ndi.label(mask)
    sizes = np.bincount(labels.ravel())[1:]
    new_ids = np.zeros(n + 1, dtype=np.uint32)
    new_ids[1 + np.argsort(-sizes, kind='stable')] = np.arange(1, n + 1, dtype=np.uint32)
    return new_ids[labels]


def label_hemibody(image_path: str | Path,
                   out_folder: str | Path,
                   voxel_size_um: float = 1.5) -> Path:
    """
    Label the muscle clusters of one hemibody tif and save them (see the module docstring).

    Parameters
    ----------
    image_path : str | Path
        Hemibody tif (e.g. from `split_hemibodies.py`).
    out_folder : str | Path
        Output folder (created if needed).
    voxel_size_um : float
        Voxel size, in um.

    Returns
    -------
    Path
        Path of the saved `{name}_components.tif`.
    """
    image_path, out_folder = Path(image_path), Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    name = image_path.name.removesuffix('.tif').removesuffix('.ome')
    image = tifffile.imread(image_path)
    labels = label_components(image, voxel_size_um)
    labels_path = out_folder / f'{name}_components.tif'
    tifffile.imwrite(labels_path, labels, compression='zlib')
    # plain tif copy of the image (no OME scale), so it opens at the same scale (1) as the
    # labels when both are dropped into napari
    tifffile.imwrite(out_folder / f'{name}_image.tif', image.astype(np.float32))
    sizes = np.bincount(labels.ravel())[1:]
    with open(out_folder / f'{name}_components.json', 'w') as f:
        json.dump({'image': str(image_path), 'array_axes': 'same as the image',
                   'voxel_size_um': voxel_size_um,
                   'voxels_per_label': {int(i): int(s) for i, s in
                                        enumerate(sizes, start=1)}}, f, indent=1)
    print(f'{name}: {len(sizes)} components, largest {sizes[:5].tolist()} voxels, '
          f'{(sizes < 1000).sum()} under 1000 voxels')
    return labels_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Label the connected components of the '
                                                 'automatic muscle mask of hemibody images.')
    parser.add_argument('images', nargs='+', help='Hemibody tifs.')
    parser.add_argument('-o', '--out_folder', required=True, help='Output folder.')
    parser.add_argument('-v', '--voxel_size', type=float, default=1.5,
                        help='Voxel size, in um.')
    args = parser.parse_args()
    for image_path in args.images:
        label_hemibody(image_path, args.out_folder, args.voxel_size)
