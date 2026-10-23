"""
Register a new recording to the finished master reference, and bring a label map of the master
(e.g. leg segments, or later identified muscles) back onto the recording, in the recording's own
frame, so that the recording can be analysed without being deformed.

Starts, like `build_master_reference.py`, from the projection images of the recording
(`{sample#_run#}/analysis/projection_images.h5`, computed in the `scape` folder), and uses the
final master reference saved by `build_master_reference.py` (`master_reference/master_segments.tif`
and `.json`: the master muscle mask labelled by leg segment). The steps are those of the master,
for one recording:
1. split the minimum projection into its two isotropic hemibodies (`split_hemibodies.py`),
2. compute their muscle masks (`label_mask_components.py`; no manual labelling needed),
3. register each hemibody to the master muscle mask, leg segment by leg segment
   (`registration_round.register_hemibody`, with `master_segments.tif` as the template), and
   bring it into the master frame
   (`registration_round.warp_to_template`). These registered images can be used to improve the
   master: add the recording to `RECORDINGS` in `build_master_reference.py` and rerun the rounds,
4. bring the label map back: for each voxel of a hemibody, the master positions given by the
   segment transforms (target -> master) are averaged with weights that fall off with the
   distance to each master segment as placed in the hemibody by its own transform (the inverse of
   the blending of step 3), and the label map is sampled there (nearest label). The hemibody
   labels are then mapped back through the split (`split_hemibodies.hemibodies_to_original`) onto
   the grid of the original projection.

Outputs, in `out_folder` (default `{sample#_run#}/analysis/master_registration`):
- `hemibodies/`: the hemibodies, their split json/png and muscle clusters (as in
  `master_reference`),
- `{name}_{side}_registered.tif`, `{name}_{side}_registered_muscles.tif`: the hemibodies in the
  master frame (as the `_registered` files of a round),
- `{name}_{side}_labels.tif`: the label map on each hemibody (uint32, hemibody frame),
- `master_labels.tif`: the label map on the original projection (uint32, same shape and (Z, X, Y)
  order as the projections in `projection_images.h5`, so it overlays `load_proj` arrays; for the
  `.ome.tif` projections, which are (Z, Y, X), use `.transpose(0, 2, 1)`), with
  `master_labels.json` (label names) and `master_labels.png` (labels over the projection),
- `registration.json`: the fits and their checks.

Run from the terminal: `python register_recording.py sample#_run#_folder [--master_path ...]
[--label_map labels.tif --label_names names.json]` (see `--help`), or import and call
`register_recording`.

register_recording: register one recording to the master and bring the label map back.
labels_to_hemibody: sample a master label map on a registered hemibody.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import tifffile

import label_mask_components
import register_segments as rs
import registration_round as rr
import split_hemibodies

MASTER_PATH = Path('/mnt/labserver/data/MA/SCAPE/master_reference')
VOXEL_SIZE_UM = rr.VOXEL_SIZE_UM


def labels_to_hemibody(label_map: np.ndarray,
                       template_segments: np.ndarray,
                       fit: dict,
                       field_of_view: np.ndarray) -> np.ndarray:
    """
    Sample a label map of the master on a registered hemibody (step 4 of the module docstring).

    Parameters
    ----------
    label_map : np.ndarray
        Integer labels on the master grid.
    template_segments : np.ndarray
        Master segment map (1.., same grid), whose segments set the blending weights.
    fit : dict
        Output of `registration_round.register_hemibody` for the hemibody (transforms from
        hemibody to master coordinates).
    field_of_view : np.ndarray
        Boolean field of view of the hemibody (same grid); labels are kept only inside it.

    Returns
    -------
    np.ndarray
        Labels on the hemibody grid (dtype of `label_map`).
    """
    shape = label_map.shape
    weights = []
    for index, transform in enumerate(fit['transforms'], start=1):
        # the master segment as placed in the hemibody by its own transform
        placed = rs.warp_labels((template_segments == index).astype(np.uint8), transform) > 0
        distance = ndi.distance_transform_edt(~placed, sampling=VOXEL_SIZE_UM)
        weights.append(np.exp(-distance ** 2 / (2 * rr.BLEND_SIGMA_UM ** 2)).astype(np.float32))
    global_weight = float(np.exp(-rr.GLOBAL_DISTANCE_UM ** 2 / (2 * rr.BLEND_SIGMA_UM ** 2)))
    (labels,) = rr._warp_blended([label_map], fit['transforms'] + [fit['global_fit']],
                                 weights + [global_weight], shape, order=0)
    return np.where(field_of_view, labels, 0).astype(label_map.dtype)


def register_recording(recording: str | Path,
                       master_path: str | Path = MASTER_PATH,
                       out_folder: Optional[str | Path] = None,
                       label_map: Optional[str | Path] = None,
                       label_names: Optional[str | Path] = None) -> Path:
    """
    Register one recording to the master and bring a master label map back onto it (see the
    module docstring).

    Parameters
    ----------
    recording : str | Path
        `sample#_run#` folder, with `analysis/projection_images.h5`.
    master_path : str | Path
        Folder with the final master reference (`master_segments.tif` and `.json`, saved by
        `build_master_reference.py`).
    out_folder : str | Path or None
        Output folder. None: `{recording}/analysis/master_registration`.
    label_map : str | Path or None
        uint tif of labels on the master grid (e.g. identified muscles). None: the leg segments of
        the master muscle mask.
    label_names : str | Path or None
        json of the label names (`{"id_to_name": {"1": "name", ...}}` or `{"1": "name", ...}`), for
        `master_labels.json`. Ignored if `label_map` is None (the segment names are used).

    Returns
    -------
    Path
        Path of `master_labels.tif`.
    """
    recording, master_path = Path(recording), Path(master_path)
    out_folder = Path(out_folder) if out_folder else recording / 'analysis' / 'master_registration'
    hemibody_folder = out_folder / 'hemibodies'
    name = f'{recording.parent.name}_{recording.name}'

    # 1-2. hemibodies and their muscle clusters
    split_hemibodies.split_recording(recording, hemibody_folder, target_voxel_size=VOXEL_SIZE_UM)
    for side in ('L', 'R'):
        if not (hemibody_folder / f'{name}_{side}_components.tif').exists():
            label_mask_components.label_hemibody(hemibody_folder / f'{name}_{side}.ome.tif',
                                                 hemibody_folder, VOXEL_SIZE_UM)

    # master template and label map, on a grid that fits the master and both hemibodies
    template_segments = tifffile.imread(master_path / 'master_segments.tif').astype(np.uint32)
    names = json.loads((master_path / 'master_segments.json').read_text())['id_to_name']
    segment_names = [names[str(i)] for i in range(1, len(names) + 1)]
    if label_map is None:
        labels = template_segments
        id_to_name = {i: segment for i, segment in enumerate(segment_names, start=1)}
    else:
        labels = tifffile.imread(label_map).astype(np.uint32)
        if labels.shape != template_segments.shape:
            raise ValueError(f'label_map shape {labels.shape} is not the master grid '
                             f'{template_segments.shape}')
        id_to_name = {}
        if label_names is not None:
            content = json.loads(Path(label_names).read_text())
            id_to_name = {int(k): v for k, v in content.get('id_to_name', content).items()}
    master_shape = template_segments.shape
    hemibody_shapes = [tifffile.TiffFile(hemibody_folder / f'{name}_{side}.ome.tif').series[0].shape
                       for side in ('L', 'R')]
    shape = tuple(int(s) for s in np.max([master_shape] + hemibody_shapes, axis=0))
    template_segments, labels = rr._pad(template_segments, shape), rr._pad(labels, shape)
    weights, near, global_weight = rr.segment_weights(template_segments, len(segment_names))

    # 3-4. register each hemibody, bring it into the master frame, and the labels onto it
    report = {'master': str(master_path), 'segments': segment_names, 'hemibodies': {}}
    hemibody_labels = {}
    for side, hemibody_shape in zip(('L', 'R'), hemibody_shapes):
        start = time.time()
        image = rr._pad(tifffile.imread(hemibody_folder / f'{name}_{side}.ome.tif')
                        .astype(np.float32), shape)
        components = rr._pad(tifffile.imread(hemibody_folder / f'{name}_{side}_components.tif'),
                             shape)
        muscles = components > 0
        fit = rr.register_hemibody(template_segments, segment_names, image, muscles, near,
                                   label=f'{name}_{side}')
        registered, valid, registered_muscles = rr.warp_to_template(
            image, muscles, muscles, fit, weights, global_weight)
        crop = tuple(slice(0, s) for s in master_shape)
        tifffile.imwrite(out_folder / f'{name}_{side}_registered.tif', registered[crop])
        tifffile.imwrite(out_folder / f'{name}_{side}_registered_muscles.tif',
                         registered_muscles[crop].astype(np.uint32), compression='zlib')
        entry = fit['entry']
        entry['muscle_dice_all'] = rr._dice(registered_muscles, (template_segments > 0) & valid)
        report['hemibodies'][side] = entry

        hemibody_labels[side] = labels_to_hemibody(labels, template_segments, fit,
                                                   image > 0)[tuple(slice(0, s)
                                                                    for s in hemibody_shape)]
        tifffile.imwrite(out_folder / f'{name}_{side}_labels.tif', hemibody_labels[side],
                         compression='zlib')
        print(f'{name}_{side}: muscle Dice with the master {entry["muscle_dice_all"]:.2f} '
              f'({time.time() - start:.0f}s)')
        (out_folder / 'registration.json').write_text(json.dumps(report, indent=1))

    # labels on the original projection
    split_info = json.loads((hemibody_folder / f'{name}_split.json').read_text())
    with h5py.File(recording / 'analysis' / 'projection_images.h5', 'r') as f:
        projection = f[split_info['projection']][:]
    original_labels = split_hemibodies.hemibodies_to_original(
        hemibody_labels['L'], hemibody_labels['R'], split_info, projection.shape, order=0)
    labels_path = out_folder / 'master_labels.tif'
    tifffile.imwrite(labels_path, original_labels.astype(np.uint32), compression='zlib')
    present = [int(i) for i in np.unique(original_labels) if i > 0]
    (out_folder / 'master_labels.json').write_text(json.dumps(
        {'recording': str(recording), 'master': str(master_path),
         'label_map': str(label_map) if label_map else 'leg segments of the master',
         'array_axes': 'ZXY (as the projections in projection_images.h5)',
         'id_to_name': {i: id_to_name.get(i, str(i)) for i in present}}, indent=1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    for ax, axis in zip(axes, (0, 2)):
        ax.imshow(projection.max(axis=axis), cmap='gray',
                  vmax=np.percentile(projection, 99.9), aspect='auto')
        label_projection = original_labels.max(axis=axis).astype(float)
        ax.imshow(np.ma.masked_equal(label_projection, 0), cmap='tab20', alpha=0.4,
                  interpolation='nearest', aspect='auto')
        ax.set_title(f'{name}: master labels (max along axis {axis})', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_folder / 'master_labels.png', dpi=100)
    plt.close(fig)
    print(f'Saved {labels_path} ({len(present)} labels)')
    return labels_path


if __name__ == '__main__':
    matplotlib.use('Agg')
    parser = argparse.ArgumentParser(description='Register a recording to the master reference '
                                                 'and bring a master label map back onto it.')
    parser.add_argument('recording', type=Path, help='sample#_run# folder.')
    parser.add_argument('--master_path', type=Path, default=MASTER_PATH,
                        help='Folder with the final master reference (master_segments.tif).')
    parser.add_argument('--out_folder', type=Path, default=None,
                        help='Output folder (default: {recording}/analysis/master_registration).')
    parser.add_argument('--label_map', type=Path, default=None,
                        help='Labels on the master grid (default: its leg segments).')
    parser.add_argument('--label_names', type=Path, default=None,
                        help='json of the label names.')
    args = parser.parse_args()
    register_recording(args.recording, args.master_path, args.out_folder, args.label_map,
                       args.label_names)
