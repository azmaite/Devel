"""
Register a new recording to the finished master reference, and bring a label map of the master
(e.g. leg segments, or later identified muscles) back onto the recording, in the recording's own
frame, so that the recording can be analysed without being deformed.

Starts, like `master_mean_generation/build_master_mean.py`, from the projection images of the
recording (`{sample#_run#}/analysis/projection_images.h5`, computed in the `scape` folder), and
uses the master saved by `build_master_mean.py` (`master_segments.tif` and `.json`: the master
muscle mask labelled by leg segment, in `build_master_mean.OUT_PATH`). The steps are those of the
master (the modules are imported from `registration/`), for one recording:
1. split the projection the master was built from (`build_master_mean.PROJ_TYPE`, the median)
   into its two isotropic hemibodies (`registration/hemibodies.py`),
2. compute their muscle masks (`registration/muscle_mask.py`; no manual labelling needed),
3. register each hemibody to the master muscle mask, leg segment by leg segment
   (`deformation.register_hemibody`, with `master_segments.tif` as the template), and
   bring it into the master frame
   (`deformation.warp_to_template`),
4. bring the label map back: for each voxel of a hemibody, the master positions given by the
   segment transforms (target -> master) are blended as in step 3, but around the master
   segments as placed in the hemibody by their own transforms (the inverse of the blending of
   step 3: each segment's own transform inside it, smooth blending in the gaps), and the label
   map is sampled there (nearest label). The hemibody
   labels are then mapped back through the split (`hemibodies.hemibodies_to_original`) onto
   the grid of the original projection. The master image (`master_mean.tif`) is brought back
   the same way (linear interpolation), so the master is seen aligned to the undeformed
   recording.

Outputs, in `out_folder` (default `{sample#_run#}/analysis/registration_to_{master folder name}`):
- `hemibodies/`: the hemibodies, their split json/png and muscle clusters (as for the master),
- `{name}_{side}_registered.tif`, `{name}_{side}_registered_muscles.tif`: the hemibodies in the
  master frame (as the `_registered` files of a round),
- `{name}_{side}_labels.tif`: the label map on each hemibody (uint32, hemibody frame),
- `master_labels.tif`: the label map on the original projection (uint32, same shape and (Z, X, Y)
  order as the projections in `projection_images.h5`, so it overlays `load_proj` arrays; for the
  `.ome.tif` projections, which are (Z, Y, X), use `.transpose(0, 2, 1)`), with
  `master_labels.json` (label names) and `master_labels.png` (labels over the projection),
- `master_mean_on_recording.tif`: the master image on the original projection (float32, same
  shape and order as `master_labels.tif`; 0 outside the hemibodies' field of view), with
  `master_mean_on_recording.png` (projection in red, master in green),
- `registration.json`: the fits and their checks, and `segment_overlap.csv`: the overlap of each
  registered hemibody with the master muscle mask, per segment (as for a registration round).

Run from the terminal: `python register_recording.py sample#_run#_folder [--master_path ...]
[--label_map labels.tif --label_names names.json]` (see `--help`), or import and call
`register_recording`.

register_recording: register one recording to the master and bring the label map back.
master_to_hemibody: sample a master label map (or image) on a registered hemibody.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).parent / 'registration'))
sys.path.insert(0, str(Path(__file__).parent / 'master_mean_generation'))
import build_master_mean  # noqa: E402
import deformation as dfm  # noqa: E402
import hemibodies  # noqa: E402
import muscle_mask  # noqa: E402
import register_segments as rs  # noqa: E402

MASTER_PATH = build_master_mean.OUT_PATH
PROJ_TYPE = build_master_mean.PROJ_TYPE
VOXEL_SIZE_UM = dfm.VOXEL_SIZE_UM


def master_to_hemibody(label_map: np.ndarray,
                       template_segments: np.ndarray,
                       fit: dict,
                       field_of_view: np.ndarray,
                       order: int = 0) -> np.ndarray:
    """
    Sample a label map (or image) of the master on a registered hemibody (step 4 of the module
    docstring).

    Parameters
    ----------
    label_map : np.ndarray
        Integer labels (or, with `order` 1, an image) on the master grid.
    template_segments : np.ndarray
        Master segment map (1.., same grid), whose segments set the blending weights.
    fit : dict
        Output of `deformation.register_hemibody` for the hemibody (transforms from
        hemibody to master coordinates).
    field_of_view : np.ndarray
        Boolean field of view of the hemibody (same grid); labels are kept only inside it.
    order : int
        Interpolation order: 0 (nearest) for labels, 1 (linear) for images.

    Returns
    -------
    np.ndarray
        Labels on the hemibody grid (dtype of `label_map`; float32 for `order` 1).
    """
    shape = label_map.shape
    # the master segments as placed in the hemibody by their own transforms, and the blending
    # weights of `deformation.segment_weights` on them (each segment's own transform
    # inside it, smooth blending in the gaps)
    placed = np.zeros(shape, np.uint32)
    for index, transform in enumerate(fit['transforms'], start=1):
        mask = rs.warp_labels((template_segments == index).astype(np.uint8), transform) > 0
        placed[mask & (placed == 0)] = index
    weights, _, global_weight = dfm.segment_weights(placed, len(fit['transforms']))
    (labels,) = dfm.warp_blended([label_map], fit['transforms'] + [fit['global_fit']],
                                 weights + [global_weight], shape, order=order)
    return np.where(field_of_view, labels, 0).astype(label_map.dtype if order == 0
                                                      else np.float32)


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
        Folder with the master (`master_segments.tif` and `.json`, saved by
        `build_master_mean.py`), and `master_mean.tif`, brought back too if present.
    out_folder : str | Path or None
        Output folder. None: `{recording}/analysis/registration_to_{master_path.name}`.
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
    out_folder = Path(out_folder or recording / 'analysis' / f'registration_to_{master_path.name}')
    hemibody_folder = out_folder / 'hemibodies'
    name = f'{recording.parent.name}_{recording.name}'

    # 1-2. hemibodies and their muscle clusters
    hemibodies.split_recording(recording, hemibody_folder, proj_type=PROJ_TYPE,
                               target_voxel_size=VOXEL_SIZE_UM)
    for side in ('L', 'R'):
        if not (hemibody_folder / f'{name}_{side}_components.tif').exists():
            muscle_mask.label_hemibody(hemibody_folder / f'{name}_{side}.ome.tif',
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
    template_segments, labels = dfm.pad(template_segments, shape), dfm.pad(labels, shape)
    mean_path = master_path / 'master_mean.tif'
    master_mean = (dfm.pad(tifffile.imread(mean_path).astype(np.float32), shape)
                   if mean_path.exists() else None)
    weights, near, global_weight = dfm.segment_weights(template_segments, len(segment_names))
    nearest_segment = np.argmax(np.stack(weights), axis=0) + 1  # for the per-segment overlap

    # 3-4. register each hemibody, bring it into the master frame, and the labels onto it
    report = {'master': str(master_path), 'segments': segment_names, 'hemibodies': {}}
    hemibody_labels, hemibody_means = {}, {}
    for side, hemibody_shape in zip(('L', 'R'), hemibody_shapes):
        start = time.time()
        image = dfm.pad(tifffile.imread(hemibody_folder / f'{name}_{side}.ome.tif')
                        .astype(np.float32), shape)
        components = dfm.pad(tifffile.imread(hemibody_folder / f'{name}_{side}_components.tif'),
                             shape)
        muscles = components > 0
        fit = dfm.register_hemibody(template_segments, segment_names, image, muscles, near,
                                    label=f'{name}_{side}')
        registered, valid, registered_muscles = dfm.warp_to_template(
            image, muscles, muscles, fit, weights, global_weight)
        crop = tuple(slice(0, s) for s in master_shape)
        tifffile.imwrite(out_folder / f'{name}_{side}_registered.tif', registered[crop])
        tifffile.imwrite(out_folder / f'{name}_{side}_registered_muscles.tif',
                         registered_muscles[crop].astype(np.uint32), compression='zlib')
        entry = dfm.add_segment_overlap(fit['entry'], fit, muscles, registered_muscles, valid,
                                        template_segments > 0, nearest_segment, segment_names)
        report['hemibodies'][side] = entry

        hemibody_crop = tuple(slice(0, s) for s in hemibody_shape)
        hemibody_labels[side] = master_to_hemibody(labels, template_segments, fit,
                                                   image > 0)[hemibody_crop]
        if master_mean is not None:
            hemibody_means[side] = master_to_hemibody(master_mean, template_segments, fit,
                                                      image > 0, order=1)[hemibody_crop]
        tifffile.imwrite(out_folder / f'{name}_{side}_labels.tif', hemibody_labels[side],
                         compression='zlib')
        print(f'{name}_{side}: muscle Dice with the master {entry["muscle_dice_all"]:.2f} '
              f'({time.time() - start:.0f}s)')
        (out_folder / 'registration.json').write_text(json.dumps(report, indent=1))
        dfm.write_segment_overlap(segment_names, {f'{name}_{s}': e for s, e in
                                                  report['hemibodies'].items()},
                                  out_folder / 'segment_overlap.csv')

    # labels on the original projection
    split_info = json.loads((hemibody_folder / f'{name}_split.json').read_text())
    with h5py.File(recording / 'analysis' / 'projection_images.h5', 'r') as f:
        projection = f[split_info['projection']][:]
    original_labels = hemibodies.hemibodies_to_original(
        hemibody_labels['L'], hemibody_labels['R'], split_info, projection.shape, order=0)
    labels_path = out_folder / 'master_labels.tif'
    tifffile.imwrite(labels_path, original_labels.astype(np.uint32), compression='zlib')
    present = [int(i) for i in np.unique(original_labels) if i > 0]
    (out_folder / 'master_labels.json').write_text(json.dumps(
        {'recording': str(recording), 'master': str(master_path),
         'label_map': str(label_map) if label_map else 'leg segments of the master',
         'array_axes': 'ZXY (as the projections in projection_images.h5)',
         'id_to_name': {i: id_to_name.get(i, str(i)) for i in present}}, indent=1))

    if master_mean is not None:
        original_mean = hemibodies.hemibodies_to_original(
            hemibody_means['L'], hemibody_means['R'], split_info, projection.shape, order=1)
        tifffile.imwrite(out_folder / 'master_mean_on_recording.tif',
                         original_mean.astype(np.float32))
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        for ax, axis in zip(axes, (0, 2)):
            rgb = np.zeros((*np.delete(projection.shape, axis), 3))
            for channel, volume in ((0, projection), (1, original_mean)):
                view = volume.max(axis=axis).astype(float)
                rgb[..., channel] = np.clip(view / max(np.percentile(view, 99.5), 1e-6), 0, 1)
            ax.imshow(rgb, aspect='auto')
            ax.set_title(f'{name}: {split_info["projection"]} (red) / master mean (green), max '
                         f'along axis {axis}', fontsize=8)
        fig.tight_layout()
        fig.savefig(out_folder / 'master_mean_on_recording.png', dpi=100)
        plt.close(fig)
        print(f'Saved {out_folder / "master_mean_on_recording.tif"}')

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
                        help='Folder with the master (master_segments.tif).')
    parser.add_argument('--out_folder', type=Path, default=None,
                        help='Output folder (default: '
                             '{recording}/analysis/registration_to_{master folder name}).')
    parser.add_argument('--label_map', type=Path, default=None,
                        help='Labels on the master grid (default: its leg segments).')
    parser.add_argument('--label_names', type=Path, default=None,
                        help='json of the label names.')
    args = parser.parse_args()
    register_recording(args.recording, args.master_path, args.out_folder, args.label_map,
                       args.label_names)
