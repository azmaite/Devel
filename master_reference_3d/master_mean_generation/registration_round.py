"""
Register several hemibodies to a template, leg segment by leg segment, and average them into a
master image (one registration round). Run once with the labelled reference hemibody as the
template (round 1), then again with the master of the previous round as the template (round 2,
...), so that the master is not biased towards one animal. The rounds are run by
`build_master_mean.py`; the registration itself is in `../registration/deformation.py`.

Inputs (from `../registration/hemibodies.py` and `../registration/muscle_mask.py`, in
`in_folder`): for every hemibody, `{name}_image.tif` and its automatic muscle mask, labelled by
muscle cluster, `{name}_components.tif`. Each template segment is fitted to the whole muscle mask
of the target (all its clusters, the head included): the targets need no segment labels.

The template is a muscle mask labelled by leg segment (`template_segments`):
- round 1: the reference hemibody's muscles, labelled by leg segment (see
  `build_master_mean.reference_segments`); the reference is also part of the average, as is,
- later rounds: the consensus of the registered muscle masks of the previous round
  (`template_from_round`); all hemibodies are registered to it.
The grid is the template's, padded at the high end of each axis to fit the largest hemibody.

Each hemibody is registered to the template (`deformation.register_hemibody`) and brought into
the template grid (`deformation.warp_to_template`), normalised to 0 at its background and 1 at
its muscles. The master is the mean of the normalised registered images over the hemibodies that
have data at each voxel (zero padding and absent segments excluded); the standard deviation and
the count are saved too.

Outputs, all on the template grid, as plain tifs that open together in napari at scale 1
(1.5 um voxels), in `out_folder`:
- `reference_segments.tif`: the template muscles labelled by leg segment (uint32; the names are
  in `reference_segments.json`),
- `{name}_registered.tif`: normalised registered image of each hemibody (float32, 0 where it has
  no data),
- `{name}_registered_muscles.tif`: its registered muscle mask (uint32, 0/1),
- `master_mean.tif`, `master_std.tif`, `master_count.tif`,
- `registration.json`: transforms (target -> template physical coordinates, um, SimpleITK x, y, z
  order), the checks of each segment fit, and the overlap of the registered muscles with the
  template muscles near each segment,
- `segment_overlap.csv`: that overlap as a table (one row per hemibody, one column per segment;
  'absent' where a segment was taken as absent), see `deformation.write_segment_overlap`,
- `registration.png`: projections of the master and of each registered hemibody.

registration_round: run one round.
template_from_round: master muscle mask of a finished round, labelled by leg segment.
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import tifffile

sys.path.insert(0, str(Path(__file__).parents[1] / 'registration'))
import deformation as dfm  # noqa: E402

VOXEL_SIZE_UM = dfm.VOXEL_SIZE_UM
TEMPLATE_SEGMENT_DISTANCE_UM = 10.0  # max distance of a voxel to the segment it is labelled with


def template_from_round(folder: Path) -> tuple:
    """
    Master muscle mask of a finished round, labelled by leg segment: the consensus of its
    registered muscle masks (muscle where at least half of the hemibodies that cover a voxel have
    muscle), each voxel labelled with the segment of the round's template within
    `TEMPLATE_SEGMENT_DISTANCE_UM` (else unlabelled, e.g. the head). It is the template of the next
    round, and what new recordings are registered to (`register_recording.py`).

    Parameters
    ----------
    folder : Path
        Output folder of the round.

    Returns
    -------
    tuple
        (segment map, uint32, on the round's grid; list of segment names, in label order).
    """
    previous = tifffile.imread(folder / 'reference_segments.tif')
    names = json.loads((folder / 'reference_segments.json').read_text())['id_to_name']
    segment_names = [names[str(i)] for i in range(1, len(names) + 1)]
    count = tifffile.imread(folder / 'master_count.tif')
    votes = np.zeros(previous.shape, np.float32)
    for path in sorted(folder.glob('*_registered_muscles.tif')):
        votes += tifffile.imread(path) > 0
    consensus = (count > 0) & (votes >= 0.5 * count)
    distance, indices = ndi.distance_transform_edt(previous == 0, sampling=VOXEL_SIZE_UM,
                                                   return_indices=True)
    nearest = previous[tuple(indices)]
    segments = np.where(consensus & (distance <= TEMPLATE_SEGMENT_DISTANCE_UM), nearest, 0)
    return segments.astype(np.uint32), segment_names


def _save_master(out_folder: Path,
                 registered: dict,
                 valid: dict,
                 template_muscles: np.ndarray) -> None:
    """
    Average the registered hemibodies into the master (mean, std, count) and save it, with the
    overview png (see the module docstring).

    Parameters
    ----------
    out_folder : Path
        Output folder.
    registered : dict
        Hemibody name -> normalised registered image.
    valid : dict
        Hemibody name -> boolean mask of where it has data.
    template_muscles : np.ndarray
        Template muscle mask (drawn on the png).

    Returns
    -------
    None
    """
    shape = template_muscles.shape
    count = np.zeros(shape, np.float32)
    total = np.zeros(shape, np.float64)
    total_sq = np.zeros(shape, np.float64)
    for name in registered:
        values = np.where(valid[name], registered[name], 0)
        count += valid[name]
        total += values
        total_sq += values ** 2
    mean = np.where(count > 0, total / np.maximum(count, 1), 0)
    std = np.sqrt(np.maximum(np.where(count > 0, total_sq / np.maximum(count, 1), 0) - mean ** 2,
                             0))
    tifffile.imwrite(out_folder / 'master_mean.tif', mean.astype(np.float32))
    tifffile.imwrite(out_folder / 'master_std.tif', std.astype(np.float32))
    tifffile.imwrite(out_folder / 'master_count.tif', count)

    names = list(registered)
    n_columns = (len(names) + 2) // 2
    fig, axes = plt.subplots(2, n_columns, figsize=(4 * n_columns, 12))
    for ax in axes.ravel():
        ax.axis('off')
    for ax, (title, volume) in zip(axes.ravel(), [('master mean', mean)]
                                   + [(name, registered[name]) for name in names]):
        ax.imshow(volume.max(axis=0), cmap='gray', vmin=0, vmax=np.percentile(volume, 99.9))
        ax.contour(template_muscles.max(axis=0), [0.5], colors='c', linewidths=0.4)
        ax.set_title(title, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_folder / 'registration.png', dpi=100)
    plt.close(fig)
    print(f'Saved the master and the registered hemibodies to {out_folder}')


def registration_round(in_folder: Path,
                       out_folder: Path,
                       template_segments: np.ndarray,
                       segment_names: list,
                       template_name: str,
                       targets: list,
                       reference: Optional[str] = None) -> None:
    """
    Register hemibodies to a template and average them (one round, see the module docstring).

    Parameters
    ----------
    in_folder : Path
        Folder with `{name}_image.tif` and `{name}_components.tif` of every hemibody.
    out_folder : Path
        Output folder (created if needed).
    template_segments : np.ndarray
        Template muscle mask labelled by leg segment (1.. in the order of `segment_names`).
    segment_names : list of str
        Names of the template segments.
    template_name : str
        Description of the template, for the json outputs.
    targets : list of str
        Hemibodies (`{recording}_{side}`) registered to the template.
    reference : str or None
        Round 1: the reference hemibody, whose muscles are the template; it is not registered,
        and enters the average as is. None for later rounds.

    Returns
    -------
    None
    """
    out_folder.mkdir(parents=True, exist_ok=True)

    def load(name: str) -> tuple:
        return (dfm.pad(tifffile.imread(in_folder / f'{name}_image.tif').astype(np.float32), shape),
                dfm.pad(tifffile.imread(in_folder / f'{name}_components.tif'), shape))

    # the template's grid, padded at the high end of each axis to fit the largest hemibody
    names = targets + ([reference] if reference else [])
    shapes = [template_segments.shape] + [
        tifffile.TiffFile(in_folder / f'{name}_image.tif').series[0].shape for name in names]
    shape = tuple(int(s) for s in np.max(shapes, axis=0))
    template_segments = dfm.pad(template_segments, shape)
    template_muscles = template_segments > 0
    print(f'Template: {template_name}; grid {shape}')
    tifffile.imwrite(out_folder / 'reference_segments.tif', template_segments,
                     compression='zlib')
    (out_folder / 'reference_segments.json').write_text(json.dumps(
        {'template': template_name, 'grid_shape': list(shape), 'voxel_size_um': VOXEL_SIZE_UM,
         'id_to_name': {i: name for i, name in enumerate(segment_names, start=1)}}, indent=1))

    registered, valid = {}, {}
    if reference:
        image, components = load(reference)
        registered[reference] = dfm.normalise(image, components > 0)
        valid[reference] = image > 0
        tifffile.imwrite(out_folder / f'{reference}_registered.tif', registered[reference])
        tifffile.imwrite(out_folder / f'{reference}_registered_muscles.tif',
                         template_muscles.astype(np.uint32), compression='zlib')

    print('Computing blending weights...')
    weights, near, global_weight = dfm.segment_weights(template_segments, len(segment_names))
    nearest_segment = np.argmax(np.stack(weights), axis=0) + 1  # for the per-segment overlap

    report = {'template': template_name, 'segments': segment_names, 'targets': {}}
    for name in targets:
        start = time.time()
        image, components = load(name)
        muscles = components > 0
        fit = dfm.register_hemibody(template_segments, segment_names, image, muscles, near, name)
        registered[name], valid[name], registered_muscles = dfm.warp_to_template(
            image, muscles, muscles, fit, weights, global_weight)
        tifffile.imwrite(out_folder / f'{name}_registered.tif', registered[name])
        tifffile.imwrite(out_folder / f'{name}_registered_muscles.tif',
                         registered_muscles.astype(np.uint32), compression='zlib')

        entry = dfm.add_segment_overlap(fit['entry'], fit, muscles, registered_muscles,
                                        valid[name], template_muscles, nearest_segment,
                                        segment_names)
        report['targets'][name] = entry
        print(f'{name}: muscle Dice {entry["muscle_dice_centroid_only"]:.2f} (centroids) -> '
              f'{entry["muscle_dice_all"]:.2f} ({time.time() - start:.0f}s)')
        (out_folder / 'registration.json').write_text(json.dumps(report, indent=1))
        dfm.write_segment_overlap(segment_names, report['targets'],
                                  out_folder / 'segment_overlap.csv')

    _save_master(out_folder, registered, valid, template_muscles)
