"""
Register several hemibodies to a template, leg segment by leg segment, and average them into a
master image. Run once with a labelled hemibody as the template (round 1), then again with the
master of the previous round as the template (round 2, ...), so that the master is not biased
towards one animal.

Inputs (from `split_hemibodies.py` and `label_mask_components.py`, in
`master_reference/segment_masks`): for every hemibody, `{name}_image.tif` and the labelled
automatic muscle mask `{name}_components.tif`. The leg segment of each label is given by hand in
`{recording}_segment_ids.json` files (`segments`, then per side, 'L' and/or 'R', a dict segment
name -> list of labels). Only the round-1 template needs one: each template segment is fitted to
the whole muscle mask of the target (all its components, the head included). The segment tables
of the targets are ignored unless `USE_TARGET_SEGMENTS` is set (then each template segment is
fitted only to the same segment of a labelled target, a segment the target does not list is taken
as absent, and the target's head is removed from its mask).

The template is a muscle mask labelled by leg segment, on the output grid:
- round 1: the reference hemibody's muscles, labelled with its segment table (the grid is the
  reference, padded at the high end of each axis to fit the largest hemibody); the reference is
  also part of the average, as is,
- later rounds (`template_folder`, the output folder of the previous round): the consensus of the
  registered muscle masks of the previous round (muscle where at least half of the hemibodies
  that cover a voxel have muscle), each voxel labelled with the segment of the previous template
  within `TEMPLATE_SEGMENT_DISTANCE_UM` (else unlabelled, e.g. the head); all hemibodies are
  registered to it.
Segments are registered one at a time, each as one piece (all its labels together). The
segments in `IGNORED_SEGMENTS` (the head) get no fit of their own: their muscles are removed from
the template, and the global fit carries them along (they are in the images, roughly aligned, but
not fitted).

Each hemibody (the target) is registered to the template by the shape of the muscles (see
`register_segments.py`), with the target's field of view: a segment that is cut off in the target
is matched by its visible part, and is neither pushed back inside nor lets escape out of view:
1. a translation between the centroids of all the muscles of the two,
2. a global affine fit of all the template muscles to all the target muscles, from 1,
3. one fit per leg segment, from 2: affine for large segments, rigid for small ones (fewer than
   `MIN_AFFINE_VOXELS` muscle voxels).
A segment is taken as absent from the target, and the target gives no data near it, if its fit
moves it more than `MAX_SEGMENT_SHIFT_UM` away from
where the global fit puts it, or if less than `MIN_SEGMENT_IN_VIEW` of the fitted segment is in
the target's field of view, or less than `MIN_SEGMENT_ON_MUSCLE` of that part lies on target
muscle (e.g. a hind tibia that is not in the image).

The target is then brought into the template grid with one smooth deformation: for each voxel,
the positions that the segment transforms (and the global one) map it to are averaged, with
weights that fall off with the distance to each template segment (Gaussian, `BLEND_SIGMA_UM`;
the global fit takes over far from all segments), and the target is sampled once there. (Blending
the transformed images instead doubles the edges wherever the transforms differ.) Before
averaging, each image is normalised to 0 at its background (median of the valid voxels outside
the muscle mask) and 1 at its muscles (median inside). The master is the mean of the normalised
registered images over the hemibodies that have data at each voxel (zero padding and absent
segments excluded); the standard deviation and the count are saved too.

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
- `registration.png`: projections of the master and of each registered hemibody.

Usually run through `build_master_reference.py`, which runs all the rounds. On its own:
`python registration_round.py` (round 1, settings at the top of the script), then
`python registration_round.py --template_folder registered_hemibodies --out_folder
registered_hemibodies_round2` (see `--help`).

registration_round: run one round.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk
import tifffile

import register_segments as rs

MASTER_PATH = Path('/mnt/labserver/data/MA/SCAPE/master_reference')
REFERENCE = '20261021_V7xPR_pupa_long_paired_early_sample1_run1_R'
HEMIBODIES = ['20261021_V7xPR_pupa_long_paired_early_sample1_run1_L',
              '20261016_V7xPR_pupa_long_paired_sample1_run3_short_L',
              '20261016_V7xPR_pupa_long_paired_sample1_run3_short_R',
              '20261021_V7xPR_pupa_long_scape_sample1_run01_short_20261021_1638_L',
              '20261021_V7xPR_pupa_long_scape_sample1_run01_short_20261021_1638_R']
VOXEL_SIZE_UM = 1.5
IGNORED_SEGMENTS = ['head']  # no fit of their own (their muscles are removed from the template)
USE_TARGET_SEGMENTS = False  # fit each segment to the same segment of labelled targets only
MIN_AFFINE_VOXELS = 30000  # smaller segments are fitted rigidly (affine is unstable on them)
MAX_SEGMENT_SHIFT_UM = 30.0  # max distance between segment and global fit at the segment centre
MIN_SEGMENT_IN_VIEW = 0.2  # min fraction of the fitted segment in the target's field of view
MIN_SEGMENT_ON_MUSCLE = 0.3  # min fraction of that part on target muscle
BLEND_SIGMA_UM = 10.0  # fall-off of the segment weights with the distance to the segment
GLOBAL_DISTANCE_UM = 30.0  # the global fit weighs as much as a segment this far away
ABSENT_DISTANCE_UM = 20.0  # a target gives no data this close to a segment it does not have
TEMPLATE_SEGMENT_DISTANCE_UM = 10.0  # later rounds: max distance to a previous template segment
CHUNK_PLANES = 16  # planes of the output grid deformed at a time (memory)


def _pad(array: np.ndarray,
         shape: tuple) -> np.ndarray:
    """
    Zero-pad an array at the high end of each axis to `shape`.

    Parameters
    ----------
    array : np.ndarray
        Array.
    shape : tuple of int
        Output shape (at least `array.shape`).

    Returns
    -------
    np.ndarray
        Padded array.
    """
    return np.pad(array, [(0, s - n) for s, n in zip(shape, array.shape)])


def _physical_centroid(mask: np.ndarray) -> np.ndarray:
    """
    Centroid of a mask in physical SimpleITK coordinates (x, y, z, in um).

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask, array order (z, y, x).

    Returns
    -------
    np.ndarray
        Centroid (x, y, z), in um.
    """
    return np.array(ndi.center_of_mass(mask))[::-1] * VOXEL_SIZE_UM


def _matrix_offset(transform: sitk.Transform) -> tuple:
    """
    Matrix and offset of an affine-like transform (centre at the origin).

    Parameters
    ----------
    transform : sitk.Transform
        Affine-like transform.

    Returns
    -------
    tuple of np.ndarray
        (3 x 3 matrix, offset), acting on physical (x, y, z) points.
    """
    affine = rs.to_affine(transform)
    return np.array(affine.GetMatrix()).reshape(3, 3), np.array(affine.GetTranslation())


def _warp_blended(arrays: list,
                  transforms: list,
                  weights: list,
                  shape: tuple) -> list:
    """
    Bring target arrays into the template grid with a blend of affine transforms.

    For each template voxel, the target positions given by each transform (template -> target)
    are averaged with the voxel's weights, and the arrays are sampled there (linear
    interpolation, 0 outside the target), so each array is sampled once.

    Parameters
    ----------
    arrays : list of np.ndarray
        Arrays on the (padded) target grid, all of shape `shape`.
    transforms : list of sitk.Transform
        Template -> target physical coordinates, one per weight.
    weights : list
        Weight of each transform on the template grid: arrays of shape `shape` or scalars.
    shape : tuple of int
        Shape of the template grid (and of the arrays).

    Returns
    -------
    list of np.ndarray
        Float32 warped arrays, shape `shape`.
    """
    matrices = [_matrix_offset(transform) for transform in transforms]
    outputs = [np.zeros(shape, np.float32) for _ in arrays]
    y, x = np.meshgrid(np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    for start in range(0, shape[0], CHUNK_PLANES):
        stop = min(start + CHUNK_PLANES, shape[0])
        z = np.arange(start, stop)[:, None, None]
        # physical (x, y, z) of the template voxels of this chunk
        points = np.stack(np.broadcast_arrays(x[None] * VOXEL_SIZE_UM, y[None] * VOXEL_SIZE_UM,
                                              z * VOXEL_SIZE_UM), axis=-1)
        position = np.zeros(points.shape, np.float64)
        total = np.zeros(points.shape[:-1], np.float64)
        for (matrix, offset), weight in zip(matrices, weights):
            w = weight[start:stop] if isinstance(weight, np.ndarray) else weight
            position += np.asarray(w)[..., None] * (points @ matrix.T + offset)
            total += w
        position /= total[..., None]
        # target array indices (z, y, x)
        coordinates = np.moveaxis(position[..., ::-1] / VOXEL_SIZE_UM, -1, 0)
        for array, output in zip(arrays, outputs):
            output[start:stop] = ndi.map_coordinates(array, coordinates, order=1, mode='constant',
                                                     cval=0.0)
    return outputs


def _normalise(image: np.ndarray,
               muscles: np.ndarray) -> np.ndarray:
    """
    Scale an image to 0 at its background and 1 at its muscles (medians of the valid voxels
    outside and inside `muscles`).

    Parameters
    ----------
    image : np.ndarray
        Image (0 = padding).
    muscles : np.ndarray
        Boolean muscle mask.

    Returns
    -------
    np.ndarray
        Float32 normalised image (0 on the padding; validity is tracked separately).
    """
    valid = image > 0
    background = np.median(image[valid & ~muscles])
    foreground = np.median(image[muscles])
    return np.where(valid, (image - background) / (foreground - background), 0).astype(np.float32)


def _affine_dict(transform: sitk.Transform) -> dict:
    """
    Matrix and translation of an affine-like transform, for json.

    Parameters
    ----------
    transform : sitk.Transform
        Affine-like transform.

    Returns
    -------
    dict
        `matrix` (3 x 3, row-major) and `translation` (x, y, z, um), centre at the origin.
    """
    matrix, offset = _matrix_offset(transform)
    return {'matrix': matrix.round(6).tolist(), 'translation': offset.round(4).tolist()}


def _dice(a: np.ndarray,
          b: np.ndarray) -> float:
    """
    Dice coefficient of two boolean masks.

    Parameters
    ----------
    a, b : np.ndarray
        Boolean masks.

    Returns
    -------
    float
        Dice (0 if both are empty).
    """
    return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


def _load_segment_tables(folder: Path) -> tuple:
    """
    Read all `{recording}_segment_ids.json` files of a folder.

    Parameters
    ----------
    folder : Path
        Folder with the json files.

    Returns
    -------
    tuple
        (dict hemibody name -> {segment name: list of labels}, list of all segment names in
        order of first appearance).
    """
    tables, names = {}, []
    for path in sorted(folder.glob('*_segment_ids.json')):
        content = json.loads(path.read_text())
        recording = path.name.removesuffix('_segment_ids.json')
        names += [s for s in content['segments'] if s not in names]
        for side in ('L', 'R'):
            if side in content:
                tables[f'{recording}_{side}'] = content[side]
    return tables, names


def _template_from_previous_round(folder: Path) -> tuple:
    """
    Template of a later round: consensus muscle mask of a previous round, labelled by the
    previous template's segments (see the module docstring).

    Parameters
    ----------
    folder : Path
        Output folder of the previous round.

    Returns
    -------
    tuple
        (segment map, uint32; list of segment names).
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


def registration_round(master_path: Path = MASTER_PATH,
                       reference: str = REFERENCE,
                       hemibodies: list = HEMIBODIES,
                       template_folder: Optional[Path] = None,
                       out_folder: Optional[Path] = None) -> None:
    """
    Register hemibodies to a template and average them (one round, see the module docstring).

    Parameters
    ----------
    master_path : Path
        Folder holding `segment_masks` (inputs).
    reference : str
        Round 1: name of the reference hemibody (`{recording}_{side}`), the template; it needs a
        segment table. Later rounds: registered like the others.
    hemibodies : list of str
        The other hemibodies.
    template_folder : Path or None
        None: round 1. Otherwise the output folder of the previous round, whose consensus muscle
        mask is the template (relative paths are taken in `master_path`).
    out_folder : Path or None
        Output folder (relative paths are taken in `master_path`). None:
        `master_path / 'registered_hemibodies'`.

    Returns
    -------
    None
    """
    in_folder = master_path / 'segment_masks'
    out_folder = master_path / (out_folder or 'registered_hemibodies')
    out_folder.mkdir(parents=True, exist_ok=True)
    tables, _ = _load_segment_tables(in_folder)

    def load(name: str) -> tuple:
        return (tifffile.imread(in_folder / f'{name}_image.tif').astype(np.float32),
                tifffile.imread(in_folder / f'{name}_components.tif'))

    registered, valid = {}, {}
    if template_folder is None:
        segment_names = [s for s in tables[reference] if s not in IGNORED_SEGMENTS]
        shapes = [tifffile.TiffFile(in_folder / f'{name}_image.tif').series[0].shape
                  for name in [reference] + hemibodies]
        shape = tuple(int(s) for s in np.max(shapes, axis=0))
        targets = list(hemibodies)
        template_name = reference
    else:
        template_folder = master_path / template_folder
        template_segments, segment_names = _template_from_previous_round(template_folder)
        shape = template_segments.shape
        targets = [reference] + list(hemibodies)
        template_name = f'consensus of {template_folder.name}'
    print(f'Template: {template_name}; grid {shape}')

    def segment_map(components: np.ndarray, table: dict) -> np.ndarray:
        # segment index (1..) of each voxel, in the order of `segment_names`; ignored and
        # unlisted labels -> 0
        lookup = np.zeros(int(components.max()) + 1, dtype=np.uint32)
        for index, segment in enumerate(segment_names, start=1):
            labels = [label for label in table.get(segment, []) if label < lookup.size]
            lookup[labels] = index
        return lookup[components]

    def known_muscles(components: np.ndarray, table: dict) -> np.ndarray:
        # all muscles except those of the ignored segments
        ignored = sum((table.get(s, []) for s in IGNORED_SEGMENTS), [])
        return (components > 0) & ~np.isin(components, ignored)

    if template_folder is None:
        reference_image, reference_components = map(lambda a: _pad(a, shape), load(reference))
        template_segments = segment_map(reference_components, tables[reference])
        registered[reference] = _normalise(reference_image, reference_components > 0)
        valid[reference] = reference_image > 0
    template_muscles = template_segments > 0
    tifffile.imwrite(out_folder / 'reference_segments.tif', template_segments,
                     compression='zlib')
    (out_folder / 'reference_segments.json').write_text(json.dumps(
        {'template': template_name, 'grid_shape': list(shape), 'voxel_size_um': VOXEL_SIZE_UM,
         'id_to_name': {i: name for i, name in enumerate(segment_names, start=1)}}, indent=1))

    # blending weights of each segment on the template grid (Gaussian of the distance to it)
    print('Computing blending weights...')
    weights, near = [], []
    for index in range(1, len(segment_names) + 1):
        distance = ndi.distance_transform_edt(template_segments != index, sampling=VOXEL_SIZE_UM)
        weights.append(np.exp(-distance ** 2 / (2 * BLEND_SIGMA_UM ** 2)).astype(np.float32))
        near.append(distance <= ABSENT_DISTANCE_UM)
    global_weight = float(np.exp(-GLOBAL_DISTANCE_UM ** 2 / (2 * BLEND_SIGMA_UM ** 2)))
    nearest_segment = np.argmax(np.stack(weights), axis=0) + 1  # for the per-segment overlap

    report = {'template': template_name, 'segments': segment_names, 'targets': {}}
    for name in targets:
        start = time.time()
        image, components = map(lambda a: _pad(a, shape), load(name))
        field_of_view = image > 0
        table = tables.get(name) if USE_TARGET_SEGMENTS else None
        target_segments = segment_map(components, table) if table is not None else None
        muscles = known_muscles(components, table or {})

        shift = _physical_centroid(template_muscles) - _physical_centroid(muscles)
        initial = sitk.TranslationTransform(3, shift.tolist())
        global_fit = rs.register_segment_shape(template_muscles, muscles, model='affine',
                                               initial=initial, clip_um=45.0,
                                               target_valid=field_of_view)
        entry = {'labelled_segments': table is not None, 'global': _affine_dict(global_fit),
                 'segments': {}}
        print(f'{name}: global fit done ({time.time() - start:.0f}s)')

        transforms, absent = [], np.zeros(shape, bool)
        for index, segment in enumerate(segment_names, start=1):
            segment_muscles = template_segments == index
            info = {}
            if table is not None and not table.get(segment):
                fit, info['absent'] = global_fit, 'not in the segment table'
            else:
                target_mask = (target_segments == index) if table is not None else muscles
                model = 'affine' if segment_muscles.sum() >= MIN_AFFINE_VOXELS else 'rigid'
                fit = rs.register_segment_shape(segment_muscles, target_mask, model=model,
                                                initial=global_fit, target_valid=field_of_view)
                centre = tuple(_physical_centroid(segment_muscles))
                shift_um = float(np.linalg.norm(np.subtract(
                    rs.invert_affine(fit).TransformPoint(centre),
                    rs.invert_affine(global_fit).TransformPoint(centre))))
                # where the fitted segment lands in the target (on the target grid)
                landed = rs.warp_labels(segment_muscles.astype(np.uint8), fit) > 0
                in_view = float((landed & field_of_view).sum() / max(landed.sum(), 1))
                on_muscle = float((landed & field_of_view & target_mask).sum()
                                  / max((landed & field_of_view).sum(), 1))
                info.update({'model': model, 'shift_from_global_um': shift_um,
                             'fraction_in_view': in_view, 'fraction_on_muscle': on_muscle})
                if shift_um > MAX_SEGMENT_SHIFT_UM:
                    info['absent'] = 'fit ran away from the global fit'
                elif in_view < MIN_SEGMENT_IN_VIEW:
                    info['absent'] = 'out of the field of view'
                elif on_muscle < MIN_SEGMENT_ON_MUSCLE:
                    info['absent'] = 'not on target muscle'
                if 'absent' in info:
                    fit = global_fit
            if 'absent' in info:
                absent |= near[index - 1]
            transforms.append(fit)
            entry['segments'][segment] = {**info, **_affine_dict(fit)}
            print(f'  {segment}: ' + (f'{info["model"]}, {info["shift_from_global_um"]:.1f} um '
                                      f'from global, {info["fraction_in_view"]:.2f} in view, '
                                      f'{info["fraction_on_muscle"]:.2f} on muscle'
                                      if 'model' in info else '')
                  + (f' -> ABSENT ({info["absent"]})' if 'absent' in info else ''))

        # bring the target into the template grid with one blended deformation
        warped_image, warped_view, warped_muscles = _warp_blended(
            [_normalise(image, components > 0), field_of_view.astype(np.float32),
             muscles.astype(np.float32)],
            [rs.invert_affine(t) for t in transforms + [global_fit]],
            weights + [global_weight], shape)
        valid_target = (warped_view > 0.5) & ~absent
        registered[name] = np.where(valid_target, warped_image / np.maximum(warped_view, 1e-6), 0)
        valid[name] = valid_target
        registered_muscles = (warped_muscles > 0.5) & valid_target
        tifffile.imwrite(out_folder / f'{name}_registered.tif',
                         registered[name].astype(np.float32))
        tifffile.imwrite(out_folder / f'{name}_registered_muscles.tif',
                         registered_muscles.astype(np.uint32), compression='zlib')

        # overlap with the template muscles, in the region closest to each segment
        entry['muscle_dice_all'] = _dice(registered_muscles, template_muscles & valid_target)
        for index, segment in enumerate(segment_names, start=1):
            region = (nearest_segment == index) & valid_target
            entry['segments'][segment]['muscle_dice_near_segment'] = _dice(
                registered_muscles & region, template_muscles & region)
        before = rs.warp_labels(muscles.astype(np.uint8), rs.invert_affine(initial)) > 0
        entry['muscle_dice_centroid_only'] = _dice(before, template_muscles)
        report['targets'][name] = entry
        print(f'{name}: muscle Dice {entry["muscle_dice_centroid_only"]:.2f} (centroids) -> '
              f'{entry["muscle_dice_all"]:.2f} ({time.time() - start:.0f}s)')
        (out_folder / 'registration.json').write_text(json.dumps(report, indent=1))

    if template_folder is None:
        tifffile.imwrite(out_folder / f'{reference}_registered.tif',
                         registered[reference].astype(np.float32))
        tifffile.imwrite(out_folder / f'{reference}_registered_muscles.tif',
                         template_muscles.astype(np.uint32), compression='zlib')

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


if __name__ == '__main__':
    matplotlib.use('Agg')
    parser = argparse.ArgumentParser(description='Register hemibodies segment by segment and '
                                                 'average them into a master image (one round).')
    parser.add_argument('--master_path', type=Path, default=MASTER_PATH)
    parser.add_argument('--reference', default=REFERENCE)
    parser.add_argument('--hemibodies', nargs='*', default=HEMIBODIES)
    parser.add_argument('--template_folder', type=Path, default=None,
                        help='Output folder of the previous round (default: round 1, with the '
                             'reference hemibody as the template).')
    parser.add_argument('--out_folder', type=Path, default=None,
                        help='Output folder (default: registered_hemibodies).')
    args = parser.parse_args()
    registration_round(args.master_path, args.reference, args.hemibodies,
                       args.template_folder, args.out_folder)
