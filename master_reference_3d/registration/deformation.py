"""
Register a hemibody (the target) to a template muscle mask labelled by leg segment, leg segment
by leg segment, and bring it into the template grid with one smooth deformation. Used to build
the master (`../master_mean_generation/registration_round.py`) and to register new recordings
to it (`../register_recording.py`).

Each hemibody is registered to the template by the shape of the muscles (see
`register_segments.py`), with the target's field of view: a segment that is cut off in the target
is matched by its visible part, and is neither pushed back inside nor lets escape out of view
(`register_hemibody`):
1. a translation between the centroids of all the muscles of the two,
2. a global affine fit of all the template muscles to all the target muscles, from 1,
3. one fit per leg segment, from 2, against the whole muscle mask of the target (no segment
   labels needed on the target): affine for large segments, rigid for small ones (fewer than
   `MIN_AFFINE_VOXELS` muscle voxels). The segments in `IGNORED_SEGMENTS` (the head) get no fit
   of their own: they are left out of the template, and the global fit carries them along.
A segment is taken as absent from the target, and the target gives no data near it, if its fit
moves it more than `MAX_SEGMENT_SHIFT_UM` away from where the global fit puts it, or if less than
`MIN_SEGMENT_IN_VIEW` of the fitted segment is in the target's field of view, or less than
`MIN_SEGMENT_ON_MUSCLE` of that part lies on target muscle (e.g. a hind tibia that is not in the
image).

The target is then brought into the template grid with one deformation (`warp_to_template`, with
the weights of `segment_weights`): inside each template segment (and within `OWN_MARGIN_UM` of
it) the segment's own transform, so that no segment is bent by its neighbours; between segments,
the positions that the nearby segment transforms (and the global one, which takes over far from
all segments) map each voxel to are averaged with inverse-square distance weights, so the
deformation goes smoothly from one segment's transform to the next across the gaps, and all the
bending happens there. The target is sampled once at these positions (`warp_blended`). (Blending
the transformed images instead doubles the edges wherever the transforms differ.) Where two
segments touch, the deformation jumps from one transform to the other. The image is normalised
to 0 at its background (median of the valid voxels outside the muscle mask) and 1 at its muscles
(median inside, `normalise`).

Transforms are SimpleITK affine-like transforms in physical coordinates (um, x, y, z order);
arrays are in (z, y, x) order, with `VOXEL_SIZE_UM` isotropic voxels.

register_hemibody: register one hemibody to a template (global, then per-segment fits).
warp_to_template: bring a registered hemibody into the template grid.
segment_weights: blending weights of the segments on a grid.
warp_blended: sample arrays through a weighted blend of affine transforms.
add_segment_overlap: overlap of a registered hemibody with the template, per segment.
write_segment_overlap: save the per-segment overlaps of several hemibodies as a csv.
pad, physical_centroid, normalise, dice, affine_dict, affine_from_dict: small helpers.
"""

import time
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk

import register_segments as rs

VOXEL_SIZE_UM = 1.5
IGNORED_SEGMENTS = ['head']  # no fit of their own (their muscles are left out of the template)
MIN_AFFINE_VOXELS = 30000  # smaller segments are fitted rigidly (affine is unstable on them)
MAX_SEGMENT_SHIFT_UM = 30.0  # max distance between segment and global fit at the segment centre
MIN_SEGMENT_IN_VIEW = 0.2  # min fraction of the fitted segment in the target's field of view
MIN_SEGMENT_ON_MUSCLE = 0.3  # min fraction of that part on target muscle
OWN_MARGIN_UM = 3.0  # voxels within this distance of a segment follow its transform only
GLOBAL_DISTANCE_UM = 30.0  # the global fit weighs as much as a segment this far (beyond its margin)
ABSENT_DISTANCE_UM = 20.0  # a target gives no data this close to a segment it does not have
CHUNK_PLANES = 16  # planes of the output grid deformed at a time (memory)


def pad(array: np.ndarray,
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


def physical_centroid(mask: np.ndarray) -> np.ndarray:
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


def normalise(image: np.ndarray,
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


def dice(a: np.ndarray,
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


def affine_dict(transform: sitk.Transform) -> dict:
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


def affine_from_dict(entry: dict) -> sitk.AffineTransform:
    """
    Affine transform from its `affine_dict` (as saved in `registration.json`).

    Parameters
    ----------
    entry : dict
        `matrix` and `translation`.

    Returns
    -------
    sitk.AffineTransform
        The transform.
    """
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(np.ravel(entry['matrix']).tolist())
    transform.SetTranslation(list(entry['translation']))
    return transform


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


def warp_blended(arrays: list,
                 transforms: list,
                 weights: list,
                 shape: tuple,
                 order: int = 1) -> list:
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
        Shape of the output grid.
    order : int
        Interpolation order (1: linear, for images; 0: nearest, for labels).

    Returns
    -------
    list of np.ndarray
        Warped arrays, shape `shape` (float32, or the arrays' dtype for `order` 0).
    """
    matrices = [_matrix_offset(transform) for transform in transforms]
    outputs = [np.zeros(shape, array.dtype if order == 0 else np.float32) for array in arrays]
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
            output[start:stop] = ndi.map_coordinates(array, coordinates, order=order,
                                                     mode='constant', cval=0)
    return outputs


def segment_weights(segments: np.ndarray,
                    n_segments: int) -> tuple:
    """
    Blending weights of each segment and of the global fit, and the region near each segment
    (within `ABSENT_DISTANCE_UM`).

    Each voxel within `OWN_MARGIN_UM` of a segment (the nearest one, if several) follows that
    segment's transform only (weight 1, all the others 0). Every other voxel blends all the
    segments with weights 1 / d^2, d being its distance to the segment's margin (so the weight of a
    segment grows to infinity at its margin and the deformation is continuous there), and the
    global fit with weight 1 / `GLOBAL_DISTANCE_UM`^2 (so it takes over far from all segments).

    Parameters
    ----------
    segments : np.ndarray
        Segment map (1..`n_segments`, 0 elsewhere).
    n_segments : int
        Number of segments.

    Returns
    -------
    tuple
        (list of float32 weight arrays, list of boolean 'near' arrays, float32 weight array of
        the global fit).
    """
    distance, indices = ndi.distance_transform_edt(segments == 0, sampling=VOXEL_SIZE_UM,
                                                   return_indices=True)
    owner = np.where(distance <= OWN_MARGIN_UM, segments[tuple(indices)], 0)
    del indices
    outside = owner == 0
    weights, near = [], []
    for index in range(1, n_segments + 1):
        distance = ndi.distance_transform_edt(segments != index, sampling=VOXEL_SIZE_UM)
        gap = np.maximum(distance - OWN_MARGIN_UM, 1e-3)
        weight = np.where(outside, 1.0 / gap ** 2, (owner == index).astype(np.float32))
        weights.append(weight.astype(np.float32))
        near.append(distance <= ABSENT_DISTANCE_UM)
    global_weight = np.where(outside, 1.0 / GLOBAL_DISTANCE_UM ** 2, 0).astype(np.float32)
    return weights, near, global_weight


def register_hemibody(template_segments: np.ndarray,
                      segment_names: list,
                      image: np.ndarray,
                      muscles: np.ndarray,
                      near: list,
                      label: str = '') -> dict:
    """
    Register one hemibody (the target) to the template (steps 1-3 of the module docstring).

    Parameters
    ----------
    template_segments : np.ndarray
        Template segment map (1.. in the order of `segment_names`).
    segment_names : list of str
        Names of the template segments.
    image : np.ndarray
        Target image, same shape as `template_segments` (padded; 0 = outside the field of view).
    muscles : np.ndarray
        Target muscle mask, same shape.
    near : list of np.ndarray
        Region near each template segment (see `segment_weights`).
    label : str
        Name printed in the progress messages.

    Returns
    -------
    dict
        `global_fit` and `transforms` (one per segment; target -> template physical
        coordinates, the global fit for absent segments), `absent` (boolean mask, on the
        template grid, of where the target gives no data), `initial` (centroid translation) and
        `entry` (the fit checks, for json).
    """
    start = time.time()
    template_muscles = template_segments > 0
    field_of_view = image > 0
    shift = physical_centroid(template_muscles) - physical_centroid(muscles)
    initial = sitk.TranslationTransform(3, shift.tolist())
    global_fit = rs.register_segment_shape(template_muscles, muscles, model='affine',
                                           initial=initial, clip_um=45.0,
                                           target_valid=field_of_view)
    entry = {'global': affine_dict(global_fit), 'segments': {}}
    print(f'{label}: global fit done ({time.time() - start:.0f}s)')

    transforms, absent = [], np.zeros(template_segments.shape, bool)
    for index, segment in enumerate(segment_names, start=1):
        segment_muscles = template_segments == index
        model = 'affine' if segment_muscles.sum() >= MIN_AFFINE_VOXELS else 'rigid'
        fit = rs.register_segment_shape(segment_muscles, muscles, model=model,
                                        initial=global_fit, target_valid=field_of_view)
        centre = tuple(physical_centroid(segment_muscles))
        shift_um = float(np.linalg.norm(np.subtract(
            rs.invert_affine(fit).TransformPoint(centre),
            rs.invert_affine(global_fit).TransformPoint(centre))))
        # where the fitted segment lands in the target (on the target grid)
        landed = rs.warp_labels(segment_muscles.astype(np.uint8), fit) > 0
        in_view = float((landed & field_of_view).sum() / max(landed.sum(), 1))
        on_muscle = float((landed & field_of_view & muscles).sum()
                          / max((landed & field_of_view).sum(), 1))
        info = {'model': model, 'shift_from_global_um': shift_um,
                'fraction_in_view': in_view, 'fraction_on_muscle': on_muscle}
        if shift_um > MAX_SEGMENT_SHIFT_UM:
            info['absent'] = 'fit ran away from the global fit'
        elif in_view < MIN_SEGMENT_IN_VIEW:
            info['absent'] = 'out of the field of view'
        elif on_muscle < MIN_SEGMENT_ON_MUSCLE:
            info['absent'] = 'not on target muscle'
        if 'absent' in info:
            fit = global_fit
            absent |= near[index - 1]
        transforms.append(fit)
        entry['segments'][segment] = {**info, **affine_dict(fit)}
        print(f'  {segment}: {model}, {shift_um:.1f} um from global, {in_view:.2f} in view, '
              f'{on_muscle:.2f} on muscle'
              + (f' -> ABSENT ({info["absent"]})' if 'absent' in info else ''))
    return {'global_fit': global_fit, 'transforms': transforms, 'absent': absent,
            'initial': initial, 'entry': entry}


def warp_to_template(image: np.ndarray,
                     muscles: np.ndarray,
                     all_muscles: np.ndarray,
                     fit: dict,
                     weights: list,
                     global_weight: float) -> tuple:
    """
    Bring a registered hemibody into the template grid with one blended deformation (step 4 of
    the module docstring).

    Parameters
    ----------
    image : np.ndarray
        Target image (padded to the template grid; 0 = outside the field of view).
    muscles : np.ndarray
        Target muscle mask used for the fits.
    all_muscles : np.ndarray
        Target muscle mask used for the normalisation (all clusters).
    fit : dict
        Output of `register_hemibody`.
    weights : list
        Segment weights on the template grid (see `segment_weights`).
    global_weight : float or np.ndarray
        Weight of the global fit (see `segment_weights`).

    Returns
    -------
    tuple of np.ndarray
        (normalised registered image, 0 where there is no data; boolean mask of where there is
        data; registered muscle mask).
    """
    shape = image.shape
    warped_image, warped_view, warped_muscles = warp_blended(
        [normalise(image, all_muscles), (image > 0).astype(np.float32),
         muscles.astype(np.float32)],
        [rs.invert_affine(t) for t in fit['transforms'] + [fit['global_fit']]],
        weights + [global_weight], shape)
    valid = (warped_view > 0.5) & ~fit['absent']
    registered = np.where(valid, warped_image / np.maximum(warped_view, 1e-6), 0)
    return registered.astype(np.float32), valid, (warped_muscles > 0.5) & valid


def add_segment_overlap(entry: dict,
                        fit: dict,
                        muscles: np.ndarray,
                        registered_muscles: np.ndarray,
                        valid: np.ndarray,
                        template_muscles: np.ndarray,
                        nearest_segment: np.ndarray,
                        segment_names: list) -> dict:
    """
    Add the overlap of a registered hemibody with the template muscles to its fit entry (for
    `registration.json` and `write_segment_overlap`): the Dice over all the template muscles
    (`muscle_dice_all`), in the region closest to each segment (`muscle_dice_near_segment`) and
    with the centroid translation only (`muscle_dice_centroid_only`).

    Parameters
    ----------
    entry : dict
        The fit entry (`register_hemibody(...)['entry']`), modified in place.
    fit : dict
        Output of `register_hemibody`.
    muscles : np.ndarray
        Target muscle mask used for the fits (target grid).
    registered_muscles : np.ndarray
        Registered muscle mask (template grid, `warp_to_template`).
    valid : np.ndarray
        Where the registered hemibody has data (`warp_to_template`).
    template_muscles : np.ndarray
        Template muscle mask.
    nearest_segment : np.ndarray
        Index (1..) of the segment with the largest blending weight at each voxel.
    segment_names : list of str
        Names of the template segments.

    Returns
    -------
    dict
        `entry`.
    """
    entry['muscle_dice_all'] = dice(registered_muscles, template_muscles & valid)
    for index, segment in enumerate(segment_names, start=1):
        region = (nearest_segment == index) & valid
        entry['segments'][segment]['muscle_dice_near_segment'] = dice(
            registered_muscles & region, template_muscles & region)
    before = rs.warp_labels(muscles.astype(np.uint8), rs.invert_affine(fit['initial'])) > 0
    entry['muscle_dice_centroid_only'] = dice(before, template_muscles)
    return entry


def write_segment_overlap(segment_names: list,
                          entries: dict,
                          path: Path) -> None:
    """
    Save the overlap of each registered hemibody with the template, per segment, as a csv table:
    one row per hemibody, one column per segment (Dice of the registered and template muscle
    masks in the region closest to the segment, or 'absent' if the segment was taken as absent),
    then the Dice over all segments (`all`) and with the centroid translation only
    (`centroid_only`, if computed).

    Parameters
    ----------
    segment_names : list of str
        Segment names, in column order.
    entries : dict
        Hemibody name -> its entry of `registration.json` (`segments`, `muscle_dice_all`, ...).
    path : Path
        Output csv.

    Returns
    -------
    None
    """
    columns = ['hemibody'] + segment_names + ['all', 'centroid_only']
    lines = [','.join(columns)]
    for name, entry in entries.items():
        values = [name]
        for segment in segment_names:
            info = entry['segments'].get(segment, {})
            if 'absent' in info:
                values.append('absent')
            elif 'muscle_dice_near_segment' in info:
                values.append(f'{info["muscle_dice_near_segment"]:.3f}')
            else:
                values.append('')
        values += [f'{entry[key]:.3f}' if key in entry else ''
                   for key in ('muscle_dice_all', 'muscle_dice_centroid_only')]
        lines.append(','.join(values))
    path.write_text('\n'.join(lines) + '\n')
