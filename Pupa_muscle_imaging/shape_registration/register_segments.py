"""
Register the leg segments of a reference image to a target image by their SHAPE, one leg
segment at a time.

The leg segments of one leg (femur, tibia, ...) move relative to each other, so each one gets
its own transform. Registering the intensity of the muscles does not work well (see
`SPAM_registration/README.md` and `pipeline.md`): a muscle is a long tube, and the image match is nearly flat when
the tube slides along its own axis, so the optimiser has nothing to hold on to and does not
converge. The shape of a segment's muscles (its outline, with its ends) fixes the position along
the axis. Here each segment is therefore represented by the binary mask of its muscles, and the
target by the binary mask of all its muscles (e.g. from `muscle_mask.py`).

Both masks are turned into clipped signed distance maps (negative inside, positive outside), which
are smooth and have a gradient everywhere within the clip distance. A rigid transform is found
first (rotation + translation about the segment's centre), then, optionally, an affine one
starting from it, by minimising the mean squared difference of the two distance maps, only inside
the segment's muscles (the target also holds the muscles of the neighbouring segments, which would
push the segment away if the metric counted its surroundings). This runs on a downsampled copy, in
physical units (um), and the transform is then applied to the full-resolution labels.

Only the reference needs manual segment labels: the target needs just its automatic muscle mask,
and warping the reference segments with their transforms gives the target's segments.

signed_distance_map: clipped signed distance map of a binary mask, as a SimpleITK image.
register_segment_shape: register one segment's muscle mask to the target mask (rigid, then
    optionally affine); returns the SimpleITK transform (target -> reference coordinates).
warp_labels: resample a label volume of the reference into the target's grid with such a
    transform (nearest neighbour).
"""

from typing import Literal, Optional

import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk


def _to_image(array: np.ndarray, voxel_size_um: float) -> sitk.Image:
    """
    Wrap a 3D array as a SimpleITK image with isotropic spacing and zero origin.

    Parameters
    ----------
    array : np.ndarray
        3D array.
    voxel_size_um : float
        Voxel size, in um.

    Returns
    -------
    sitk.Image
        The image (array axes in reversed order as SimpleITK axes x, y, z).
    """
    image = sitk.GetImageFromArray(np.ascontiguousarray(array))
    image.SetSpacing((voxel_size_um,) * 3)
    return image


def _downsample_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample a boolean mask by block-averaging, thresholded at 0.5.

    Parameters
    ----------
    mask : np.ndarray
        3D boolean mask.
    factor : int
        Integer downsampling factor.

    Returns
    -------
    np.ndarray
        Boolean mask of the shape `ceil(shape / factor)`; voxel `i` stands for the block
        `[factor * i, factor * (i + 1))`.
    """
    pad = [(0, -s % factor) for s in mask.shape]
    blocks = np.pad(mask.astype(np.float32), pad).reshape(
        *[v for s in mask.shape for v in ((s + (-s % factor)) // factor, factor)])
    return blocks.mean(axis=(1, 3, 5)) > 0.5


def signed_distance_map(mask: np.ndarray,
                        voxel_size_um: float,
                        clip_um: float) -> np.ndarray:
    """
    Clipped signed distance map of a binary mask (negative inside, positive outside), in um.

    Parameters
    ----------
    mask : np.ndarray
        3D boolean mask.
    voxel_size_um : float
        Voxel size (isotropic), in um.
    clip_um : float
        Distances are clipped to `[-clip_um, clip_um]`.

    Returns
    -------
    np.ndarray
        Float32 array, same shape as `mask`.
    """
    outside = ndi.distance_transform_edt(~mask, sampling=voxel_size_um)
    inside = ndi.distance_transform_edt(mask, sampling=voxel_size_um)
    return np.clip(outside - inside, -clip_um, clip_um).astype(np.float32)


def _unwrap(transform: sitk.Transform) -> sitk.Transform:
    """
    Return the single transform inside a (nested) composite transform, as returned by
    `sitk.ImageRegistrationMethod.Execute` with `inPlace=False`.

    Parameters
    ----------
    transform : sitk.Transform
        A transform, possibly a composite holding one transform.

    Returns
    -------
    sitk.Transform
        The innermost transform.
    """
    while transform.GetName() == 'CompositeTransform':
        transform = sitk.CompositeTransform(transform).GetNthTransform(0)
    return transform


def to_affine(transform: sitk.Transform) -> sitk.AffineTransform:
    """
    Flatten an affine-like transform (rigid, affine, translation, or a composite of these) into a
    single `sitk.AffineTransform` (centre at the origin).

    The matrix and offset are read off the images of the origin and the unit vectors, which is
    exact for any composition of affine maps.

    Parameters
    ----------
    transform : sitk.Transform
        Affine-like transform.

    Returns
    -------
    sitk.AffineTransform
        The same mapping, as one affine transform.
    """
    origin = np.array(transform.TransformPoint((0.0, 0.0, 0.0)))
    matrix = np.stack([np.array(transform.TransformPoint(tuple(e))) - origin
                       for e in np.eye(3)], axis=1)
    affine = sitk.AffineTransform(3)
    affine.SetMatrix(matrix.ravel().tolist())
    affine.SetTranslation(origin.tolist())
    return affine


def invert_affine(transform: sitk.Transform) -> sitk.AffineTransform:
    """
    Inverse of an affine-like transform (see `to_affine`).

    Parameters
    ----------
    transform : sitk.Transform
        Affine-like transform.

    Returns
    -------
    sitk.AffineTransform
        Its inverse.
    """
    affine = to_affine(transform)
    matrix = np.array(affine.GetMatrix()).reshape(3, 3)
    inverse_matrix = np.linalg.inv(matrix)
    inverse = sitk.AffineTransform(3)
    inverse.SetMatrix(inverse_matrix.ravel().tolist())
    inverse.SetTranslation((-inverse_matrix @ np.array(affine.GetTranslation())).tolist())
    return inverse


def _register(fixed: sitk.Image,
              moving: sitk.Image,
              support: sitk.Image,
              initial: sitk.Transform,
              iterations: int,
              shrink_factors: tuple,
              moving_initial: Optional[sitk.Transform] = None,
              fixed_support: Optional[sitk.Image] = None) -> sitk.Transform:
    """
    Register `moving` to `fixed` (mean squares, coarse to fine), within `support`.

    Parameters
    ----------
    fixed : sitk.Image
        Fixed image (here: target distance map).
    moving : sitk.Image
        Moving image (here: segment distance map).
    support : sitk.Image
        UInt8 mask in the moving image's space; only that region counts in the metric.
    initial : sitk.Transform
        Starting transform (modified copy is optimised).
    iterations : int
        Maximum number of iterations per resolution level.
    shrink_factors : tuple of int
        Shrink factor of each resolution level, coarse to fine (no smoothing beyond that).
    moving_initial : sitk.Transform or None
        Fixed transform applied after the optimised one (fixed -> optimised -> moving_initial ->
        moving coordinates), not optimised.
    fixed_support : sitk.Image or None
        UInt8 mask in the fixed image's space; points mapped outside it do not count.

    Returns
    -------
    sitk.Transform
        The optimised transform alone (without `moving_initial`).
    """
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMeanSquares()
    method.SetMetricMovingMask(support)
    if fixed_support is not None:
        method.SetMetricFixedMask(fixed_support)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=1e-4,
                                                    numberOfIterations=iterations,
                                                    relaxationFactor=0.7,
                                                    gradientMagnitudeTolerance=1e-9)
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel(list(shrink_factors))
    method.SetSmoothingSigmasPerLevel([0] * len(shrink_factors))
    if moving_initial is not None:
        method.SetMovingInitialTransform(moving_initial)
    method.SetInitialTransform(initial, inPlace=False)
    return _unwrap(method.Execute(fixed, moving))


def register_segment_shape(segment_mask: np.ndarray,
                           target_mask: np.ndarray,
                           voxel_size_um: float = 1.5,
                           model: Literal['rigid', 'affine'] = 'affine',
                           downsample: int = 2,
                           clip_um: float = 30.0,
                           support_um: float = 0.0,
                           shrink_factors: tuple = (2, 1),
                           iterations: int = 200,
                           initial: Optional[sitk.Transform] = None,
                           target_valid: Optional[np.ndarray] = None) -> sitk.AffineTransform:
    """
    Register the muscle mask of one reference segment to the muscle mask of the target.

    See the module docstring. The reference and the target must already be roughly aligned (e.g.
    by a global rigid pre-alignment): the optimisation is local, with a capture range of about
    `clip_um`.

    Parameters
    ----------
    segment_mask : np.ndarray
        Boolean mask, on the target's grid, of the muscles of one segment of the reference (e.g.
        the automatic muscle mask of the reference within the segment's manual mask).
    target_mask : np.ndarray
        Boolean mask of all the muscles of the target, same shape.
    voxel_size_um : float
        Voxel size (isotropic), in um.
    model : 'rigid' or 'affine'
        'rigid': rotation and translation. 'affine': a rigid fit followed by an affine one
        (adds scaling and shear).
    downsample : int
        Factor by which the masks are shrunk for the registration.
    clip_um : float
        Clip distance of the distance maps, in um (the capture range).
    support_um : float
        Only the voxels within this distance of the segment's muscles count in the metric
        (0, the default: only the voxels inside them). The target holds the muscles of the
        neighbouring segments too, which the segment's mask does not, so counting the
        neighbourhood penalises the segment for being next to its neighbours and pushes it away.
        Inside the segment's muscles, the metric asks whether the target has muscle of a
        matching thickness there (the distance map inside a muscle is minus the distance to its
        surface).
    shrink_factors : tuple of int
        Shrink factors of the resolution levels, coarse to fine, on top of `downsample`.
        Coarser levels blur thin muscles away.
    iterations : int
        Maximum number of iterations per resolution level.
    initial : sitk.Transform or None
        Starting estimate of the transform (target -> reference coordinates, e.g. a global fit
        or a translation between centroids). The fit refines it, so the two images need only be
        within the capture range after it. None: identity.
    target_valid : np.ndarray or None
        Boolean mask of the target's field of view (e.g. `target_image > 0`), same shape.
        Outside it (and outside the target grid) the target is treated as unknown, at a small
        cost, so a segment that is cut off by the target's field of view is matched by its
        visible part, without being pushed back inside or escaping out of it. (Ignoring the
        points outside altogether does not work: moving the whole segment out of view then
        leaves nothing to penalise.) None: the target's distance map is used everywhere.

    Returns
    -------
    sitk.AffineTransform
        Transform from target to reference physical coordinates (in um, SimpleITK axis
        order), including `initial`, as one affine transform (see `to_affine`). To be used with
        `warp_labels`; its inverse (`invert_affine`) maps reference to target coordinates.
    """
    small_voxel = voxel_size_um * downsample
    # BinShrink-like placement: voxel i of the small grid is centred on the centre of its block
    origin = (small_voxel - voxel_size_um) / 2
    small_segment = _downsample_mask(segment_mask, downsample)
    small_target = _downsample_mask(target_mask, downsample)

    def make(array: np.ndarray) -> sitk.Image:
        image = _to_image(array, small_voxel)
        image.SetOrigin((origin,) * 3)
        return image

    target_distance = signed_distance_map(small_target, small_voxel, clip_um)
    if target_valid is None:
        fixed = make(target_distance)
    else:
        # outside the field of view (from a little inside its edge) the target is unknown: the
        # distance map is set to 0 there, and the grid gets a margin of unknown all around, so
        # that a cut-off part of the segment costs little (the squared distance to its own
        # surface, small for a muscle) but less than lying on target muscle, and the segment can
        # neither be pushed back inside the field of view nor escape out of it
        small_valid = ndi.binary_erosion(_downsample_mask(target_valid, downsample), iterations=2)
        target_distance[~small_valid] = 0.0
        margin = int(np.ceil(2 * clip_um / small_voxel))
        fixed = _to_image(np.pad(target_distance, margin), small_voxel)
        fixed.SetOrigin((origin - margin * small_voxel,) * 3)
    segment_distance = signed_distance_map(small_segment, small_voxel, clip_um)
    moving = make(segment_distance)
    support = make((segment_distance <= support_um).astype(np.uint8))
    fixed_support = None  # (a fixed mask lets the segment escape out of it, see above)

    centre = ndi.center_of_mass(small_segment)  # array order (z, y, x)
    centre_physical = tuple(float(c) * small_voxel + origin for c in centre[::-1])
    if initial is not None:
        # the optimised transform acts in target coordinates (before `initial`), so rotate about
        # the segment centre as seen from the target
        initial = to_affine(initial)
        centre_physical = invert_affine(initial).TransformPoint(centre_physical)
    rigid = sitk.Euler3DTransform()
    rigid.SetCenter(centre_physical)
    transform = _register(fixed, moving, support, rigid, iterations, shrink_factors, initial,
                          fixed_support)
    if model == 'affine':
        rigid = sitk.Euler3DTransform(transform)
        affine = sitk.AffineTransform(3)
        affine.SetMatrix(rigid.GetMatrix())
        affine.SetTranslation(rigid.GetTranslation())
        affine.SetCenter(rigid.GetCenter())
        transform = _register(fixed, moving, support, affine, iterations, shrink_factors,
                              initial, fixed_support)
    if initial is None:
        return to_affine(transform)
    # target -> optimised -> initial -> reference
    return to_affine(sitk.CompositeTransform([initial, transform]))


def warp_labels(labels: np.ndarray,
                transform: sitk.Transform,
                voxel_size_um: float = 1.5,
                output_shape: Optional[tuple] = None) -> np.ndarray:
    """
    Resample a label volume of the reference into the target's grid (nearest neighbour).

    Parameters
    ----------
    labels : np.ndarray
        Integer or boolean label volume on the target's grid (as the reference masks given to
        `register_segment_shape`).
    transform : sitk.Transform
        Transform from `register_segment_shape`.
    voxel_size_um : float
        Voxel size (isotropic), in um.
    output_shape : tuple or None
        Shape of the output grid (the target's); defaults to the shape of `labels`.

    Returns
    -------
    np.ndarray
        Warped labels, same dtype as `labels`, on the target's grid.
    """
    shape = tuple(labels.shape if output_shape is None else output_shape)
    moving = _to_image(labels.astype(np.uint32), voxel_size_um)
    grid = _to_image(np.zeros(shape, dtype=np.uint8), voxel_size_um)
    warped = sitk.Resample(moving, grid, transform, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt32)
    return sitk.GetArrayFromImage(warped).astype(labels.dtype)
