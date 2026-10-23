"""
Script to automatically segment muscles in a 3D fluorescence image of fly legs (e.g. a SCAPE
min projection, or a hemibody of it). Muscles are long bright tubes surrounded by darker tissue,
so the image is segmented as follows:

1. Log-transform (fluorescence is multiplicative: a muscle is some factor brighter than its
   surroundings, whatever the local illumination) and subtract a smooth estimate of the local
   background. The background is a Gaussian blur computed only over the valid voxels (voxels
   that are 0, e.g. the padding left by deskewing/rotating, are ignored, and their borders are
   eroded away), on a downsampled copy for speed. The remaining 'contrast' is divided by its
   robust spread (MAD), so thresholds are in units of the noise-like spread of the contrast.
2. Hysteresis threshold of the contrast: voxels above a low threshold are kept if they connect
   to voxels above a high one.
3. Clean-up: remove small components, then components that aren't elongated (round blobs and
   specks are not muscles, which are long tubes), then fill small holes.

Tested on the left hemibody of the reference SCAPE recording (sample1_run3_short min
projection, 1.5 um isotropic voxels), the defaults recover 92% of the voxels of the manually
segmented muscles, and (within the leg segments that were manually labelled, where only about
half of the volume is labelled muscle) give a Dice of 0.83 against them; a few seconds to run.
The same settings also segment the other recording's hemibodies well by eye. This is a binary
mask: touching muscles are not separated. The Sato/Frangi tube filter was also tried and did
not improve the match to the manual muscles over the contrast threshold alone, so it is not
used.

The binary mask can then be split into individual muscles (`split_muscles`), as a first pass
to be curated by hand: a seeded watershed of a band-pass (difference of Gaussians, 3 and 9 um)
of the log image inside the mask. Each muscle is brighter than the dim seams that separate it
from its neighbours, and the band-pass also gives a muscle that sits on the shoulder of a
brighter one a maximum of its own (in the raw image it has none, and is merged into the bright
one). The band-pass is smoothed along the local tube axis (from the structure tensor of the
image) before finding the markers, so that the brightness fluctuations along one tube do not
cut it into consecutive pieces, while the dim seams between side-by-side tubes (which run
parallel to the axis) are kept. The markers are the local maxima that stand out from the
saddle towards any brighter neighbour by at least `h` standard deviations of the band-pass.
They are found on a downsampled copy and the boundaries are then refined at full resolution.
Finally, pieces that were still cut across a tube are merged if their contact cuts across the
local tube axis and has no dark seam, with strict thresholds (see `_merge_end_to_end`).
`h` sets the granularity: lower splits more, higher merges. The aim is to never join two
muscles, even at the cost of cutting some in pieces (pieces are easy to merge by hand, a
joined pair has to be split by hand).

Tuned against manual labels of 11 muscles of the left hemibody of the reference recording
(sample1_run3_short; two further manually labelled muscles were left out, a tiny one that the
mask misses and one with an unreliable label). With the defaults (64 labels in the whole
hemibody), no label holds a substantial part of two manual muscles (98.5% of the manually
labelled voxels are in a label whose main muscle is their own), each manual muscle is covered
by 1.7 labels on average, and the union of the labels that lie mostly inside a manual muscle
has a Dice of 0.77 on average with it (0.81-0.87 for most muscles; about the limit set by how
well the binary mask outline agrees with the hand-drawn one). Without smoothing along the tube
the same scores need 129 labels and 3.9 labels per muscle. The manual muscles group some
sub-muscles and leave others out, so extra labels beyond them are expected.

The length of the smoothing along the tube is a trade-off: longer joins more of the pieces of
a tube cut by brightness fluctuations, but also blurs a small distinct structure lying in line
with a muscle, joined to it by a thin neck, until it is merged into the muscle (at 24 um one
such structure was; at 18 um it stays separate, and the pieces of the tubes checked by hand
are still joined).

Known limit: some consecutive pieces of one long, bent tube are not merged. They cannot be
told apart from the pieces of two different muscles that meet end to end with the contact
alignment, dip, intensity, size and orientation features tried (they overlap), and looser
thresholds join different manual muscles. Merge those by hand.

Can be run from the terminal
(`python auto_segment_muscles.py image.tif output.tif --voxel-size 1.5 [--labels-output
labels.tif]`) or imported and called as a function. The input can be a .tif or a .npy file
with a 3D array.

segment_muscles: segment the muscles of a 3D image and return a boolean mask.
get_log_contrast: log-contrast of a 3D image relative to its local background, in robust
    standard deviations.
filter_components: remove small and non-elongated connected components of a mask.
split_muscles: split a muscle mask into individual muscles (integer labels, 0 = background).
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.ndimage as ndi
import tifffile
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import h_maxima
from skimage.segmentation import watershed


def get_log_contrast(volume: np.ndarray,
                     valid: np.ndarray,
                     voxel_size_um: float,
                     background_sigma_um: float = 30.0,
                     smooth_sigma_um: float = 1.5,
                     downsample: int = 2) -> np.ndarray:
    """
    Log-contrast of `volume` relative to its local background, in robust standard deviations.

    The log of the image is smoothed lightly (to reduce noise) and its local background, a
    Gaussian blur over the valid voxels only (normalized convolution, computed on a copy
    downsampled by `downsample` and brought back to full resolution), is subtracted. The result
    is divided by the median absolute deviation (x 1.4826) of the valid voxels. Invalid voxels
    are set to 0.

    Parameters
    ----------
    volume : np.ndarray
        3D image (isotropic voxels), brighter = more fluorescence.
    valid : np.ndarray
        Boolean mask of the voxels to use (e.g. not padding).
    voxel_size_um : float
        Voxel size of `volume` (isotropic), in um.
    background_sigma_um : float
        Standard deviation of the Gaussian used to estimate the local background, in um. Should
        be larger than the width of a muscle.
    smooth_sigma_um : float
        Standard deviation of the Gaussian used to denoise the image, in um.
    downsample : int
        Factor by which the volume is shrunk to estimate the (smooth) background.

    Returns
    -------
    np.ndarray
        Float32 array, same shape as `volume`: log-contrast in robust standard deviations
        (0 outside `valid`).
    """
    log_image = np.log(np.maximum(volume, 1)).astype(np.float32)
    log_image = ndi.gaussian_filter(log_image, smooth_sigma_um / voxel_size_um)

    # background by normalized convolution on a downsampled copy, then back to full resolution
    small_shape = tuple(int(np.ceil(s / downsample)) for s in volume.shape)
    zoom_down = tuple(n / s for n, s in zip(small_shape, volume.shape))
    valid_small = ndi.zoom(valid.astype(np.float32), zoom_down, order=1)
    log_small = ndi.zoom(np.where(valid, log_image, 0), zoom_down, order=1)
    sigma = background_sigma_um / voxel_size_um / downsample
    weight = ndi.gaussian_filter(valid_small, sigma)
    background_small = ndi.gaussian_filter(log_small, sigma) / np.maximum(weight, 1e-3)
    zoom_up = tuple(s / n for s, n in zip(volume.shape, small_shape))
    background = ndi.zoom(background_small, zoom_up, order=1)
    # zoom can be off by a voxel from the target shape
    background = background[tuple(slice(0, s) for s in volume.shape)]
    if background.shape != volume.shape:
        background = np.pad(background, [(0, s - b) for s, b in zip(volume.shape, background.shape)],
                            mode='edge')

    contrast = np.where(valid, log_image - background, 0).astype(np.float32)
    spread = np.median(np.abs(contrast[valid] - np.median(contrast[valid]))) * 1.4826
    return contrast / spread


def filter_components(mask: np.ndarray,
                      voxel_size_um: float,
                      min_volume_um3: float,
                      min_elongation: float,
                      min_length_um: float,
                      keep_volume_um3: Optional[float] = None) -> np.ndarray:
    """
    Remove the small and the non-elongated connected components of a mask.

    A component is kept if its volume is at least `min_volume_um3`, and, from the principal axes
    of its voxel coordinates (standard deviations `s1 >= s2 >= s3`), its elongation `s1 / s2`
    is at least `min_elongation` and its length (`4 * s1`, about the extent of the component
    along its main axis) is at least `min_length_um`. A tube is long compared with its width,
    whereas a round blob or a speck has an elongation near 1. Components of at least
    `keep_volume_um3` are kept whatever their shape (bulky muscles, or several short muscles
    joined together, are not elongated as a whole).

    Parameters
    ----------
    mask : np.ndarray
        Boolean 3D mask.
    voxel_size_um : float
        Voxel size (isotropic), in um.
    min_volume_um3 : float
        Minimum component volume, in um^3.
    min_elongation : float
        Minimum ratio of the largest to the second largest principal standard deviation.
    min_length_um : float
        Minimum component length (see above), in um.
    keep_volume_um3 : float or None
        Components at least this large (in um^3) skip the shape test. None: no exception.

    Returns
    -------
    np.ndarray
        Boolean mask with only the components that were kept.
    """
    labels, n_labels = ndi.label(mask, structure=np.ones((3, 3, 3)))
    if n_labels == 0:
        return mask
    min_voxels = min_volume_um3 / voxel_size_um ** 3
    sizes = np.bincount(labels.ravel(), minlength=n_labels + 1)
    keep = np.zeros(n_labels + 1, dtype=bool)
    for label, bbox in enumerate(ndi.find_objects(labels), start=1):
        if bbox is None or sizes[label] < min_voxels:
            continue
        if keep_volume_um3 is not None and sizes[label] * voxel_size_um ** 3 >= keep_volume_um3:
            keep[label] = True
            continue
        coords = np.argwhere(labels[bbox] == label).astype(np.float64)
        eigenvalues = np.linalg.eigvalsh(np.cov(coords.T))[::-1]  # descending
        s1, s2 = np.sqrt(np.maximum(eigenvalues[:2], 1e-12))
        keep[label] = (s1 / s2 >= min_elongation) and (4 * s1 * voxel_size_um >= min_length_um)
    return keep[labels]


def segment_muscles(volume: np.ndarray,
                    voxel_size_um: float = 1.5,
                    valid: Optional[np.ndarray] = None,
                    low_threshold: float = 4.0,
                    high_threshold: float = 8.0,
                    background_sigma_um: float = 30.0,
                    smooth_sigma_um: float = 1.5,
                    edge_erosion_um: float = 9.0,
                    min_volume_um3: float = 5000.0,
                    min_elongation: float = 1.8,
                    min_length_um: float = 40.0,
                    keep_volume_um3: Optional[float] = None,
                    fill_holes: bool = True) -> np.ndarray:
    """
    Segment the muscles (long bright tubes) of a 3D fluorescence image.

    See the module docstring for the method. Thresholds are in units of the robust standard
    deviation of the log-contrast (see `get_log_contrast`): the defaults were chosen on a SCAPE
    min projection of fly legs, 1.5 um isotropic voxels, where they recover about 92% of the
    voxels of manually segmented muscles (see the module docstring). Lower thresholds find dimmer muscles (and more
    background), higher ones keep only the brightest.

    Parameters
    ----------
    volume : np.ndarray
        3D image (isotropic voxels), brighter = more fluorescence. Voxels equal to 0 are
        treated as invalid (padding) unless `valid` is given.
    voxel_size_um : float
        Voxel size of `volume` (isotropic), in um.
    valid : np.ndarray or None
        Boolean mask of the voxels to analyse. If None, `volume > 0`.
    low_threshold : float
        Low hysteresis threshold of the contrast (see above).
    high_threshold : float
        High hysteresis threshold of the contrast: a component must reach it to be kept.
    background_sigma_um : float
        See `get_log_contrast`.
    smooth_sigma_um : float
        See `get_log_contrast`.
    edge_erosion_um : float
        Distance from the border of the valid region that is ignored, since the padding edge
        looks like a bright edge after background subtraction.
    min_volume_um3 : float
        Minimum volume of a component (see `filter_components`).
    min_elongation : float
        Minimum elongation of a component (see `filter_components`).
    min_length_um : float
        Minimum length of a component (see `filter_components`).
    keep_volume_um3 : float or None
        Components at least this large skip the shape test (see `filter_components`). None
        (default, as tuned against the manual muscles): no exception. Large bright regions that
        are not tube-shaped (e.g. bulky muscles, 50-110k um^3, elongation 1.2-1.8, in the
        reference recording) are then dropped, so set it (e.g. 30000) when the mask must cover
        all the muscles.
    fill_holes : bool
        Whether to fill the holes in the mask, in the 2D slices along each of the three axes
        (muscles are solid, so a hole is a dim patch inside one).

    Returns
    -------
    np.ndarray
        Boolean mask, same shape as `volume`, True on the muscles.
    """
    if valid is None:
        valid = volume > 0
    valid = ndi.binary_erosion(valid, iterations=max(int(round(edge_erosion_um / voxel_size_um)), 1))

    contrast = get_log_contrast(volume, valid, voxel_size_um, background_sigma_um,
                                smooth_sigma_um)
    mask = apply_hysteresis_threshold(contrast, low_threshold, high_threshold) & valid
    mask = filter_components(mask, voxel_size_um, min_volume_um3, min_elongation, min_length_um,
                             keep_volume_um3)

    if fill_holes:
        for axis in range(3):
            mask = np.moveaxis(np.stack([ndi.binary_fill_holes(plane)
                                         for plane in np.moveaxis(mask, axis, 0)]), 0, axis)
    return mask


def _resize_labels(labels: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Resize a label volume to `shape` by nearest-neighbour interpolation.

    Parameters
    ----------
    labels : np.ndarray
        3D integer label volume.
    shape : tuple of int
        Target shape.

    Returns
    -------
    np.ndarray
        Labels, with shape `shape`.
    """
    resized = ndi.zoom(labels, [t / s for t, s in zip(shape, labels.shape)], order=0)
    out = np.zeros(shape, dtype=labels.dtype)
    common = tuple(slice(0, min(a, b)) for a, b in zip(shape, resized.shape))
    out[common] = resized[common]
    return out


def _tube_axes(log_image: np.ndarray,
               mask: np.ndarray,
               voxel_size_um: float,
               gradient_sigma_um: float,
               tensor_sigma_um: float) -> tuple:
    """
    Local tube axis at every voxel of `mask`, from the structure tensor of the log image.

    In a tube the gradient is perpendicular to the axis, so the axis is the direction of the
    smallest eigenvalue of the structure tensor (the Gaussian-smoothed outer product of the
    gradient). The axis is only defined up to its sign.

    Parameters
    ----------
    log_image : np.ndarray
        3D log image.
    mask : np.ndarray
        Boolean mask of the voxels to compute the axis at.
    voxel_size_um : float
        Voxel size of `log_image` (isotropic), in um.
    gradient_sigma_um : float
        Standard deviation of the Gaussian smoothing before taking the gradient, in um.
    tensor_sigma_um : float
        Standard deviation of the Gaussian that averages the tensor, in um (about the width
        of a muscle).

    Returns
    -------
    indices : np.ndarray
        Array (N, 3) of the voxel coordinates of the voxels of `mask`.
    axes : np.ndarray
        Array (N, 3) of unit vectors, the local axis at each of those voxels.
    """
    gradient = np.gradient(ndi.gaussian_filter(log_image, gradient_sigma_um / voxel_size_um))
    sigma = tensor_sigma_um / voxel_size_um
    indices = np.argwhere(mask)
    tensor = np.zeros((len(indices), 3, 3))
    for i in range(3):
        for j in range(i, 3):
            component = ndi.gaussian_filter(gradient[i] * gradient[j], sigma)[mask]
            tensor[:, i, j] = component
            tensor[:, j, i] = component
    return indices, np.linalg.eigh(tensor)[1][:, :, 0]


def _smooth_along_axes(image: np.ndarray,
                       mask: np.ndarray,
                       indices: np.ndarray,
                       axes: np.ndarray,
                       sigma_um: float,
                       voxel_size_um: float) -> np.ndarray:
    """
    Smooth `image` inside `mask` with a 1D Gaussian along the local tube axis of every voxel.

    Fluctuations of the intensity along a tube average out, while the dim seams between
    neighbouring tubes, which run parallel to the axis, are kept. Voxels outside `mask` do not
    contribute (normalized average).

    Parameters
    ----------
    image : np.ndarray
        3D image to smooth.
    mask : np.ndarray
        Boolean mask (the voxels at which the result is defined).
    indices : np.ndarray
        Voxel coordinates of the voxels of `mask`, array (N, 3).
    axes : np.ndarray
        Local axis of those voxels, array (N, 3) (see `_tube_axes`).
    sigma_um : float
        Standard deviation of the Gaussian along the axis, in um.
    voxel_size_um : float
        Voxel size (isotropic), in um.

    Returns
    -------
    np.ndarray
        Copy of `image` with the voxels of `mask` replaced by their smoothed value.
    """
    sigma = sigma_um / voxel_size_um
    half = int(np.ceil(2.5 * sigma))
    weights_sum = np.zeros(len(indices))
    values_sum = np.zeros(len(indices))
    mask_float = mask.astype(np.float32)
    for step in range(-half, half + 1):
        weight = np.exp(-0.5 * (step / sigma) ** 2)
        coordinates = (indices + step * axes).T
        inside = ndi.map_coordinates(mask_float, coordinates, order=1, mode='nearest')
        values = ndi.map_coordinates(image, coordinates, order=1, mode='nearest')
        values_sum += weight * values * inside
        weights_sum += weight * inside
    smoothed = image.copy()
    smoothed[mask] = values_sum / np.maximum(weights_sum, 1e-6)
    return smoothed


def _merge_end_to_end(labels: np.ndarray,
                      log_image: np.ndarray,
                      indices: np.ndarray,
                      axes: np.ndarray,
                      voxel_size_um: float,
                      min_alignment: float,
                      max_dip: float,
                      min_contact_um2: float,
                      rounds: int = 3) -> np.ndarray:
    """
    Merge pieces of a tube that were cut across it, conservatively.

    Two touching labels are merged if (1) the local tube axis at their contact is along the
    normal of the contact surface (|normal . axis| >= `min_alignment`, i.e. the contact cuts
    across the tube instead of running along it, also for a curved tube), (2) the log intensity
    at the contact is not lower than the lower of the two labels' median by more than `max_dip`
    (no dark seam) and (3) the contact is at least `min_contact_um2`. In each round the best
    (lowest dip) non-overlapping pairs are merged; the rounds stop when nothing is left to
    merge. These criteria do not separate every pair of pieces of one muscle from the pairs of
    two different muscles, so they are kept strict (tested against manual muscles).

    Parameters
    ----------
    labels : np.ndarray
        3D integer label volume (0 = background).
    log_image : np.ndarray
        3D log image, same shape.
    indices : np.ndarray
        Voxel coordinates of the voxels with a known axis, array (N, 3) (see `_tube_axes`).
    axes : np.ndarray
        Local tube axis at those voxels, array (N, 3).
    voxel_size_um : float
        Voxel size (isotropic), in um.
    min_alignment : float
        Minimum |normal . axis| at the contact (0 = along the contact, 1 = across it).
    max_dip : float
        Maximum dip, in log intensity, of the contact below the dimmer label.
    min_contact_um2 : float
        Minimum area of the contact, in um^2.
    rounds : int
        Maximum number of merging rounds.

    Returns
    -------
    np.ndarray
        Label volume with the merged labels, numbered 1..N.
    """
    axis_volume = np.zeros(labels.shape + (3,), dtype=np.float32)
    axis_volume[tuple(indices.T)] = axes
    face_area = voxel_size_um ** 2
    for _ in range(rounds):
        n_labels = labels.max() + 1
        contacts = {}
        for axis in range(3):
            first = [slice(None)] * 3
            second = [slice(None)] * 3
            first[axis], second[axis] = slice(0, -1), slice(1, None)
            first, second = tuple(first), tuple(second)
            label_a, label_b = labels[first], labels[second]
            faces = (label_a != label_b) & (label_a > 0) & (label_b > 0)
            lower = np.minimum(label_a[faces], label_b[faces]).astype(np.int64)
            upper = np.maximum(label_a[faces], label_b[faces]).astype(np.int64)
            keys, inverse = np.unique(lower * n_labels + upper, return_inverse=True)
            intensity = (log_image[first][faces] + log_image[second][faces]) / 2
            axis_a, axis_b = axis_volume[first][faces], axis_volume[second][faces]
            scatter = (axis_a[:, :, None] * axis_a[:, None, :]
                       + axis_b[:, :, None] * axis_b[:, None, :]).reshape(-1, 9)
            count = np.bincount(inverse)
            intensity_sum = np.bincount(inverse, weights=intensity)
            scatter_sum = np.stack([np.bincount(inverse, weights=scatter[:, k])
                                    for k in range(9)], axis=1)
            for key, c, i_sum, s_sum in zip(keys, count, intensity_sum, scatter_sum):
                entry = contacts.setdefault(int(key), [np.zeros(3), 0.0, np.zeros(9)])
                entry[0][axis] += c
                entry[1] += i_sum
                entry[2] += s_sum
        if not contacts:
            break
        ids = np.arange(1, n_labels)
        medians = np.zeros(n_labels)
        medians[1:] = ndi.labeled_comprehension(log_image, labels, ids, np.median, float, 0)
        candidates = []
        for key, (counts, intensity_sum, scatter_sum) in contacts.items():
            a, b = divmod(key, n_labels)
            n_faces = counts.sum()
            if n_faces * face_area < min_contact_um2:
                continue
            normal = counts / np.linalg.norm(counts)
            main_axis = np.linalg.eigh(scatter_sum.reshape(3, 3))[1][:, -1]
            dip = min(medians[a], medians[b]) - intensity_sum / n_faces
            if abs(normal @ main_axis) >= min_alignment and dip <= max_dip:
                candidates.append((dip, a, b))
        if not candidates:
            break
        lookup = np.arange(n_labels)
        used = set()
        for dip, a, b in sorted(candidates):
            if a not in used and b not in used:
                lookup[b] = a
                used.update((a, b))
        labels = lookup[labels]
        labels = np.unique(labels, return_inverse=True)[1].reshape(labels.shape).astype(np.int32)
    return labels


def split_muscles(volume: np.ndarray,
                  mask: np.ndarray,
                  voxel_size_um: float = 1.5,
                  h: float = 0.4,
                  small_sigma_um: float = 3.0,
                  large_sigma_um: float = 9.0,
                  along_sigma_um: float = 18.0,
                  tensor_sigma_um: float = 15.0,
                  merge_end_to_end: bool = True,
                  min_volume_um3: float = 3000.0,
                  downsample: int = 2) -> np.ndarray:
    """
    Split a muscle mask into individual muscles (see the module docstring).

    The landscape is a band-pass (difference of Gaussians) of the log image: a muscle that sits
    on the shoulder of a brighter one has no local maximum of its own in the raw image, but it
    does in the band-pass. The band-pass is then smoothed along the local tube axis (see
    `_tube_axes`, `_smooth_along_axes`), so that brightness fluctuations along one tube do not
    cut it into pieces, while the dim seams between side-by-side tubes are kept. Markers are
    the local maxima of this landscape inside `mask` that are at least `h` standard deviations
    (of the landscape inside the mask) above the saddle towards any brighter maximum
    (`skimage.morphology.h_maxima`), found on a copy downsampled by `downsample`. A watershed
    on the inverted landscape, restricted to the mask, grows them into regions; regions smaller
    than `min_volume_um3` are merged into their neighbours (and mask components smaller than
    that are dropped), and, if `merge_end_to_end`, pieces of a tube cut across it are merged
    (see `_merge_end_to_end`). The labels are then brought back to full resolution and the
    boundaries refined there: a second watershed on the full-resolution band-pass, from the
    labels eroded by one voxel, which grows them into every voxel of the mask that is connected
    to one.

    Parameters
    ----------
    volume : np.ndarray
        3D image (isotropic voxels), brighter = more fluorescence.
    mask : np.ndarray
        Boolean muscle mask of `volume`, e.g. from `segment_muscles`.
    voxel_size_um : float
        Voxel size (isotropic), in um.
    h : float
        Minimum height of a marker over the saddle to its brighter neighbour, in standard
        deviations of the landscape inside the mask. Lower = more, smaller muscles.
    small_sigma_um : float
        Standard deviation of the narrow Gaussian of the band-pass, in um (about the
        thickness of the seam between two muscles).
    large_sigma_um : float
        Standard deviation of the wide Gaussian of the band-pass, in um (about the width of a
        muscle or more).
    along_sigma_um : float
        Standard deviation, in um, of the smoothing along the tube axis (0 = none). Longer
        gives fewer cuts along a tube.
    tensor_sigma_um : float
        Scale, in um, at which the local tube axis is estimated (about the width of a muscle).
    merge_end_to_end : bool
        Whether to merge pieces of a tube that were cut across it (conservatively).
    min_volume_um3 : float
        Minimum volume of a muscle, in um^3.
    downsample : int
        Factor by which the volume is shrunk to find the markers.

    Returns
    -------
    np.ndarray
        Int32 label volume, same shape as `volume`: 1..N for the muscles, 0 for background.
    """
    log_image = np.log(np.maximum(volume, 1)).astype(np.float32)

    def band_pass(image: np.ndarray, voxel_um: float) -> np.ndarray:
        return (ndi.gaussian_filter(image, small_sigma_um / voxel_um)
                - ndi.gaussian_filter(image, large_sigma_um / voxel_um))

    small_log = ndi.zoom(log_image, 1 / downsample, order=1)
    small_mask = _resize_labels(mask.astype(np.uint8), small_log.shape) > 0
    small_voxel_um = voxel_size_um * downsample
    small = band_pass(small_log, small_voxel_um)
    indices, axes = _tube_axes(small_log, small_mask, small_voxel_um, small_sigma_um,
                               tensor_sigma_um)
    if along_sigma_um > 0:
        small = _smooth_along_axes(small, small_mask, indices, axes, along_sigma_um,
                                   small_voxel_um)
    height = h * small[small_mask].std()
    small = np.where(small_mask, small, small[small_mask].min() - 1)

    min_voxels = min_volume_um3 / small_voxel_um ** 3
    structure = np.ones((3, 3, 3))
    markers, _ = ndi.label(h_maxima(small, height) & small_mask, structure=structure)
    labels = watershed(-small, markers, mask=small_mask)
    for _ in range(2):  # merging regions can leave their neighbours small too
        sizes = np.bincount(labels.ravel())
        too_small = np.flatnonzero(sizes < min_voxels)
        too_small = too_small[too_small > 0]
        if too_small.size == 0:
            break
        markers[np.isin(markers, too_small)] = 0
        markers, _ = ndi.label(markers > 0, structure=structure)
        labels = watershed(-small, markers, mask=small_mask)
    # mask components smaller than the minimum volume stay unlabelled; renumber 1..N
    labels = np.unique(labels, return_inverse=True)[1].reshape(labels.shape).astype(np.int32)
    if merge_end_to_end:
        labels = _merge_end_to_end(labels, small_log, indices, axes, small_voxel_um,
                                   min_alignment=0.75, max_dip=0.0,
                                   min_contact_um2=360.0)

    # back to full resolution: refine the boundaries there, from the labels eroded by one voxel
    full = _resize_labels(labels, volume.shape)
    eroded = np.where(ndi.minimum_filter(full, size=3) == ndi.maximum_filter(full, size=3), full, 0)
    markers_full = np.where(mask, eroded, 0)
    refined = watershed(-band_pass(log_image, voxel_size_um), markers_full, mask=mask)  # no marker = stays 0
    return refined.astype(np.int32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automatically segment the muscles of a 3D '
                                                 'fly-leg image (see module docstring).')
    parser.add_argument('image', help='3D image, .tif or .npy.')
    parser.add_argument('output', help='Output boolean mask, saved as a uint8 .tif (0/1).')
    parser.add_argument('--voxel-size', type=float, default=1.5,
                        help='Isotropic voxel size in um (default 1.5).')
    parser.add_argument('--low', type=float, default=4.0, help='Low hysteresis threshold.')
    parser.add_argument('--high', type=float, default=8.0, help='High hysteresis threshold.')
    parser.add_argument('--labels-output', default=None,
                        help='If given, also split the mask into individual muscles and save '
                             'the labels (uint32 .tif, opens as a Labels layer in napari) here.')
    parser.add_argument('--h', type=float, default=0.4,
                        help='Marker height for splitting into muscles, in standard deviations '
                             'of the band-pass image (lower splits more).')
    args = parser.parse_args()

    image_path = Path(args.image)
    image = np.load(image_path) if image_path.suffix == '.npy' else tifffile.imread(image_path)
    muscle_mask = segment_muscles(image, voxel_size_um=args.voxel_size,
                                  low_threshold=args.low, high_threshold=args.high)
    tifffile.imwrite(args.output, muscle_mask.astype(np.uint8))
    print(f'Saved {args.output}: {muscle_mask.sum()} voxels ({100 * muscle_mask.mean():.1f}%)')
    if args.labels_output:
        muscle_labels = split_muscles(image, muscle_mask, voxel_size_um=args.voxel_size, h=args.h)
        # uint32: napari opens only 32/64-bit integer tifs as Labels (smaller types as Images)
        tifffile.imwrite(args.labels_output, muscle_labels.astype(np.uint32))
        print(f'Saved {args.labels_output}: {muscle_labels.max()} muscles')
