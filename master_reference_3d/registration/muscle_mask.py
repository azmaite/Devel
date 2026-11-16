"""
Automatic muscle mask of a 3D fluorescence image of fly legs (e.g. an isotropic SCAPE hemibody,
1.5 um voxels), and its muscle clusters (connected components). Muscles are long bright tubes
surrounded by darker tissue:

1. Log-transform (fluorescence is multiplicative: a muscle is some factor brighter than its
   surroundings, whatever the local illumination) and subtract a smooth estimate of the local
   background. The background is a Gaussian blur computed only over the valid voxels (voxels
   that are 0, e.g. the padding left by deskewing/rotating, are ignored, and their borders are
   eroded away), on a downsampled copy for speed. The remaining 'contrast' is divided by its
   robust spread (MAD), so thresholds are in units of the noise-like spread of the contrast.
2. Hysteresis threshold of the contrast: voxels above a low threshold are kept if they connect
   to voxels above a high one.
3. Clean-up: remove small components, then components that aren't elongated (round blobs and
   specks are not muscles, which are long tubes), unless they are large (`keep_volume_um3`, for
   bulky muscles and clusters of short muscles), then fill small holes.

On the left hemibody of the reference SCAPE recording (sample1_run3_short min projection, 1.5 um
isotropic voxels), the defaults recover 92% of the voxels of the manually segmented muscles. This
is a binary mask: touching muscles are not separated.

The muscle clusters of a hemibody (`label_hemibody`) are the connected components of its mask
(face connectivity, so components touching only at an edge or corner stay separate), computed
with the default settings except that components of at least `KEEP_VOLUME_UM3` are kept whatever
their shape (the default elongation test drops bulky muscles, which matter here). They are
numbered from 1 by decreasing size and saved as a plain uint32 tif (opens as a Labels layer in
napari, same array shape and order as the image) as `{out_folder}/{image name}_components.tif`,
with a json of the voxel count of each label, and a plain tif copy of the image
(`{image name}_image.tif`, no scale) so both open together at scale 1.

segment_muscles: segment the muscles of a 3D image and return a boolean mask.
get_log_contrast: log-contrast of a 3D image relative to its local background, in robust
    standard deviations.
filter_components: remove small and non-elongated connected components of a mask.
label_components: muscle mask of an image, labelled by connected component (muscle cluster).
label_hemibody: label the muscle clusters of one hemibody tif and save them.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.ndimage as ndi
import tifffile
from skimage.filters import apply_hysteresis_threshold

KEEP_VOLUME_UM3 = 30000.0  # clusters: mask components this large skip the elongation test


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
    mask = segment_muscles(image, voxel_size_um=voxel_size_um, keep_volume_um3=KEEP_VOLUME_UM3)
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
        Hemibody tif (e.g. from `hemibodies.py`).
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
