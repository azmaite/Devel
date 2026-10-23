"""
Mask and unmask images and videos: keep only the pixels inside a mask (flattened), and put them
back into an image of the bounding box of the mask.

mask_image: pixels of an image inside a mask, flattened.
unmask_image: put flattened mask pixels back into an image of the mask's bounding box.
mask_video: `mask_image` for every frame of a video.
unmask_video: `unmask_image` for every frame.
merge_masks: union of a mask and a submask placed in it.
intersect_masks: intersection of two masks.
"""

import numpy as np


def mask_image(image: np.ndarray,
               mask: np.ndarray,
               bounding_box: np.ndarray | None = None) -> np.ndarray:
    """Extract 1D array of image values under the mask."""

    # check that the dimensions of the mask and image are compatible
    assert mask.shape == image.shape, f'Mask shape {mask.shape} does not match image shape {image.shape}'

    if bounding_box is not None:
        x, y, w, h = bounding_box
        mask = mask[y:y+h, x:x+w]
    return image[mask.astype(bool)]


def unmask_image(values: np.ndarray,
                 mask: np.ndarray,
                 bounding_box: np.ndarray | None = None) -> np.ndarray:
    """Reconstruct 2D image from masked values, zeros elsewhere
    and crop to bounding box (x, y, w, h) if provided."""

    # check that the number of values matches the number of pixels in the mask
    assert np.sum(mask) == len(values), f'Number of values {len(values)} does not match number of pixels in mask {np.sum(mask)}'

    result = np.zeros(mask.shape)
    result[mask.astype(bool)] = values

    if bounding_box is not None:
        x, y, w, h = bounding_box
        result = result[y:y+h, x:x+w]

    return result


def mask_video(video: np.ndarray,
               mask: np.ndarray,
               bounding_box: np.ndarray | None = None) -> np.ndarray:
    """Apply mask to each frame of video, returning 2D array of masked values."""
    return np.array([mask_image(frame, mask, bounding_box) for frame in video])


def unmask_video(masked_frames: np.ndarray,
                 mask: np.ndarray,
                 bounding_box: np.ndarray | None = None) -> np.ndarray:
    """Reconstruct video from masked frames, using unmask_image for each frame."""
    return np.array([unmask_image(frame, mask, bounding_box) for frame in masked_frames])


def merge_masks(mask: np.ndarray,
                submask: np.ndarray) -> np.ndarray:
    """
    Combine a submask with the original mask, returning a new mask with the shape
    of the original but with the submask selected pixels only.
    """
    # check that the submask is a 1d array 
    # with the same number of elements as positive values in the original mask
    assert submask.ndim == 1, f'Submask must be a 1D array, but has shape {submask.shape}'
    assert len(mask) == np.sum(mask), f'Number of pixels in mask {len(mask)} does not match number of pixels in submask {np.sum(submask)}'

    # merge masks
    new_mask = np.zeros_like(mask)
    new_mask[submask.astype(bool)] = 1

    return new_mask


def intersect_masks(mask1: np.ndarray,
                    mask2: np.ndarray) -> np.ndarray:
    """ Combine two masks by taking the intersection of their positive pixels."""
    # check that the masks have the same shape
    assert mask1.shape == mask2.shape, f'Mask shapes {mask1.shape} and {mask2.shape} do not match'

    new_mask = np.zeros_like(mask1)
    new_mask[(mask1.astype(bool)) & (mask2.astype(bool))] = 1
    return new_mask

