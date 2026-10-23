"""
Non-negative matrix factorisation (NMF) of flattened video frames into temporal and spatial
components, and the frame selection and normalisation used before it.

diff_norm_frames: frames minus a background image, normalised.
nmf_get_temporal: temporal components of frames for given spatial components.
fit_nmf: fit an NMF (optionally on selected frames only).
nmf_variance_explained: fraction of the variance explained by an NMF.
select_most_different_frames: pick the frames that differ most from each other.
"""

from typing import Any

import numpy as np
import tqdm
from scipy import ndimage
from sklearn.decomposition import NMF


def diff_norm_frames(frames: np.ndarray,
                     background_img: np.ndarray | None = None,
                     rolling_win: None | int = 8) -> np.ndarray:
    """
    For each frame, get the difference from the background image (median frame or
    min_proj) across all frames, and normalize the difference from 0 to 1 (ignoring
    nans). If rolling_win is provided, use a rolling window to compute the background
    image and subtract it to remove bleaching and other slow changes. 
    All This can help to enhance the signal for NMF or other methods. 
    Frames can be flat or not.
    """

    if background_img is None:
        background_img = np.nanmin(frames, axis=0)
        print('Using min projection as background image for frame differencing')

    frames_dif = frames - background_img
    frames_dif[frames_dif < 0] = 0

    if rolling_win is not None:
        rolling_min = ndimage.minimum_filter1d(frames_dif, size=rolling_win, axis=0, mode='nearest')
        frames_dif = frames_dif - rolling_min

    frames_dif_norm = (frames_dif - np.nanmin(frames_dif)) / (np.nanmax(frames_dif) - np.nanmin(frames_dif))

    return frames_dif_norm


def nmf_get_temporal(frames: np.ndarray,
                     H: np.ndarray) -> np.ndarray:
    """Given spatial components H, solve for temporal components W using frames."""

    # check that frames are non-negative and go from 0 to 1
    assert np.all(frames >= 0), "Frames must be non-negative"
    assert np.all(frames <= 1), "Frames must be normalized to [0, 1]"

    model = NMF(n_components=H.shape[0], init='custom')
    frames_type = type(frames[0, 0])
    model.components_ = H.astype(frames_type)
    W = model.transform(frames)

    return W


def fit_nmf(frames: np.ndarray,
            n_components: int,
            selected_idx: np.ndarray | None = None,
            **nmf_kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit NMF to the given frames, returning temporal and spatial components.
    If `selected_idx` is provided, only use those frames to learn the spatial
    components, but then solve for temporal components using all frames.
    This allows us to get spatial components from a subset of frames (e.g. those
    with high activity) while still getting temporal components for the whole video.

    Parameters
    ----------
    frames : np.ndarray
        2D array of shape (n_frames, n_pixels).
    n_components : int
        Number of NMF components.
    selected_idx : np.ndarray or None
        Indices of frames to use for fitting spatial components. If None, all
        frames are used.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        W (temporal components for all frames), 
        H (spatial components),
        W_orig (temporal components for selected frames only).
    """

    # check that frames are non-negative
    assert np.all(frames >= 0), "Frames must be non-negative"

    if selected_idx is None:
        selected_idx = np.arange(frames.shape[0])

    kwargs: dict[str, Any] = dict(n_components=n_components, max_iter=4000, init='nndsvda')
    kwargs.update(nmf_kwargs)
    model = NMF(**kwargs)
    W_orig = model.fit_transform(frames[selected_idx])
    H = model.components_

    if len(selected_idx) < frames.shape[0]:
        W = nmf_get_temporal(frames, H)
    else:
        W = W_orig

    return W, H, W_orig


def nmf_variance_explained(V: np.ndarray,
                            W: np.ndarray,
                            H: np.ndarray) -> float:
    """Calculate the variance explained (R^2) of the NMF reconstruction."""

    V_hat = W @ H
    ss_res = np.linalg.norm(V - V_hat, 'fro')**2
    ss_tot = np.linalg.norm(V - V.mean(), 'fro')**2

    return float(1 - ss_res / ss_tot)


def select_most_different_frames(frames_flat_dif_norm: np.ndarray,
                                  n_select: int = 200) -> list[int]:
    """Given a set of frames, select the `n_select` most different frames."""

    scores = frames_flat_dif_norm.mean(axis=1)
    selected_idx = [scores.argmax()]

    dists = np.linalg.norm(frames_flat_dif_norm - frames_flat_dif_norm[selected_idx[0]], axis=1)
    dists[selected_idx] = -np.inf

    print('Selecting most different frames...')
    for _ in tqdm.tqdm(range(n_select - 1)):
        new_selected_idx = np.argmax(dists)
        selected_idx.append(new_selected_idx)

        dists_new = np.linalg.norm(frames_flat_dif_norm - frames_flat_dif_norm[new_selected_idx], axis=1)
        dists = np.minimum(dists, dists_new)
        dists[selected_idx] = -np.inf

    return selected_idx

