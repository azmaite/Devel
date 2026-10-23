"""
Plot the temporal and spatial components of a factorisation (e.g. NMF) of a video.

plot_temporal_components: plot temporal components over time.
plot_spatial_components: plot spatial components as images.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from common.masks import unmask_image


def _check_order(order: np.ndarray | list,
                 n_components: int) -> None:
    """
    Check that `order` is valid for `n_components` components.

    Parameters
    ----------
    order : np.ndarray or list[float]
        Ordering array. May contain NaN values for blank slots. Non-NaN values
        must be unique integers in [0, n_components).
    n_components : int
        Expected number of non-NaN entries in `order`.
    """

    assert isinstance(order, (list, np.ndarray)), 'Order must be a list or numpy array'
    assert len(order) >= n_components, 'Order must be at least as long as the number of components'
    assert np.sum(~np.isnan(order)) == n_components, 'Order must have as many non-nan elements as components'

    order = np.array(order)
    non_nan = order[~np.isnan(order)].astype(int)
    assert np.array_equal(np.sort(non_nan), np.arange(0, n_components)), 'All non-nan elements must be unique and go from 0 to n_components'


def plot_temporal_components(W: np.ndarray,
                              timestamps: np.ndarray | None = None,
                              order: np.ndarray | list | None = None,
                              color: list | np.ndarray | None = None,
                              norm: bool = False,
                              fig: Figure | None = None,
                              legend: bool = True,
                              xlims: tuple | None = None) -> None:
    """Plot the temporal components as line plots, with options for ordering, coloring, and normalization."""

    if timestamps is None:
        timestamps = np.arange(W.shape[0])
    if order is None:
        order = np.arange(W.shape[1])
    else:
        _check_order(order, W.shape[1])

    if color is not None and len(color) == 1:
        color = color * len(order)

    blank = np.zeros(len(timestamps))
    if fig is None:
        plt.figure(figsize=(15, 5))

    for i, comp_i in enumerate(order):
        if np.isnan(comp_i):
            plt.plot(timestamps, blank - 1)
        else:
            temp = W[:, int(comp_i)]
            if norm:
                temp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
            if color is not None:
                plt.plot(timestamps, temp + i, label=f'Comp {int(comp_i)}', linewidth=1, color=color[i])
            else:
                plt.plot(timestamps, temp + i, label=f'Comp {int(comp_i)}', linewidth=1)

    plt.ylim(-0.5, len(order) + 0.5)
    plt.yticks(np.arange(len(order)), [f'Comp {int(comp_i)}' if not np.isnan(comp_i) else '' for comp_i in order])
    plt.xlabel('Time (s)')
    plt.ylabel('Component Activation')
    plt.title('Temporal Components from NMF')
    if legend:
        plt.legend()

    if xlims is not None:
        plt.xlim(xlims)
    else:
        plt.xlim(timestamps[0], timestamps[-1])

    if fig is None:
        plt.show()


def plot_spatial_components(H: np.ndarray,
                             mask: np.ndarray,
                             bounding_box: np.ndarray,
                             roi: np.ndarray | None = None,
                             order: np.ndarray | list | None = None,
                             cols: int = 6,
                             aspect_ratio : float = 1.0) -> None:
    """Plot the spatial components as images, with the roi contour overlaid (if present).
    If the spatial component (and mask) are 3d, they will be averaged across the z dimension before plotting."""

    if order is None:
        order = np.arange(H.shape[0])
    else:
        _check_order(order, H.shape[0])

    rows = int(np.ceil(len(order) / cols))
    _, axes = plt.subplots(rows, cols, figsize=(15, 2 * rows))
    axes = axes.flatten()

    for n, comp_i in enumerate(order):
        if np.isnan(comp_i):
            axes[n].axis('off')
            continue

        img = unmask_image(H[int(comp_i)], mask, bounding_box)

        # If the spatial component are 3D, average across z dimension
        if img.ndim == 3:
            img = img.mean(axis=0)

        axes[n].imshow(img, cmap='viridis', aspect=aspect_ratio)
        if roi is not None:
            roi_cropped = roi - bounding_box[:2]
            axes[n].plot(np.array(roi_cropped)[:, 0], np.array(roi_cropped)[:, 1], 'r--', linewidth=0.4)
        axes[n].axis('off')
        axes[n].set_title(f'Component {int(comp_i)}', fontsize=8)

    for j in range(len(order), len(axes)):
        axes[j].axis('off')
    plt.show()

