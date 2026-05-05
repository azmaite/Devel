#!/usr/bin/env python3
"""
Interactive GUI widget for aligning 3D pupa anatomy (CT-scan Neuroglancer traces)
onto a 2D fluorescence image.

Displays an image (from a .mkv or .h5 file) alongside a set of sliders controlling
a 6-DOF rigid transformation: scale, X/Y/Z rotations, and X/Y pixel shifts. As the
sliders are adjusted, concave-hull outlines of the 3D leg segments are projected
onto the image in real time. A debounce timer prevents lag during slider dragging.

When the user clicks 'Done' or closes the window, the final transformation parameters
are saved to 'pupa_segments_aligned.h5' in the same folder as the input image and
returned as a dict. If that file already exists, its values are loaded as initial
slider positions.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import h5py
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as geom
import shapely.ops as ops
from matplotlib.widgets import Button, Slider
from scipy.spatial import Delaunay

import utils


def _rotate_points(points: np.ndarray,
                   rx: float,
                   ry: float,
                   rz: float) -> np.ndarray:
    """
    Apply extrinsic XYZ rotation to an array of 3D points.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 3).
    rx : float
        Rotation around X axis in degrees.
    ry : float
        Rotation around Y axis in degrees.
    rz : float
        Rotation around Z axis in degrees.

    Returns
    -------
    np.ndarray
        Rotated points of shape (N, 3).
    """
    cx, sx = np.cos(np.radians(rx)), np.sin(np.radians(rx))
    cy, sy = np.cos(np.radians(ry)), np.sin(np.radians(ry))
    cz, sz = np.cos(np.radians(rz)), np.sin(np.radians(rz))

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    return points @ Rx.T @ Ry.T @ Rz.T


def _shade(base: tuple | np.ndarray,
           factor: float) -> tuple:
    """
    Interpolate a base RGB color toward white.

    Parameters
    ----------
    base : tuple or np.ndarray
        Base RGB color with values in [0, 1].
    factor : float
        Blend factor in [0, 1]. 0 returns `base`, 1 returns white.

    Returns
    -------
    tuple
        Interpolated RGB color.
    """
    base = np.array(base)
    return tuple(base + (1 - base) * factor)


def _set_colors_by_segment() -> dict[str, tuple]:
    """
    Build a color map for leg segments, with a light-to-dark gradient from
    proximal (COXA) to distal (TARSUS) within each leg type.

    Returns
    -------
    dict[str, tuple]
        Mapping of segment name (e.g. 'Front FEMUR') to RGB color tuple.
    """
    leg_parts = ['COXA', 'TROCHANTER', 'FEMUR', 'TIBIA', 'TARSUS']
    factors = np.linspace(0.0, 0.6, len(leg_parts))  # darker → lighter distal

    colors_dict = {}
    for i, part in enumerate(leg_parts):
        colors_dict[f'Front {part}'] = _shade(mcolors.to_rgb('firebrick'),   factors[i])
        colors_dict[f'Mid {part}']   = _shade(mcolors.to_rgb('yellowgreen'), factors[i])
        colors_dict[f'Hind {part}']  = _shade(mcolors.to_rgb('dodgerblue'),  factors[i])

    return colors_dict


def _alpha_shape(points: np.ndarray,
                 alpha: float = 0.04) -> np.ndarray:
    """
    Compute the alpha shape (concave hull) of a set of 2D points.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 2).
    alpha : float
        Alpha parameter controlling concavity. Larger values give more concave
        hulls; smaller values approach the convex hull.

    Returns
    -------
    np.ndarray
        Ordered (x, y) coordinates of the hull outline, shape (M, 2).
    """
    tri = Delaunay(points)
    edges: set = set()

    def add_edge(i, j):
        if (i, j) in edges or (j, i) in edges:
            edges.discard((j, i))
        else:
            edges.add((i, j))

    for ia, ib, ic in tri.simplices:
        pa, pb, pc = points[[ia, ib, ic]]
        a = np.linalg.norm(pb - pc)
        b = np.linalg.norm(pa - pc)
        c = np.linalg.norm(pa - pb)
        s = (a + b + c) / 2.0
        area = np.sqrt(max(s*(s-a)*(s-b)*(s-c), 0))
        if area == 0:
            continue
        circum_r = a*b*c / (4.0*area)
        if circum_r < 1.0 / alpha:
            add_edge(ia, ib)
            add_edge(ib, ic)
            add_edge(ic, ia)

    edge_points = [(points[i], points[j]) for i, j in edges]
    m = geom.MultiLineString(edge_points)
    polys = list(ops.polygonize(m))
    return np.array(polys[0].exterior.coords)


def interactive_alignment(image_path: str | Path) -> dict:
    """
    Launch the interactive 3D-to-2D alignment GUI.

    Loads the background image from `image_path` (.mkv or .h5), then opens a
    Matplotlib window with sliders for scale, XYZ rotation, and XY shift. Leg
    segment outlines (from a hardcoded CT-scan trace file) are redrawn after
    each slider interaction. Blocks until the window is closed.

    If 'pupa_segments_aligned.h5' exists in the same folder as `image_path`,
    its stored parameters are used as initial slider values. On exit, the final
    parameters are saved to that file.

    Parameters
    ----------
    image_path : str or Path
        Path to a .mkv raw recording or a .h5 file containing a 'median' dataset
        (and optionally an 'aspect_ratio' scalar).

    Returns
    -------
    dict
        Final alignment parameters with keys 'scale' (float),
        'rotation_deg_xyz' (tuple of 3 floats), 'shift_px_xy' (tuple of 2 floats).
    """
    image_path = Path(image_path)
    h5_save_path = image_path.parent / 'pupa_segments_aligned.h5'

    # load the background image
    aspect_ratio = 1.0
    if image_path.suffix == '.mkv':
        image = utils.load_median_frame(str(image_path))
    elif image_path.suffix == '.h5':
        with h5py.File(image_path, 'r') as f:
            image = f['median'][:]
            if 'aspect_ratio' in f:
                aspect_ratio = float(f['aspect_ratio'][()])
    else:
        raise ValueError(f"Unsupported image format '{image_path.suffix}'. Expected .mkv or .h5.")

    # load 3D traced leg coordinates from CT-scan Neuroglancer annotations
    traces_file = Path('/mnt/labserver/data/MA/Development_project/CT_PupaP15_traced/pupa_traced_coordinates_rot.pkl')
    with open(traces_file, 'rb') as f:
        points_3d = pickle.load(f)

    # determine which segments to overlay and their display order
    plot_order = ['Mid TIBIA', 'Front COXA', 'Front TROCHANTER', 'Front FEMUR', 'Front TIBIA']
    plot_order = [s.replace(' ', ' leg ') for s in plot_order]
    plot_order = [f'{side} {seg}' for seg in plot_order for side in ['Left', 'Right']]

    colors_dict_0 = _set_colors_by_segment()
    colors_dict = {seg: colors_dict_0[f"{seg.split()[1]} {seg.split()[3]}"] for seg in plot_order}

    # load previous alignment values as starting values if the h5 file exists
    init_scale = 0.6
    init_rx, init_ry, init_rz = 90, 0, -90
    init_tx, init_ty = image.shape[1] // 2, image.shape[0] // 2

    if h5_save_path.exists():
        print(f'Found existing alignment: {h5_save_path}. Loading as starting values.')
        with h5py.File(h5_save_path, 'r') as f:
            init_scale = float(f['scale'][()])
            init_rx, init_ry, init_rz = [float(v) for v in f['rotation_deg_xyz'][:]]
            init_tx, init_ty = [float(v) for v in f['shift_px_xy'][:]]

    # build figure
    fig, ax = plt.subplots(figsize=(18, 8))
    plt.subplots_adjust(right=0.75, left=0)
    ax.imshow(image, cmap='gray', aspect=aspect_ratio)
    ax.set_title("Align 3D legs to image.  Close window or click 'Done' to finish.")

    outline_plots = {
        seg: ax.plot([], [], color=colors_dict[seg], lw=1, zorder=5)[0]
        for seg in plot_order
    }

    # debounce timer: redraws outlines only after sliders stop moving for 50 ms
    timer = fig.canvas.new_timer(interval=50)

    def compute_and_draw_outlines():
        """
        Recompute and redraw all segment outlines from current slider values.
        Only called by the debounce timer, not directly on each slider event.
        """
        s = s_scale.val
        rx, ry, rz = s_rx.val, s_ry.val, s_rz.val
        tx, ty = s_tx.val, s_ty.val

        for segment in plot_order:
            p = _rotate_points(points_3d[segment] * s, rx, ry, rz)
            points_2d = np.column_stack((p[:, 0] + tx, p[:, 1] + ty))
            outline = _alpha_shape(points_2d, alpha=0.04)
            x_out = outline[:, 0]
            y_out = outline[:, 1] / aspect_ratio if aspect_ratio != 1.0 else outline[:, 1]
            outline_plots[segment].set_data(x_out, y_out)

        fig.canvas.draw_idle()

    timer.add_callback(compute_and_draw_outlines)

    # peek timer: briefly hides outlines to compare with the raw image
    peek_timer = fig.canvas.new_timer(interval=200)
    peek_timer.single_shot = True

    def show_outlines():
        for plot in outline_plots.values():
            plot.set_visible(True)
        fig.canvas.draw_idle()

    peek_timer.add_callback(show_outlines)

    # sliders
    left, width, height = 0.74, 0.21, 0.025
    s_scale = Slider(plt.axes([left, 0.80, width, height]), 'Scale',    0.3,  2.0,                    valinit=init_scale, valstep=0.01, valfmt='%.2f')
    s_rx    = Slider(plt.axes([left, 0.70, width, height]), 'Rot X',    0,    180,                    valinit=init_rx,    valstep=1,    valfmt='%d°')
    s_ry    = Slider(plt.axes([left, 0.65, width, height]), 'Rot Y',    -90,  90,                     valinit=init_ry,    valstep=1,    valfmt='%d°')
    s_rz    = Slider(plt.axes([left, 0.60, width, height]), 'Rot Z',    -270, 90,                     valinit=init_rz,    valstep=1,    valfmt='%d°')
    s_tx    = Slider(plt.axes([left, 0.50, width, height]), 'Shift X',  -100, image.shape[1] + 100,   valinit=init_tx,    valstep=1,    valfmt='%d')
    s_ty    = Slider(plt.axes([left, 0.45, width, height]), 'Shift Y',  -100, image.shape[0] + 100,   valinit=init_ty,    valstep=1,    valfmt='%d')

    def update(val):
        """Restart the debounce timer on every slider change."""
        timer.stop()
        timer.start()
        fig.canvas.draw_idle()

    for s in [s_scale, s_rx, s_ry, s_rz, s_tx, s_ty]:
        s.on_changed(update)

    # buttons
    first_button_x, button_height = 0.735, 0.35

    btn_reset = Button(plt.axes([first_button_x,          button_height, 0.07, 0.04]), 'Reset',         color='lightgray',  hovercolor='gray')
    btn_peek  = Button(plt.axes([first_button_x + 0.08,   button_height, 0.07, 0.04]), 'Hide outlines', color='lightblue',  hovercolor='dodgerblue')
    btn_done  = Button(plt.axes([first_button_x + 0.08*2, button_height, 0.07, 0.04]), 'Done',          color='#90EE90',    hovercolor='#32CD32')

    def reset(event):
        for s in [s_scale, s_rx, s_ry, s_rz, s_tx, s_ty]:
            s.reset()
    btn_reset.on_clicked(reset)

    def peek(event):
        for plot in outline_plots.values():
            plot.set_visible(False)
        fig.canvas.draw_idle()
        peek_timer.start()
    btn_peek.on_clicked(peek)

    def done(event):
        plt.close(fig)
    btn_done.on_clicked(done)

    update(None)
    plt.show()  # blocks until window is closed (via Done button or window X)

    # collect and save final parameters
    params = {
        'scale': s_scale.val,
        'rotation_deg_xyz': (s_rx.val, s_ry.val, s_rz.val),
        'shift_px_xy': (s_tx.val, s_ty.val),
    }

    with h5py.File(h5_save_path, 'w') as f:
        f.create_dataset('scale', data=params['scale'])
        f.create_dataset('rotation_deg_xyz', data=params['rotation_deg_xyz'])
        f.create_dataset('shift_px_xy', data=params['shift_px_xy'])
    print(f'Saved alignment parameters to {h5_save_path}')

    return params


if __name__ == '__main__':
    import tkinter as tk
    from tkinter import filedialog

    tk_root = tk.Tk()
    tk_root.withdraw()
    image_path = filedialog.askopenfilename(
        initialdir='/mnt/labserver/data/MA/',
        title='Select image file (.mkv or .h5)',
        filetypes=[('Image files', '*.mkv *.h5'), ('All files', '*.*')]
    )
    if image_path:
        params = interactive_alignment(image_path)
        print(params)
