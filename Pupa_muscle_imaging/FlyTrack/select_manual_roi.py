#!/usr/bin/env python3
"""
Draw free-hand Regions of Interest (ROIs) on the std image of a recording.

The input is a *folder* that contains a ``projection_images.h5`` or
``projection_images_grouped.h5`` file with an ``std`` dataset (written by the
FlyTrack preprocessing, or the SCAPE preprocessing for 3D data). That std image
is what the ROIs are drawn on, and the results are saved next to it in
``segment_rois.h5``.

2D data (FlyTrack)
------------------
The std image is shown and one free-hand ROI is drawn per segment.

3D data (SCAPE, std shape ``(Z, X, Y)``)
----------------------------------------
Each ROI is drawn in two views and intersected:
  1. the volume flattened across axis 0 (Z) -> an ``(X, Y)`` image; draw the ROI
     there.
  2. the volume flattened across axis 1 (X) -> a ``(Z, Y)`` image, with the Y
     columns selected by the first ROI shaded; draw the ROI there.
The final 3D mask is the overlap of the two views along the shared Y axis. Both
views are rescaled to their true physical proportions using the voxel size from
``scapeio`` so drawing is not distorted by anisotropic voxels.

Standard ROI sets
-----------------
Ready-made segment-name sets live in the `STANDARD_SETS` dictionary and are
referenced with the ``s.`` prefix (e.g. ``s.legs_front``, ``s.legs``), either in
the segment-name string or typed at the prompt. The part after ``s.`` is looked
up in `STANDARD_SETS` and expanded to its segment names.

Drawing uses matplotlib and opens a separate window per ROI. If the active
backend is non-interactive (e.g. ``Agg``, which the preprocessing modules set on
import), an interactive one is selected automatically (Tk, then Qt), so no
``%matplotlib`` magic is needed. A Qt/Tk toolkit must be installed: tkinter is
usually present, otherwise install a Qt binding (``pip install PyQt5``).

Usage
-----
    import select_manual_roi

    # interactive: pick the folder and type the segment names
    select_manual_roi()

    # pass the folder and the segment names (comma-separated string or list;
    # standard sets referenced as 's.<name>')
    select_manual_roi('/path/to/analysis_folder', 'RF_femur,RF_tibia')
    select_manual_roi('/path/to/recording', 's.legs_front')
"""


import os

import numpy as np
import h5py


MAIN_PATH = '/mnt/labserver/data/MA/Development_project'

# names of the projection-images h5 files searched for in the input folder, in
# priority order (the grouped/aligned one is preferred when both are present)
PROJ_H5_NAMES = ('projection_images_grouped.h5', 'projection_images.h5')

# tube lens is always this one for the SCAPE setup; used to get the voxel size
TUBE_LENS = '50 mm Navitar'


### STANDARD ROI SETS ###

# ready-made segment-name sets, referenced in the segment-name input by the
# `s.` prefix (e.g. `s.legs_front` -> STANDARD_SETS['legs_front']).
#
#   legs_front : the four segments (coxa, trochanter, femur, tibia) of both
#                front legs (RF, LF); 8 names.
#   legs       : legs_front plus the trochanter and tibia of the middle (M) and
#                hind (H) legs on both sides (RM, RH, LM, LH); 16 names.
_FRONT_SEGMENTS = ('coxa', 'trochanter', 'femur', 'tibia')

STANDARD_SETS: dict[str, tuple[str, ...]] = {
    'legs_front': tuple(f'{side}F_{seg}'
                        for side in ('R', 'L')
                        for seg in _FRONT_SEGMENTS),
}

STANDARD_SETS['legs'] = STANDARD_SETS['legs_front'] + tuple(
    f'{side}{pos}_{seg}'
    for side in ('R', 'L')
    for pos in ('M', 'H')
    for seg in ('trochanter', 'tibia'))


### INPUT PARSING ###

def _resolve_set(token: str) -> list[str] | None:
    """
    Expand a standard-set reference (``s.<name>``) into its segment names.

    A token starting with ``s.`` is treated as a standard-set reference: the part
    after ``s.`` is looked up in `STANDARD_SETS`. Any other token is a literal
    segment name and returns None.

    Parameters
    ----------
    token : str
        A single, stripped token from the segment-name input.

    Returns
    -------
    list of str or None
        The expanded set, or None if `token` is not a ``s.`` set reference.

    Raises
    ------
    KeyError
        If `token` is a ``s.`` reference but names no known set.
    """

    if not token.startswith('s.'):
        return None
    name = token[2:]
    if name not in STANDARD_SETS:
        raise KeyError(f"Unknown standard set '{token}'. Available: "
                       f"{', '.join('s.' + k for k in STANDARD_SETS)}")
    return list(STANDARD_SETS[name])


def _parse_segment_names(raw: str | list | tuple) -> list[str]:
    """
    Turn the segment-name input into a flat list of names, expanding sets.

    Accepts a list/tuple of names (used as-is, but any ``s.`` set reference in it
    is still expanded) or a comma-separated string. In a string, each
    comma-separated token is either a standard-set reference (``s.legs``) that is
    expanded, or a literal segment name. Whitespace and empty tokens are ignored.

    Parameters
    ----------
    raw : str or list or tuple
        The raw segment-name input.

    Returns
    -------
    list of str
        The flattened, de-duplicated (order-preserving) list of segment names.
    """

    tokens = list(raw) if isinstance(raw, (list, tuple)) else raw.split(',')

    names: list[str] = []
    for tok in tokens:
        tok = str(tok).strip()
        if not tok:
            continue
        expanded = _resolve_set(tok)
        names.extend(expanded if expanded is not None else [tok])

    # de-duplicate while preserving order
    seen: set[str] = set()
    return [n for n in names if not (n in seen or seen.add(n))]


### IMAGE LOADING AND DISPLAY HELPERS ###

def _find_proj_h5(folder: str) -> str:
    """
    Return the path to the projection-images h5 in `folder`.

    Looks for `projection_images_grouped.h5` first, then `projection_images.h5`.

    Parameters
    ----------
    folder : str
        Folder to search.

    Returns
    -------
    str
        Path to the found h5 file.

    Raises
    ------
    FileNotFoundError
        If neither file is present in `folder`.
    """

    for name in PROJ_H5_NAMES:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f'Folder does not contain {" or ".join(PROJ_H5_NAMES)}: {folder}')


def _get_voxel_size(folder: str) -> np.ndarray:
    """
    Return the (z, x, y) voxel size in microns for 3D SCAPE data.

    Uses `scapeio.get_voxel_size(folder, TUBE_LENS, deskewed=True)`. If scapeio is
    unavailable or fails, warns and falls back to isotropic (1, 1, 1) voxels so
    the views are drawn without physical rescaling.

    Parameters
    ----------
    folder : str
        Recording folder passed to scapeio.

    Returns
    -------
    np.ndarray
        Voxel size as [z_um, x_um, y_um].
    """

    try:
        import scapeio
        voxel_size = np.asarray(
            scapeio.get_voxel_size(folder, TUBE_LENS, deskewed=True), dtype=float)
        print(f'Voxel size (z, x, y): {voxel_size[0]:.3f} x {voxel_size[1]:.3f} '
              f'x {voxel_size[2]:.3f} µm')
        return voxel_size
    except Exception as e:
        print(f'Could not get voxel size from scapeio ({e}); '
              f'assuming isotropic voxels.')
        return np.ones(3)


def _autocontrast_8bit(img: np.ndarray, hi_pct: float = 99.8) -> np.ndarray:
    """
    Map an image to 8-bit, auto-contrasted between its minimum and a percentile.

    The image is linearly rescaled so its minimum maps to 0 and its `hi_pct`
    percentile maps to 255 (brighter values clipped), which keeps outliers from
    washing out the visible range.

    Parameters
    ----------
    img : np.ndarray
        Image to rescale.
    hi_pct : float, optional
        Percentile mapped to 255.

    Returns
    -------
    np.ndarray
        The auto-contrasted image as uint8.
    """

    img_f = img.astype(np.float64)
    lo = img_f.min()
    hi = np.percentile(img_f, hi_pct)
    if hi <= lo:
        hi = img_f.max()
    scaled = np.clip((img_f - lo) / (hi - lo + 1e-9), 0, 1) * 255
    return scaled.astype(np.uint8)


_INTERACTIVE_BACKENDS = ('qt', 'tk', 'gtk', 'wx', 'macosx', 'nbagg')


def _ensure_interactive_backend() -> None:
    """
    Make sure an interactive matplotlib backend is active for drawing.

    The free-hand drawing needs an interactive, windowed backend whose blocking
    event loop the lasso relies on. If the current backend is already interactive
    nothing changes; otherwise (e.g. ``Agg``, which importing a preprocessing
    module that calls ``matplotlib.use('Agg')`` leaves active) this switches to an
    interactive backend directly, so the caller does not have to run a
    ``%matplotlib`` magic (and it is not blocked by an earlier magic locking the
    GUI toolkit). Tk is tried first (needs only tkinter), then Qt.

    Raises
    ------
    RuntimeError
        If no interactive backend can be activated.
    """

    import matplotlib
    import matplotlib.pyplot as plt

    backend = matplotlib.get_backend().lower()
    if any(name in backend for name in _INTERACTIVE_BACKENDS):
        return

    for candidate in ('TkAgg', 'QtAgg'):
        try:
            plt.switch_backend(candidate)
            print(f'Switched matplotlib backend to {candidate} for ROI drawing.')
            return
        except Exception:
            continue

    raise RuntimeError(
        f'matplotlib backend is "{backend}" and no interactive backend could be '
        f'activated for drawing. Ensure tkinter is available, or install a Qt '
        f'binding (e.g. "pip install PyQt5"), then retry.')


def _tint_columns_rgb(gray: np.ndarray,
                      col_mask: np.ndarray,
                      color: tuple[int, int, int] = (0, 180, 0),
                      alpha: float = 0.35) -> np.ndarray:
    """
    Make an RGB image from a grayscale one, tinting the columns in `col_mask`.

    Used to shade, in the axis-1 (Z, Y) view, the Y columns selected by the
    axis-0 (X, Y) ROI drawn first.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale uint8 image, shape (rows, cols).
    col_mask : np.ndarray
        Boolean array of length `cols`; True columns are tinted.
    color : tuple of int, optional
        RGB tint color.
    alpha : float, optional
        Tint opacity (0-1).

    Returns
    -------
    np.ndarray
        RGB uint8 image, shape (rows, cols, 3).
    """

    rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float64)
    if np.any(col_mask):
        tint = np.array(color, dtype=np.float64)
        rgb[:, col_mask, :] = ((1 - alpha) * rgb[:, col_mask, :] + alpha * tint)
    return rgb.astype(np.uint8)


def _draw_freehand(img: np.ndarray,
                   row_um: float,
                   col_um: float,
                   title: str,
                   overlays: list[np.ndarray] | None = None) -> np.ndarray:
    """
    Show a physically-proportioned image and let the user draw one free-hand ROI.

    A matplotlib window is opened (a free-hand `LassoSelector`); the image aspect
    is set to ``row_um / col_um`` so anisotropic voxels are shown at their true
    proportions. The user drags to trace a closed region, then presses ENTER to
    accept it (``r`` clears the current trace and redraws). Reference `overlays`
    (previous ROIs, in (col, row) coordinates) are drawn dashed for context.

    Parameters
    ----------
    img : np.ndarray
        Voxel-resolution image to draw on: 2D grayscale uint8, or (rows, cols, 3)
        RGB uint8 (e.g. the tinted Z-Y view).
    row_um, col_um : float
        Physical size of a voxel along the row and column axes (microns).
    title : str
        Instruction shown as the axes title.
    overlays : list of np.ndarray or None, optional
        Polygons (each (N, 2) as (col, row) coords) drawn dashed for reference.

    Returns
    -------
    np.ndarray
        The drawn polygon as an (N, 2) int array of (col, row) coordinates, or an
        empty (0, 2) array if fewer than 3 points were drawn.
    """

    import matplotlib.pyplot as plt
    from matplotlib.widgets import LassoSelector

    _ensure_interactive_backend()

    rows, cols = img.shape[:2]
    aspect = (float(row_um) or 1.0) / (float(col_um) or 1.0)
    overlays = [np.asarray(p) for p in (overlays or []) if len(p) >= 2]

    fig, ax = plt.subplots(figsize=(9, 9))

    def _draw_base():
        ax.clear()
        if img.ndim == 2:
            ax.imshow(img, cmap='gray', aspect=aspect, interpolation='nearest')
        else:
            ax.imshow(img, aspect=aspect, interpolation='nearest')
        for p in overlays:
            ax.plot(np.append(p[:, 0], p[0, 0]), np.append(p[:, 1], p[0, 1]),
                    '--', color='yellow', lw=0.8)
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)
        ax.set_title(f'{title}\n'
                     'drag to trace   •   ENTER to accept   •   r to redo')
        fig.canvas.draw_idle()

    _draw_base()

    state: dict = {'verts': None}

    def _on_select(verts):
        state['verts'] = verts
        v = np.asarray(verts)
        ax.plot(np.append(v[:, 0], v[0, 0]), np.append(v[:, 1], v[0, 1]),
                '-', color='red', lw=1.5)
        fig.canvas.draw_idle()

    lasso = LassoSelector(ax, _on_select)

    def _on_key(event):
        if event.key == 'enter':
            fig.canvas.stop_event_loop()
        elif event.key in ('r', 'R'):
            state['verts'] = None
            _draw_base()

    fig.canvas.mpl_connect('key_press_event', _on_key)

    print(f'Draw ROI: {title}  (drag to trace, ENTER to accept, r to redo)')
    plt.show(block=False)
    fig.canvas.start_event_loop()  # blocks until ENTER -> stop_event_loop()
    lasso.disconnect_events()
    plt.close(fig)

    verts = state['verts']
    if verts is None or len(verts) < 3:
        return np.zeros((0, 2), dtype=np.int32)

    poly = np.array(verts, dtype=np.float64)  # (col, row) in data coordinates
    poly[:, 0] = np.clip(poly[:, 0], 0, cols - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, rows - 1)
    return np.round(poly).astype(np.int32)


def _poly_to_mask(poly: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """
    Rasterize a polygon (given as (col, row) points) into a filled uint8 mask.

    Parameters
    ----------
    poly : np.ndarray
        (N, 2) array of (col, row) coordinates.
    shape : tuple of int
        (rows, cols) of the output mask.

    Returns
    -------
    np.ndarray
        uint8 mask of `shape` with 1 inside the polygon, 0 elsewhere.
    """

    from skimage.draw import polygon as _sk_polygon

    mask = np.zeros(shape, dtype=np.uint8)
    if len(poly) >= 3:
        rr, cc = _sk_polygon(poly[:, 1], poly[:, 0], shape)  # (rows=y, cols=x)
        mask[rr, cc] = 1
    return mask


def _bbox_2d(mask: np.ndarray) -> list[int]:
    """
    Return the [x, y, w, h] bounding box of the non-zero pixels of a 2D mask.

    Parameters
    ----------
    mask : np.ndarray
        2D mask.

    Returns
    -------
    list of int
        [x, y, w, h] with x, y the top-left column/row and w, h the width/height.
        All zeros if the mask is empty.
    """

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x0, y0 = int(xs.min()), int(ys.min())
    return [x0, y0, int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1]


### PER-SEGMENT ROI SELECTION ###

def _select_roi_2d(std_gray: np.ndarray,
                   name: str,
                   overlays: list[np.ndarray]) -> dict:
    """
    Draw one 2D ROI on the std image.

    Parameters
    ----------
    std_gray : np.ndarray
        Grayscale uint8 std image, shape (H, W).
    name : str
        Segment name (shown as the drawing title).
    overlays : list of np.ndarray
        Previously drawn ROIs (voxel (col, row) polygons) for reference.

    Returns
    -------
    dict
        Keys: 'roi' ((N, 2) (col, row)), 'mask' (2D uint8), 'bounding_box'
        ([x, y, w, h]). Empty if no ROI was drawn.
    """

    rows, cols = std_gray.shape[:2]
    poly = _draw_freehand(std_gray, 1.0, 1.0, name, overlays)
    if len(poly) < 3:
        print(f'  No ROI drawn for {name}, skipping.')
        return {}

    mask = _poly_to_mask(poly, (rows, cols))
    return {'roi': poly, 'mask': mask, 'bounding_box': _bbox_2d(mask)}


def _select_roi_3d(std: np.ndarray,
                   name: str,
                   voxel_size: np.ndarray) -> dict:
    """
    Draw one 3D ROI as the overlap of an (X, Y) and a (Z, Y) view.

    The std volume (Z, X, Y) is max-projected across axis 0 (Z) for the first
    view and across axis 1 (X) for the second; the Y columns selected in the
    first view are shaded in the second. Both views are physically scaled by
    `voxel_size`. The final mask is the intersection along the shared Y axis:
    ``mask3d[z, x, y] = mask_xy[x, y] and mask_zy[z, y]``.

    Parameters
    ----------
    std : np.ndarray
        3D std image, shape (Z, X, Y).
    name : str
        Segment name (shown as the drawing title).
    voxel_size : np.ndarray
        [z_um, x_um, y_um] voxel size.

    Returns
    -------
    dict
        Keys: 'roi_xy', 'roi_zy' (polygons, (col, row) = (Y, X) and (Y, Z)),
        'mask' (3D uint8 (Z, X, Y)), 'bounding_box' ([z0, x0, y0, dz, dx, dy]).
        Empty if either view's ROI was not drawn.
    """

    z_um, x_um, y_um = voxel_size

    # view 1: flatten across axis 0 (Z) -> (X, Y); rows = X, cols = Y
    xy_img = _autocontrast_8bit(std.max(axis=0))
    poly_xy = _draw_freehand(xy_img, x_um, y_um, f'{name}  [X-Y view]')
    if len(poly_xy) < 3:
        print(f'  No X-Y ROI drawn for {name}, skipping.')
        return {}
    mask_xy = _poly_to_mask(poly_xy, std.shape[1:])  # (X, Y)

    # view 2: flatten across axis 1 (X) -> (Z, Y); rows = Z, cols = Y.
    # shade the Y columns selected by the X-Y ROI as a drawing guide.
    zy_gray = _autocontrast_8bit(std.max(axis=1))
    y_selected = np.any(mask_xy.astype(bool), axis=0)  # (Y,)
    zy_img = _tint_columns_rgb(zy_gray, y_selected)
    poly_zy = _draw_freehand(zy_img, z_um, y_um, f'{name}  [Z-Y view]')
    if len(poly_zy) < 3:
        print(f'  No Z-Y ROI drawn for {name}, skipping.')
        return {}
    mask_zy = _poly_to_mask(poly_zy, (std.shape[0], std.shape[2]))  # (Z, Y)

    # overlap along the shared Y axis -> (Z, X, Y)
    mask3d = (mask_zy.astype(bool)[:, None, :] &
              mask_xy.astype(bool)[None, :, :]).astype(np.uint8)

    coords = np.argwhere(mask3d)
    if len(coords) == 0:
        print(f'  X-Y and Z-Y ROIs for {name} do not overlap, skipping.')
        return {}
    z0, x0, y0 = coords.min(axis=0)
    z1, x1, y1 = coords.max(axis=0) + 1
    bounding_box = [int(z0), int(x0), int(y0),
                    int(z1 - z0), int(x1 - x0), int(y1 - y0)]

    return {'roi_xy': poly_xy, 'roi_zy': poly_zy, 'mask': mask3d,
            'bounding_box': bounding_box}


### SAVING ###

def _save_rois(folder: str,
               results: dict[str, dict],
               std: np.ndarray,
               is_3d: bool,
               voxel_size: np.ndarray | None) -> str:
    """
    Save (or update) the drawn ROIs to ``segment_rois.h5`` in `folder`.

    Existing segments with the same name are overwritten; the std image (and, for
    3D, the voxel size) is written once. Per-segment datasets: ``{name}_mask``,
    ``{name}_bounding_box``, and either ``{name}_roi`` (2D) or ``{name}_roi_xy`` /
    ``{name}_roi_zy`` (3D, with ``{name}_roi`` kept as an alias of the X-Y polygon
    for backward compatibility with `utils.load_segment_rois`).

    Parameters
    ----------
    folder : str
        Folder to write ``segment_rois.h5`` into.
    results : dict of str -> dict
        Per-segment ROI results from `_select_roi_2d` / `_select_roi_3d`.
    std : np.ndarray
        The std image the ROIs were drawn on.
    is_3d : bool
        Whether the data is 3D.
    voxel_size : np.ndarray or None
        [z, x, y] voxel size (saved for 3D data).

    Returns
    -------
    str
        Path to the written ``segment_rois.h5``.
    """

    h5_path = os.path.join(folder, 'segment_rois.h5')

    with h5py.File(h5_path, 'a') as f:
        # write the shared datasets once
        if 'std_image' not in f:
            f.create_dataset('std_image', data=std)
        if is_3d and voxel_size is not None and 'voxel_size' not in f:
            f.create_dataset('voxel_size', data=np.asarray(voxel_size))

        # union of existing and new segment names (order preserved)
        existing = []
        if 'segment_names' in f:
            existing = [n.decode('utf-8') if isinstance(n, bytes) else str(n)
                        for n in f['segment_names'][:]]
            del f['segment_names']
        names = existing + [n for n in results if n not in existing]
        f.create_dataset('segment_names', data=np.array(names, dtype='S'))

        # write each segment, replacing any datasets that already exist
        for name, res in results.items():
            for suffix in ('roi', 'roi_xy', 'roi_zy', 'mask', 'bounding_box'):
                key = f'{name}_{suffix}'
                if key in f:
                    del f[key]

            f.create_dataset(f'{name}_mask', data=res['mask'])
            f.create_dataset(f'{name}_bounding_box',
                             data=np.asarray(res['bounding_box']))
            if is_3d:
                f.create_dataset(f'{name}_roi_xy', data=res['roi_xy'])
                f.create_dataset(f'{name}_roi_zy', data=res['roi_zy'])
                # alias so utils.load_segment_rois still finds a `{name}_roi`
                f.create_dataset(f'{name}_roi', data=res['roi_xy'])
            else:
                f.create_dataset(f'{name}_roi', data=res['roi'])

    print(f'Saved {len(results)} ROI(s) to {h5_path}')
    return h5_path


### MAIN ###

def select_manual_roi(folder: str = '',
                      segment_names: str | list | tuple | None = None) -> dict:
    """
    Draw free-hand ROIs on a recording's std image and save them.

    The std image is read from ``projection_images.h5`` or
    ``projection_images_grouped.h5`` in `folder`. A 2D std gives one free-hand
    ROI per segment; a 3D std (SCAPE) gives a two-view ROI per segment (see the
    module docstring). Results are saved to ``segment_rois.h5`` in `folder`.

    Parameters
    ----------
    folder : str, optional
        Folder containing a projection-images h5. If empty, a directory-chooser
        dialog is opened.
    segment_names : str, list, tuple, or None, optional
        Segment names to draw. A comma-separated string (e.g.
        ``'RF_femur,RF_tibia'``) or a list/tuple of names. A standard set is
        referenced by name with the ``s.`` prefix (``'s.legs'``,
        ``'s.legs_front'``) and expanded. If None, the names are requested
        interactively.

    Returns
    -------
    dict
        Per-segment ROI results (see `_select_roi_2d` / `_select_roi_3d`).
    """

    # fail fast (before any dialogs / prompts) if drawing cannot be shown
    _ensure_interactive_backend()

    # pick the folder interactively if not provided
    if not folder:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(initialdir=MAIN_PATH,
                                         title='Select recording / analysis folder')
        root.destroy()

    assert folder, 'No folder selected.'
    assert os.path.isdir(folder), f'Folder {folder} does not exist'

    # ask for the segment names if not provided
    if segment_names is None:
        segment_names = input('Enter segment names separated by commas '
                              '(e.g. RF_femur,RF_tibia), or a standard set '
                              '(s.legs, s.legs_front): ')
    names = _parse_segment_names(segment_names)
    assert names, 'No segment names provided.'
    print(f'Segments to draw ({len(names)}): {", ".join(names)}')

    # load the std image the ROIs are drawn on
    h5_path = _find_proj_h5(folder)
    with h5py.File(h5_path, 'r') as f:
        assert 'std' in f, f'{os.path.basename(h5_path)} has no "std" dataset'
        std = f['std'][:]  # type: ignore[index]
    print(f'Loaded std image {std.shape} from {os.path.basename(h5_path)}')

    is_3d = std.ndim == 3
    assert std.ndim in (2, 3), f'std image must be 2D or 3D, got shape {std.shape}'

    voxel_size = _get_voxel_size(folder) if is_3d else None

    # pre-compute the 8-bit grayscale std for the 2D case (drawn on repeatedly)
    std_gray = None
    if not is_3d:
        std_gray = _autocontrast_8bit(std)

    # draw each segment's ROI
    results: dict[str, dict] = {}
    for name in names:
        if is_3d:
            res = _select_roi_3d(std, name, voxel_size)
        else:
            overlays = [r['roi'] for r in results.values()]
            res = _select_roi_2d(std_gray, name, overlays)
        if res:
            results[name] = res

    if not results:
        print('No ROIs drawn; nothing saved.')
        return {}

    _save_rois(folder, results, std, is_3d, voxel_size)
    return results


# make this module callable so it can be used in a notebook as
#   import select_manual_roi
#   select_manual_roi(folder, 's.legs')
import sys
sys.modules[__name__] = select_manual_roi


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Draw free-hand ROIs on a recording\'s std image.')
    parser.add_argument('--folder', type=str, default='',
                        help='Folder containing a projection-images h5 '
                             '(opens a dialog if omitted)')
    parser.add_argument('--segments', type=str, default=None,
                        help='Comma-separated segment names or a standard set '
                             '(e.g. "RF_femur,RF_tibia" or "s.legs")')
    args = parser.parse_args()
    select_manual_roi(args.folder, args.segments)
