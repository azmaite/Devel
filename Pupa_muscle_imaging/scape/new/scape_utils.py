import os
import json
import datetime
from pathlib import Path
import re

import h5py
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from scipy.spatial import ConvexHull
from stl import mesh as stl_mesh


def get_recording_list(main_path: str, mkv: bool=False) -> tuple[str, list[dict]]:
    """
    Prompt the user to select an experiment folder and return the analysis folder
    and a list of selected subfolder paths. 
    
    Parameters
    ----------
    main_path : str
        The main directory where experiment folders are located.
    mkv : bool, default False
        If True, only subfolders with top-level .mkv files are listed for selection,
        and the returned list contains the paths of those .mkv files. 
        If False, the returned list contains the paths of the selected subfolders."""

    paths_list_files = {True: 'scape_paths.json', False: 'mkv_paths.json'}

    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    exp_folder = filedialog.askdirectory(initialdir=main_path,
                                            title='Select experiment or analysis folder')

    assert os.path.exists(exp_folder), f'Experiment folder {exp_folder} does not exist'
    assert os.path.isdir(exp_folder), f'Experiment folder {exp_folder} must be a directory'

    # if the selected folder already contains a scape_paths.json or mkv_paths.json
    # it is an existing analysis folder: reuse it and its recorded data paths.
    # Otherwise create a new analysis folder (YYYY-MM-DD_analysis) and select recordings.
    json_path = os.path.join(exp_folder, paths_list_files[mkv])
    entry_name = {True: 'mkv_path', False: 'folder_path'}

    if os.path.exists(json_path):
        analysis_folder = exp_folder
        with open(json_path, 'r') as f:
            entries = json.load(f)
        print(f'Existing analysis folder detected: {analysis_folder}')
        print(f'Found {len(entries)} entries.')

        paths = [entry[entry_name[mkv]] for entry in entries]
    else:
        analysis_name = datetime.date.today().strftime('%Y-%m-%d') + '_analysis'
        analysis_folder = os.path.join(exp_folder, analysis_name)
        os.makedirs(analysis_folder, exist_ok=True)
        print(f'Analysis folder: {analysis_folder}')

        # let the user pick which subfolders of the input folder to include
        paths = _select_paths(exp_folder, paths_list_files[mkv], mkv=mkv)
        assert paths is not None, "No subfolders selected"
        if len(paths) == 0:
            print('No subfolders selected. Exiting.')
        
        else:
            print(f'Selected {len(paths)} subfolders.')

            # save the list of selected paths to scape_paths.json in the analysis folder
            
            entries = [{entry_name[mkv]: p} for p in paths]
            with open(os.path.join(analysis_folder, paths_list_files[mkv]), 'w') as f:
                json.dump(entries, f, indent=2)

    return analysis_folder, paths
    



def _select_paths(input_folder: str, paths_list_file: str, mkv: bool=False) -> list[str] | None:
    """
    List the input folder's subfolders and let the user pick which to include.

    The immediate subfolders of `input_folder` are listed in the terminal (if mkv=True,each
    with a number and its count of top-level .mkv files). Analysis folders (any
    subfolder containing a `*_paths.json`) (and subfolders with no top-level
    .mkv files if mkv=True) are excluded from the list. The user picks which to include with a
    comma/space-separated list of numbers and ranges (e.g. ``1,3,5-7``), ``0`` or
    ``all`` to include every listed subfolder, or ``q`` to cancel. 
    If mkv=True, the top-level .mkv files of every chosen subfolder are returned.
    If mkv=False, the paths of the chosen subfolders are returned directly.

    Parameters
    ----------
    input_folder : str
        Folder whose immediate subfolders are offered for selection.
    mkv : bool, default False
        If True, only subfolders with top-level .mkv files are listed.

    Returns
    -------
    list of str
        Ordered, de-duplicated list of selected file paths.
    """

    # immediate subfolders, excluding analysis folders (those with mkv_paths.json or scape_paths.json)
    if mkv:
        subfolders = sorted(
        p for p in Path(input_folder).iterdir()
        if p.is_dir() and not (p / paths_list_file).exists())
    else:
        # sort by the run number, padding if needed (e.g. run1, run2, run10) so that run10 is last
        def natural_key(path: Path):
            return [
                int(part) if part.isdigit() else part.lower()
                for part in re.split(r'(\d+)', path.name)
            ]

        subfolders = sorted(
            (
                p for p in Path(input_folder).iterdir()
                if p.is_dir() and not (p / paths_list_file).exists()
            ),
            key=natural_key,
        )

    # if mkv=True, keep only subfolders that actually contain top-level .mkv files
    mkv_per_folder = []
    if mkv:
        subfolders = []
        for sub in subfolders:
            mkvs = sorted(str(p) for p in sub.glob('*.mkv'))
            if mkvs:
                subfolders.append(sub)
                mkv_per_folder.append(mkvs)

    if len(subfolders) == 0 and mkv:
        print(f'No subfolders with .mkv files found in {input_folder}.')
        return []
    elif len(subfolders) == 0:
        print(f'No subfolders found in {input_folder}.')
        return []

    # list each subfolder (with its top-level .mkv count if mkv=True) and ask which to include - in 2 columns
    print(f'\nSubfolders in {input_folder}:')
    print('  [0] all')
    items = [
        f"[{i + 1}] {sub.name} ({len(mkvs)} .mkv)"
        if mkv else f"[{i + 1}] {sub.name}"
        for i, (sub, mkvs) in enumerate(zip(subfolders, mkv_per_folder))
    ] if mkv else [
        f"[{i + 1}] {sub.name}"
        for i, sub in enumerate(subfolders)
    ]

    width = max(map(len, items), default=0) + 4
    rows = (len(items) + 1) // 2

    for i in range(rows):
        left = items[i]
        right = items[i + rows] if i + rows < len(items) else ""
        print(f"  {left:<{width}}{right}")


    # ask which subfolders to include
    selection = input("\nWhich subfolders to include? "
                      "e.g. '1,3,5-7', '0' or 'all', or 'q' to cancel: ").strip().lower()
    if selection in ('q', 'quit', ''):
        return []
    if selection == 'all' or '0' in selection.replace(',', ' ').split():
        chosen = list(range(len(subfolders)))
    else:
        chosen = _parse_selection(selection, len(subfolders))

    # collect top-level .mkv files from every chosen subfolder (if mkv=True) or return the chosen subfolder paths (if mkv=False)
    paths: list[str] = []
    for idx in chosen:
        if mkv:
            paths.extend(mkv_per_folder[idx])
        else:
            paths.append(str(subfolders[idx]))

    return paths


def _parse_selection(selection: str, n: int) -> list[int]:
    """
    Parse a selection string like ``1,3,5-7`` into sorted 0-based indices.

    Numbers and ``a-b`` ranges may be separated by commas and/or spaces. Values
    are 1-based on input; anything out of ``1..n`` or unparseable is reported and
    ignored.

    Parameters
    ----------
    selection : str
        The user's selection string.
    n : int
        Number of listed items (valid 1-based indices are ``1..n``).

    Returns
    -------
    list of int
        Sorted, de-duplicated 0-based indices.
    """

    indices: set[int] = set()
    for tok in selection.replace(',', ' ').split():
        if '-' in tok:
            parts = tok.split('-')
            if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
                print(f"  Ignoring invalid range '{tok}'.")
                continue
            a, b = int(parts[0]), int(parts[1])
            for k in range(min(a, b), max(a, b) + 1):
                if 1 <= k <= n:
                    indices.add(k - 1)
                else:
                    print(f'  Ignoring out-of-range index {k}.')
        elif tok.isdigit():
            k = int(tok)
            if 1 <= k <= n:
                indices.add(k - 1)
            else:
                print(f'  Ignoring out-of-range index {k}.')
        else:
            print(f"  Ignoring invalid entry '{tok}'.")

    return sorted(indices)


_LEG_CODES = {
    'LF': ('Left', 'Front'), 'LM': ('Left', 'Mid'), 'LH': ('Left', 'Hind'),
    'RF': ('Right', 'Front'), 'RM': ('Right', 'Mid'), 'RH': ('Right', 'Hind'),
}
_LEG_MESH_NAME_RE = re.compile(r'^(?P<leg>LF|LM|LH|RF|RM|RH)_(?P<segment>\w+)-(?P<id>\d+)\.stl$')


def _rotate_points(points: np.ndarray, rx: float, ry: float, rz: float) -> np.ndarray:
    """
    Apply an extrinsic XYZ rotation to an array of 3D points.

    Parameters
    ----------
    points : np.ndarray
        Array of shape `(N, 3)`.
    rx : float
        Rotation around the first axis, in degrees.
    ry : float
        Rotation around the second axis, in degrees.
    rz : float
        Rotation around the third axis, in degrees.

    Returns
    -------
    np.ndarray
        Rotated points of shape `(N, 3)`.
    """
    cx, sx = np.cos(np.radians(rx)), np.sin(np.radians(rx))
    cy, sy = np.cos(np.radians(ry)), np.sin(np.radians(ry))
    cz, sz = np.cos(np.radians(rz)), np.sin(np.radians(rz))

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    return points @ Rx.T @ Ry.T @ Rz.T


def compute_mesh_center(meshes: dict) -> np.ndarray:
    """
    Compute the centroid of all mesh points combined (equal per-point weight).

    Parameters
    ----------
    meshes : dict of str to dict
        As returned by `load_leg_meshes`.

    Returns
    -------
    np.ndarray
        Combined centroid, shape `(3,)`.
    """
    all_points = np.concatenate([m['points'] for m in meshes.values()], axis=0)
    return all_points.mean(axis=0)


def compute_view_dims() -> dict[str, tuple]:
    """
    The volume-axis indices kept by each of the 3 dropped-axis projections.

    Returns
    -------
    dict of str to tuple
        `'z'`, `'x'`, `'y'` (the dropped volume axis) each mapped to the 2
        remaining volume-axis indices `(d0, d1)`: `d0` is plotted vertically,
        `d1` horizontally, matching `imshow`'s default row/col convention.
    """
    drop_axis = {'z': 0, 'x': 1, 'y': 2}
    return {key: tuple(i for i in range(3) if i != idx) for key, idx in drop_axis.items()}


def compute_volume_projections(volume: np.ndarray, voxel_size_zxy: tuple | None = None) -> dict:
    """
    Compute maximum-intensity projections of `volume` along each axis, plus
    the display aspect ratio and contrast limits needed to show them
    undistorted (SCAPE voxels aren't cubic).

    Parameters
    ----------
    volume : np.ndarray
        3D reference volume with axes `(z, x, y)`.
    voxel_size_zxy : tuple or None, default None
        Physical voxel size along `(z, x, y)`; if None, cubic voxels (aspect
        1 everywhere) are assumed.

    Returns
    -------
    dict
        `'projections'` (dict of view to 2D array), `'aspect'` (dict of view
        to float), `'vmin'`, `'vmax'` (display contrast limits), `'view_dims'`
        (dict of view to `(d0, d1)` volume-axis indices, from
        `compute_view_dims`).
    """
    if voxel_size_zxy is None:
        voxel_size_zxy = (1.0, 1.0, 1.0)
    drop_axis = {'z': 0, 'x': 1, 'y': 2}
    view_dims = compute_view_dims()
    projections = {key: np.max(volume, axis=idx) for key, idx in drop_axis.items()}
    aspect = {key: voxel_size_zxy[d0] / voxel_size_zxy[d1] for key, (d0, d1) in view_dims.items()}

    # deskewed volumes are commonly zero-padded outside the valid skew
    # parallelogram, so exclude zeros when picking display contrast limits
    nonzero = volume[volume > 0]
    vmin, vmax = (tuple(np.percentile(nonzero, [0.5, 99.9])) if nonzero.size > 0
                  else (float(volume.min()), float(volume.max())))

    return {'projections': projections, 'aspect': aspect, 'vmin': vmin, 'vmax': vmax, 'view_dims': view_dims}


def transform_points(points: np.ndarray,
                      scale: float, rx: float, ry: float, rz: float, shift: np.ndarray,
                      mesh_center: np.ndarray, voxel_size_zxy: np.ndarray, volume_shape: tuple) -> np.ndarray:
    """
    Apply the rigid alignment transform to mesh-space `points`.

    Centers `points` on `mesh_center`, then scales and rotates them directly
    (the mesh's own coordinates are physically isotropic, so no per-axis
    correction is needed for this part), then converts into the volume's own
    (anisotropic) voxel-index space by dividing by `voxel_size_zxy`, before
    adding the volume's center and `shift`.

    Parameters
    ----------
    points : np.ndarray
        Mesh-space points, shape `(n, 3)`.
    scale : float
        Uniform scale factor.
    rx : float
        Rotation around the first axis, in degrees (extrinsic XYZ), applied
        around `mesh_center`.
    ry : float
        Rotation around the second axis, in degrees.
    rz : float
        Rotation around the third axis, in degrees.
    shift : np.ndarray
        Shift in volume voxel-index units, shape `(3,)`, relative to the
        volume's center.
    mesh_center : np.ndarray
        Combined mesh centroid, shape `(3,)`, e.g. from `compute_mesh_center`.
    voxel_size_zxy : np.ndarray
        Physical voxel size along `(z, x, y)`, shape `(3,)`.
    volume_shape : tuple
        Shape of the reference volume, `(z, x, y)`.

    Returns
    -------
    np.ndarray
        Transformed points in volume voxel-index space, shape `(n, 3)`.
    """
    centered = points - mesh_center
    rotated = _rotate_points(centered * scale, rx, ry, rz)
    volume_center = np.array(volume_shape, dtype=np.float64) / 2
    return rotated / voxel_size_zxy + volume_center + shift


def project_points(points_3d: np.ndarray, view: str, view_dims: dict) -> np.ndarray:
    """
    Project 3D points to 2D `(col, row)` coordinates matching `imshow`'s
    convention for the given view.

    Parameters
    ----------
    points_3d : np.ndarray
        Points in volume voxel-index space, shape `(n, 3)`.
    view : 'z', 'x' or 'y'
        Which volume axis was dropped to form this projection.
    view_dims : dict of str to tuple
        Mapping of view key to the `(d0, d1)` volume-axis indices it keeps,
        e.g. from `compute_view_dims`.

    Returns
    -------
    np.ndarray
        2D points, shape `(n, 2)`.
    """
    d0, d1 = view_dims[view]
    return np.column_stack((points_3d[:, d1], points_3d[:, d0]))


def _leg_segment_colors() -> dict[tuple[str, str], tuple]:
    """
    Build a color map for leg segments, with a light-to-dark gradient from
    proximal (`coxa`) to distal (`tarsus`) within each leg position. The same
    color is used for the left and right copy of a given position/segment.

    Returns
    -------
    dict of (str, str) to tuple
        Mapping of `(position, segment)` (e.g. `('Front', 'femur')`) to an
        RGB color tuple.
    """
    segment_order = ['coxa', 'trochanter', 'femur', 'tibia', 'tarsus']
    factors = np.linspace(0.0, 0.6, len(segment_order))
    base_colors = {'Front': 'firebrick', 'Mid': 'yellowgreen', 'Hind': 'dodgerblue'}

    colors = {}
    for position, base_name in base_colors.items():
        base_rgb = np.array(mcolors.to_rgb(base_name))
        for segment, factor in zip(segment_order, factors):
            colors[(position, segment)] = tuple(base_rgb + (1 - base_rgb) * factor)

    return colors


def load_leg_meshes(cuticle_mesh_path: str) -> dict[str, dict]:
    """
    Load leg-segment STL meshes and reduce each to its convex hull's vertices.

    Each raw STL mesh has tens of thousands of triangles (a full CT-scan
    surface), far more detail than needed for rigid alignment guidance and
    too many to transform interactively. Each segment's mesh is therefore
    reduced to the (much smaller) set of vertices on its convex hull, which
    is all that's needed to draw a 2D projected outline.

    The hind legs and every leg's tarsus are skipped entirely (not loaded),
    since they're not shown in the alignment GUI.

    Parameters
    ----------
    cuticle_mesh_path : str
        Path to the folder containing a `meshes` subfolder of `.stl` files
        named like `RF_tibia-1.stl` (leg code + segment name + segment id).

    Returns
    -------
    dict of str to dict
        Keyed by display name (e.g. `'Right Front tibia'`). Each value has
        keys `'points'` (array of shape `(n, 3)`, the hull's vertices),
        `'color'` (RGB tuple), `'side'`, `'position'`, `'segment'`.
    """
    colors = _leg_segment_colors()

    stl_files = sorted(Path(cuticle_mesh_path).glob('meshes/*.stl'))
    assert len(stl_files) > 0, f"No .stl files found in {Path(cuticle_mesh_path) / 'meshes'}"

    meshes = {}
    for stl_path in stl_files:
        match = _LEG_MESH_NAME_RE.match(stl_path.name)
        assert match is not None, f"Filename '{stl_path.name}' does not match the expected '{{LEG}}_{{segment}}-{{id}}.stl' pattern"

        side, position = _LEG_CODES[match['leg']]
        segment = match['segment']
        if position == 'Hind' or segment == 'tarsus':
            continue
        name = f'{side} {position} {segment}'

        raw_points = stl_mesh.Mesh.from_file(str(stl_path)).vectors.reshape(-1, 3).astype(np.float64)
        hull = ConvexHull(raw_points)
        hull_points = raw_points[hull.vertices].astype(np.float32)

        meshes[name] = {
            'points': hull_points,
            'color': colors[(position, segment)],
            'side': side,
            'position': position,
            'segment': segment,
        }

    print(f'Loaded {len(meshes)} leg segment meshes from {cuticle_mesh_path}')
    return meshes


def align_meshes_to_volume(volume: np.ndarray,
                            meshes: dict,
                            voxel_size_zxy: tuple | None = None,
                            save_path: str | Path | None = None) -> dict:
    """
    Launch an interactive GUI to align leg-segment meshes to a reference volume.

    Displays the reference volume as 3 orthogonal maximum-intensity
    projections (dropping each axis of `volume` in turn), with every leg
    segment's outline (the 2D convex hull of its projected points) drawn as a
    translucent patch on all 3 views at once. Sliders control a single rigid
    transform (uniform scale, XYZ rotation, XYZ shift) applied to all meshes
    together, debounced so redraws only happen once slider dragging pauses.
    Rotation pivots around the combined mesh's own center of mass (done in
    physical, voxel-size-corrected space so anisotropic voxels don't distort
    the mesh), and a shift of `(0, 0, 0)` places that center of mass in the
    middle of the volume, so the shift sliders read out as an offset from
    center. Rotation starts, and is centered, at `(0, 90, 90)` degrees.

    If `save_path` already exists, its stored parameters are used as initial
    slider values (wrapped into the same fixed slider range via their 360°
    periodicity). On closing the window, the final parameters are saved
    there.

    Parameters
    ----------
    volume : np.ndarray
        3D reference volume with axes `(z, x, y)`, e.g. a deskewed SCAPE
        baseline-intensity volume.
    meshes : dict of str to dict
        As returned by `load_leg_meshes`.
    voxel_size_zxy : tuple or None, default None
        Physical voxel size along `(z, x, y)`, e.g. from
        `scapeio.get_voxel_size(rec_path, tube_lens, deskewed=True)`. Since
        SCAPE voxels are rarely cubic, each projection needs its own display
        aspect ratio to avoid stretching the volume; if None, an aspect of 1
        (cubic voxels) is assumed.
    save_path : str, Path or None, default None
        Where to save/load the final alignment parameters (as an `.h5`
        file). If None, parameters are not saved.

    Returns
    -------
    dict
        Final parameters with keys `'scale'` (float), `'rotation_deg_xyz'`
        (tuple of 3 floats) and `'shift_xyz'` (tuple of 3 floats).
    """
    save_path = Path(save_path) if save_path is not None else None
    if voxel_size_zxy is None:
        voxel_size_zxy = (1.0, 1.0, 1.0)
    voxel_size_arr = np.array(voxel_size_zxy, dtype=np.float64)

    view_info = compute_volume_projections(volume, voxel_size_zxy)
    projections, view_aspect = view_info['projections'], view_info['aspect']
    vmin, vmax, view_dims = view_info['vmin'], view_info['vmax'], view_info['view_dims']
    view_titles = {'z': 'Z-projection', 'x': 'X-projection', 'y': 'Y-projection'}

    # combine all mesh points into one array so the per-slider-change rigid
    # transform is a single matmul rather than one per segment
    mesh_names = list(meshes.keys())
    point_counts = [meshes[name]['points'].shape[0] for name in mesh_names]
    point_offsets = np.cumsum([0] + point_counts)
    all_points = np.concatenate([meshes[name]['points'] for name in mesh_names], axis=0).astype(np.float64)

    # center of mass of the combined mesh: rotation pivots around it (so
    # sliders spin the assembly in place instead of orbiting it around an
    # arbitrary point), and a shift of (0, 0, 0) puts it in the middle of the
    # volume (and so of every projection)
    mesh_center = compute_mesh_center(meshes)

    def _outline(points_2d: np.ndarray) -> np.ndarray:
        """2D convex hull outline (closed polygon vertices) of `points_2d`."""
        hull = ConvexHull(points_2d)
        return points_2d[hull.vertices]

    # data-driven initial scale: fit the combined mesh bounding box into the
    # volume's bounding box, since the CT-scan mesh units have no known
    # correspondence to the SCAPE volume's voxel units
    mesh_diag = np.linalg.norm(all_points.max(axis=0) - all_points.min(axis=0))
    volume_diag = np.linalg.norm(volume.shape)
    init_scale = 0.5 * volume_diag / mesh_diag if mesh_diag > 0 else 1.0
    # default orientation: the rotation sliders are also centered on these values
    default_rx, default_ry, default_rz = 0.0, 90.0, 90.0
    init_rx, init_ry, init_rz = default_rx, default_ry, default_rz
    init_shift = np.zeros(3)

    def _wrap_to_range(angle: float, center: float) -> float:
        """Wrap `angle` (degrees) into `[center - 180, center + 180)` using its 360° periodicity."""
        return ((angle - center + 180) % 360) - 180 + center

    if save_path is not None and save_path.exists():
        print(f'Found existing alignment: {save_path}. Loading as starting values.')
        with h5py.File(save_path, 'r') as f:
            init_scale = float(f['scale'][()])
            loaded_rx, loaded_ry, loaded_rz = [float(v) for v in f['rotation_deg_xyz'][:]]
            init_rx = _wrap_to_range(loaded_rx, default_rx)
            init_ry = _wrap_to_range(loaded_ry, default_ry)
            init_rz = _wrap_to_range(loaded_rz, default_rz)
            init_shift = np.array([float(v) for v in f['shift_xyz'][:]])

    # build figure: 3 projection views on top, sliders and buttons below
    fig, view_axes_list = plt.subplots(1, 3, figsize=(13, 9))
    plt.subplots_adjust(top=0.88, bottom=0.30, left=0.05, right=0.97, wspace=0.15)
    view_axes = dict(zip(view_dims.keys(), view_axes_list))
    axis_names = {0: 'Z', 1: 'X', 2: 'Y'}
    for key, ax in view_axes.items():
        ax.imshow(projections[key], cmap='gray', vmin=vmin, vmax=vmax, aspect=view_aspect[key], origin='lower')
        ax.set_title(view_titles[key])
        d0, d1 = view_dims[key]
        ax.set_ylabel(axis_names[d0])
        ax.set_xlabel(axis_names[d1])
    fig.suptitle("Align leg meshes to volume.  Close window or click 'Done' to finish.", y=0.99)

    outline_patches = {key: {} for key in view_dims}
    for key, ax in view_axes.items():
        for name in mesh_names:
            patch = mpatches.Polygon(np.zeros((3, 2)), closed=True,
                                      facecolor=meshes[name]['color'], edgecolor=meshes[name]['color'],
                                      alpha=0.2, linewidth=1.2)
            ax.add_patch(patch)
            outline_patches[key][name] = patch

    def _update_outlines():
        """Recompute every mesh's projected outline in all 3 views from the current slider values."""
        # shift sliders are in (X, Y, Z) order, but transform_points needs (Z, X, Y) to match volume axes
        shift = np.array([s_sz.val, s_sx.val, s_sy.val])
        transformed = transform_points(all_points, s_scale.val, s_rx.val, s_ry.val, s_rz.val, shift,
                                        mesh_center, voxel_size_arr, volume.shape)
        for key in view_dims:
            projected = project_points(transformed, key, view_dims)
            for i, name in enumerate(mesh_names):
                start, end = point_offsets[i], point_offsets[i + 1]
                outline_patches[key][name].set_xy(_outline(projected[start:end]))
        fig.canvas.draw_idle()

    # condensed legend: one entry per leg position, not all segments; placed
    # outside the plots, in the top margin
    legend_colors = {}
    for name in mesh_names:
        legend_colors.setdefault(meshes[name]['position'], meshes[name]['color'])
    legend_handles = [
        plt.Line2D([0], [0], marker='s', linestyle='none', color=color, label=position)
        for position, color in legend_colors.items()
    ]
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.93),
               ncol=len(legend_handles), fontsize=9)

    # sliders: scale + rotations on the left, shifts on the right, all below the projections
    row_y = [0.24, 0.185, 0.13, 0.075]
    height = 0.03
    left, left_width = 0.10, 0.30
    right, right_width = 0.58, 0.30
    s_scale = Slider(plt.axes([left, row_y[0], left_width, height]), 'Scale',
                      0.05 * init_scale, 5 * init_scale, valinit=init_scale, valfmt='%.3g')
    s_rx = Slider(plt.axes([left, row_y[1], left_width, height]), 'Rot X',
                  default_rx - 180, default_rx + 180, valinit=init_rx, valstep=1, valfmt='%d°')
    s_ry = Slider(plt.axes([left, row_y[2], left_width, height]), 'Rot Z',
                  default_ry - 180, default_ry + 180, valinit=init_ry, valstep=1, valfmt='%d°')
    s_rz = Slider(plt.axes([left, row_y[3], left_width, height]), 'Rot Y',
                  default_rz - 180, default_rz + 180, valinit=init_rz, valstep=1, valfmt='%d°')
    # volume axes are (Z, X, Y), so X and Y's extents come from shape[1] and shape[2]
    s_sx = Slider(plt.axes([right, row_y[0], right_width, height]), 'Shift X',
                  -volume.shape[1], volume.shape[1], valinit=init_shift[0])
    s_sy = Slider(plt.axes([right, row_y[1], right_width, height]), 'Shift Y',
                  -volume.shape[2], volume.shape[2], valinit=init_shift[1])
    s_sz = Slider(plt.axes([right, row_y[2], right_width, height]), 'Shift Z',
                  -volume.shape[0], volume.shape[0], valinit=init_shift[2])

    sliders = [s_scale, s_rx, s_ry, s_rz, s_sx, s_sy, s_sz]

    # debounce: recompute outlines only after slider dragging pauses
    timer = fig.canvas.new_timer(interval=50)
    timer.single_shot = True
    timer.add_callback(_update_outlines)

    def _on_slider_change(val):
        timer.stop()
        timer.start()

    for s in sliders:
        s.on_changed(_on_slider_change)

    _update_outlines()

    # buttons: centered below the two slider columns
    first_button_x, button_y = 0.37, 0.02
    btn_reset = Button(plt.axes([first_button_x, button_y, 0.06, 0.04]), 'Reset', color='lightgray', hovercolor='gray')
    btn_hide = Button(plt.axes([first_button_x + 0.07, button_y, 0.07, 0.04]), 'Hide meshes', color='lightblue', hovercolor='dodgerblue')
    btn_done = Button(plt.axes([first_button_x + 0.15, button_y, 0.06, 0.04]), 'Done', color='#90EE90', hovercolor='#32CD32')

    def _reset(event):
        for s in sliders:
            s.reset()
    btn_reset.on_clicked(_reset)

    mesh_visible = {'state': True}

    def _toggle_hide(event):
        mesh_visible['state'] = not mesh_visible['state']
        for view_patches in outline_patches.values():
            for patch in view_patches.values():
                patch.set_visible(mesh_visible['state'])
        btn_hide.label.set_text('Hide meshes' if mesh_visible['state'] else 'Show meshes')
        fig.canvas.draw_idle()
    btn_hide.on_clicked(_toggle_hide)

    def _done(event):
        plt.close(fig)
    btn_done.on_clicked(_done)

    plt.show()  # blocks until window is closed (via Done button or window X)

    params = {
        'scale': float(s_scale.val),
        'rotation_deg_xyz': (float(s_rx.val), float(s_ry.val), float(s_rz.val)),
        'shift_xyz': (float(s_sx.val), float(s_sy.val), float(s_sz.val)),
    }

    if save_path is not None:
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('scale', data=params['scale'])
            f.create_dataset('rotation_deg_xyz', data=params['rotation_deg_xyz'])
            f.create_dataset('shift_xyz', data=params['shift_xyz'])
        print(f'Saved alignment parameters to {save_path}')

    return params
