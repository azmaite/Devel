"""
Split SCAPE recordings into their two hemibodies, as isotropic volumes, and save them as tifs.

Starts from the projection images of a recording (`{sample#_run#}/analysis/projection_images.h5`,
with the projections in (Z, X, Y) order and `voxel_sizes_xyz`), which are computed in the `scape`
folder (`scape/MANUAL/preprocess_scape_exp.py`). For each `sample#_run#` folder given, this:
- loads the projection (`min_proj` by default) and resamples it to isotropic voxels
  (`TARGET_VOXEL_SIZE` = 1.5 um by default), so that a tilted mirror plane is a proper reflection,
- fits the (slightly tilted) mirror-symmetry plane of the whole volume, rotates the volume so the
  plane is perpendicular to axis 2, and splits it there. The axis-2 >= plane side is flipped along
  axis 2, so that both hemibodies have the same orientation,
- saves the two hemibodies as `{experiment}_{run}_L.ome.tif` and `..._R.ome.tif` (float32, array
  order (Z, X, Y) as in `projection_images.h5` and `SPAM_registration/spam_pipeline.ipynb`, with
  the isotropic voxel size in the OME metadata), plus a json with the fitted plane and a png
  showing the volume (red) over its mirror image (green), to check the plane.

The plane fitting and splitting are the same as in `spam_pipeline.ipynb`.

Run from the terminal (`python split_hemibodies.py run_folder [run_folder ...] -o out_folder`) or
import and call `split_recording`.

resample_to_isotropic: resample a (Z, X, Y) volume to isotropic voxels.
fit_symmetry_plane: fit the mirror-symmetry plane of an isotropic volume.
mirror_volume: reflect a volume across a fitted plane.
split_hemibodies: rotate a volume to its plane and split it into the two hemibodies.
split_recording: do all of the above for one recording folder and save the results.
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.ndimage import affine_transform, zoom
from scipy.optimize import minimize

TARGET_VOXEL_SIZE = 1.5  # um, isotropic (keeps the volumes at a manageable size for registration)


def resample_to_isotropic(volume: np.ndarray,
                          voxel_sizes_xyz,
                          target_voxel_size: float,
                          order: int = 1) -> np.ndarray:
    """
    Resample a volume stored in (Z, X, Y) order (as in `projection_images.h5`) to isotropic voxels.

    Parameters
    ----------
    volume : np.ndarray
        Volume, shape (Z, X, Y).
    voxel_sizes_xyz : array-like of float
        Voxel size of `volume`, in (x, y, z) order (as `voxel_sizes_xyz` in the h5).
    target_voxel_size : float
        Isotropic voxel size of the output, same unit.
    order : int
        Spline order for `scipy.ndimage.zoom` (1 for images, 0 for labels).

    Returns
    -------
    np.ndarray
        Resampled volume.
    """
    zoom_factors = (voxel_sizes_xyz[2] / target_voxel_size,
                    voxel_sizes_xyz[0] / target_voxel_size,
                    voxel_sizes_xyz[1] / target_voxel_size)
    return zoom(volume, zoom_factors, order=order)


def _plane_normal(theta_1: float,
                  theta_2: float) -> np.ndarray:
    """
    Unit normal of a plane close to perpendicular to axis 2, tilted by `theta_1` towards axis 0 and
    `theta_2` towards axis 1 (radians).

    Parameters
    ----------
    theta_1, theta_2 : float
        Tilt angles, in radians.

    Returns
    -------
    np.ndarray
        Unit normal, shape (3,), in array axis order.
    """
    normal = np.array([np.sin(theta_1), np.sin(theta_2), 1.0])
    return normal / np.linalg.norm(normal)


def mirror_volume(volume: np.ndarray,
                  theta_1: float,
                  theta_2: float,
                  offset_axis2: float,
                  cval: float = 0.0) -> np.ndarray:
    """
    Reflect an isotropic volume across the plane (`theta_1`, `theta_2`, `offset_axis2`).

    The plane goes through the point (axis-0 centre, axis-1 centre, `offset_axis2`).

    Parameters
    ----------
    volume : np.ndarray
        Volume (isotropic voxels).
    theta_1, theta_2 : float
        Tilt angles of the plane, in radians (see `_plane_normal`).
    offset_axis2 : float
        Axis-2 position of the plane at the axis-0/axis-1 centre.
    cval : float
        Value for voxels whose reflection falls outside the volume (`np.nan` to exclude them).

    Returns
    -------
    np.ndarray
        Reflected volume.
    """
    normal = _plane_normal(theta_1, theta_2)
    point = np.array([(volume.shape[0] - 1) / 2, (volume.shape[1] - 1) / 2, offset_axis2])
    matrix = np.eye(3) - 2 * np.outer(normal, normal)
    offset = point - matrix @ point
    return affine_transform(volume, matrix=matrix, offset=offset, order=1, mode='constant',
                            cval=cval)


def _symmetry_objective(params: np.ndarray,
                        volume: np.ndarray) -> float:
    """
    Negative normalised cross-correlation of `volume` with its reflection across the plane
    `params` = (`theta_1`, `theta_2`, `offset_axis2`), ignoring voxels reflected from outside.

    Parameters
    ----------
    params : np.ndarray
        Plane parameters.
    volume : np.ndarray
        Volume being fitted.

    Returns
    -------
    float
        Negative NCC (0 if there is no overlap).
    """
    mirrored = mirror_volume(volume, *params, cval=np.nan)
    valid = ~np.isnan(mirrored)
    if not np.any(valid):
        return 0.0
    a = volume[valid] - volume[valid].mean()
    b = mirrored[valid] - mirrored[valid].mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return -float((a * b).sum() / denom) if denom > 0 else 0.0


def fit_symmetry_plane(volume: np.ndarray,
                       downsample_factor: float = 0.2,
                       angle_step_deg: float = 5.0,
                       offset_step: float = 20.0) -> tuple:
    """
    Fit the mirror-symmetry plane of an isotropic volume by maximising the correlation of the
    volume with its reflection (Nelder-Mead, on a downsampled copy).

    Starts from the plane perpendicular to axis 2 through the axis-2 middle, with an explicit
    initial simplex (scipy's default barely moves parameters that start at 0).

    Parameters
    ----------
    volume : np.ndarray
        Volume (isotropic voxels).
    downsample_factor : float
        Zoom factor of the copy used for fitting.
    angle_step_deg : float
        Initial simplex step of the angles, in degrees.
    offset_step : float
        Initial simplex step of the offset, in full-resolution voxels.

    Returns
    -------
    tuple
        (`theta_1`, `theta_2`, `offset_axis2`, `ncc`), angles in radians, offset in voxels of
        `volume`.
    """
    fit_volume = zoom(volume, downsample_factor, order=1) if downsample_factor != 1 else volume
    initial = np.array([0.0, 0.0, fit_volume.shape[2] / 2])
    steps = np.diag([np.radians(angle_step_deg), np.radians(angle_step_deg),
                     offset_step * downsample_factor])
    result = minimize(_symmetry_objective, initial, args=(fit_volume,), method='Nelder-Mead',
                      options={'initial_simplex': np.vstack([initial, initial + steps])})
    theta_1, theta_2, offset_axis2 = result.x
    return float(theta_1), float(theta_2), float(offset_axis2 / downsample_factor), -result.fun


def split_hemibodies(volume: np.ndarray,
                     theta_1: float,
                     theta_2: float,
                     offset_axis2: float,
                     order: int = 1) -> tuple:
    """
    Rotate a volume so its symmetry plane is perpendicular to axis 2, and split it there.

    The rotation takes the plane's normal onto axis 2, about the plane's crossing with the
    axis-0/axis-1 centre line; the output grows so nothing is cut off (new voxels are 0).

    Parameters
    ----------
    volume : np.ndarray
        Volume (isotropic voxels).
    theta_1, theta_2, offset_axis2 : float
        The plane (see `fit_symmetry_plane`).
    order : int
        Spline order (1 for images, 0 for labels).

    Returns
    -------
    tuple of np.ndarray
        (`low_side`, `high_side`): the axis-2 < plane side as is ('L'), and the other side flipped
        along axis 2 ('R'), so that both have the same orientation.
    """
    normal = _plane_normal(theta_1, theta_2)
    axis2 = np.array([0.0, 0.0, 1.0])
    v = np.cross(normal, axis2)
    s, c = np.linalg.norm(v), normal @ axis2
    if s < 1e-12:
        rotation = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation = np.eye(3) + vx + vx @ vx * (1 - c) / s ** 2

    center_in = np.array([(volume.shape[0] - 1) / 2, (volume.shape[1] - 1) / 2, offset_axis2])
    corners = np.array([[i, j, k] for i in (0, volume.shape[0] - 1)
                        for j in (0, volume.shape[1] - 1)
                        for k in (0, volume.shape[2] - 1)], dtype=float)
    rotated_corners = (corners - center_in) @ rotation.T
    lower, upper = rotated_corners.min(axis=0), rotated_corners.max(axis=0)
    center_out = -lower
    out_shape = tuple(int(np.ceil(u - l)) + 1 for l, u in zip(lower, upper))
    matrix = rotation.T
    rotated = affine_transform(volume, matrix=matrix, offset=center_in - matrix @ center_out,
                               output_shape=out_shape, order=order, mode='constant', cval=0)
    split_index = int(round(center_out[2]))
    return rotated[:, :, :split_index], np.flip(rotated[:, :, split_index:], axis=2).copy()


def split_recording(scape_path: str | Path,
                    out_folder: str | Path,
                    proj_type: str = 'min_proj',
                    target_voxel_size: float = TARGET_VOXEL_SIZE,
                    overwrite: bool = False) -> tuple:
    """
    Split one recording into its two isotropic hemibodies and save them (see the module docstring).

    Parameters
    ----------
    scape_path : str | Path
        A `sample#_run#` folder with its projections computed (`analysis/projection_images.h5`,
        see the module docstring).
    out_folder : str | Path
        Folder to save the hemibodies in (created if needed).
    proj_type : 'min_proj', 'max_proj' or 'std_image'
        Projection to split.
    target_voxel_size : float
        Isotropic voxel size, in um.
    overwrite : bool
        If False, recordings whose two hemibody tifs already exist are skipped.

    Returns
    -------
    tuple of Path
        Paths of the L and R tifs.
    """
    scape_path, out_folder = Path(scape_path), Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    name = f'{scape_path.parent.name}_{scape_path.name}'
    paths = tuple(out_folder / f'{name}_{side}.ome.tif' for side in ('L', 'R'))
    if all(p.exists() for p in paths) and not overwrite:
        print(f'Hemibodies already found for {name}, skipping.')
        return paths

    h5_path = scape_path / 'analysis' / 'projection_images.h5'
    if not h5_path.exists():
        raise FileNotFoundError(f'{h5_path} not found: compute the projections first with '
                                f'scape/MANUAL/preprocess_scape_exp.py.')
    with h5py.File(h5_path, 'r') as f:
        projection = f[proj_type][:].astype(np.float32)
        voxel_sizes_xyz = f['voxel_sizes_xyz'][:]

    iso = resample_to_isotropic(projection, voxel_sizes_xyz, target_voxel_size)
    theta_1, theta_2, offset_axis2, ncc = fit_symmetry_plane(iso)
    print(f'{name}: {proj_type} {projection.shape} -> isotropic {iso.shape}; plane tilt '
          f'{np.degrees(theta_1):.2f} deg (axis 0), {np.degrees(theta_2):.2f} deg (axis 1), '
          f'offset {offset_axis2:.1f}, NCC {ncc:.4f}')
    hemibodies = split_hemibodies(iso, theta_1, theta_2, offset_axis2)

    metadata = {'axes': 'ZYX',  # array order is (Z, X, Y) as in projection_images.h5
                'PhysicalSizeX': target_voxel_size, 'PhysicalSizeXUnit': 'um',
                'PhysicalSizeY': target_voxel_size, 'PhysicalSizeYUnit': 'um',
                'PhysicalSizeZ': target_voxel_size, 'PhysicalSizeZUnit': 'um'}
    for path, hemibody in zip(paths, hemibodies):
        tifffile.imwrite(path, hemibody.astype(np.float32), ome=True, metadata=metadata)
        print(f'Saved {path} {hemibody.shape}')

    with open(out_folder / f'{name}_split.json', 'w') as f:
        json.dump({'recording': str(scape_path), 'projection': proj_type,
                   'array_axes': 'ZXY (as projection_images.h5)',
                   'voxel_size_um': target_voxel_size,
                   'original_voxel_sizes_xyz_um': [float(v) for v in voxel_sizes_xyz],
                   'isotropic_shape': list(iso.shape),
                   'theta_1_rad': theta_1, 'theta_2_rad': theta_2, 'offset_axis2': offset_axis2,
                   'symmetry_ncc': ncc,
                   'L_shape': list(hemibodies[0].shape), 'R_shape': list(hemibodies[1].shape),
                   'R_flipped_along_axis2': True}, f, indent=2)

    mirrored = mirror_volume(iso, theta_1, theta_2, offset_axis2)
    red, green = np.max(iso, axis=0), np.max(mirrored, axis=0)
    rgb = np.zeros((*red.shape, 3))
    rgb[..., 0] = np.clip(red / np.percentile(red, 99.9), 0, 1)
    rgb[..., 1] = np.clip(green / np.percentile(green, 99.9), 0, 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), gridspec_kw={'width_ratios': [2, 1, 1]})
    axes[0].imshow(rgb)
    axes[0].set_title(f'{name}: {proj_type} (red) / mirrored (green)')
    for ax, hemibody, side in zip(axes[1:], hemibodies, ('L', 'R')):
        ax.imshow(np.max(hemibody, axis=0), cmap='gray',
                  vmax=np.percentile(hemibody, 99.9))
        ax.set_title(side)
    fig.tight_layout()
    fig.savefig(out_folder / f'{name}_split.png', dpi=100)
    plt.close(fig)
    return paths


if __name__ == '__main__':
    matplotlib.use('Agg')
    parser = argparse.ArgumentParser(description='Split SCAPE recordings into isotropic '
                                                 'hemibodies.')
    parser.add_argument('scape_paths', nargs='+', help='sample#_run# folders.')
    parser.add_argument('-o', '--out_folder', required=True, help='Output folder.')
    parser.add_argument('-p', '--proj_type', default='min_proj',
                        choices=['min_proj', 'max_proj', 'std_image'])
    parser.add_argument('-v', '--voxel_size', type=float, default=TARGET_VOXEL_SIZE,
                        help='Isotropic voxel size, in um.')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    for scape_path in args.scape_paths:
        split_recording(scape_path, args.out_folder, args.proj_type, args.voxel_size,
                        args.overwrite)
