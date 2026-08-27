"""
Interactive napari tool for manually refining the rigid alignment of one
reference leg-segment (from the webknossos annotation) onto a SCAPE
projection.

Usage
-----
    python align_segment_napari.py /path/to/analysis/max_proj.ome.tif

Expects `pre-aligned_labels.nrrd` and `pre-aligned_labels_ids.json`
(saved by align_acc_heart.ipynb) alongside the given projection file.

Note: the projection's pixel data is read directly from
`projection_images.h5` (which sits next to the given .ome.tif) instead
of from the .ome.tif itself - that tiff's last two axes are swapped
relative to the (z, y, x) convention `T_forward` was actually fit
against (see `_get_projection_images` in scape_utils.py, which
transposes before writing the tiff for OME metadata compliance). Voxel
size is read from the .ome.tif's own OME-XML metadata instead of via
`scape_utils.load_voxel_size`, since that requires a `sample#_run#`
raw-data folder that a combined/merged projection (e.g.
`sample#_combined`) doesn't have - the embedded PhysicalSizeX/Y/Z
values are exactly `scape_utils`'s own voxel_size['x']/['y']/['z'],
re-embedded verbatim when the tiff was written, so this works
identically for both a `sample#_run#/analysis/` and a
`sample#_combined/` layout, and needs no raw-data access at all.

Displayed labels are resampled onto a world-axis-aligned grid (see
`resample_for_display`) rather than shown directly with a rotated
affine: napari's 2D slice views only fully support an orthogonal
(diagonal - translate/scale only) affine, silently dropping any
out-of-slice rotation/shear component otherwise (a real napari
limitation, not specific to this data - it emits "Non-orthogonal
slicing is being requested, but is not fully supported" and the layer
then only lines up for whichever axis order happens to make the
dropped rotation negligible). The true rotated transform is still
tracked separately and is what gets saved. Also attaches
`_MultiAxisViewWidget` from `napari_point_annotation.py`, so all 3
axis orders are visible side by side instead of needing to toggle
napari's own axis-order control.
"""
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import h5py
import napari
import nrrd
import numpy as np
import tifffile
from magicgui import magicgui
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation

sys.path.append(str(Path(__file__).parent))
from napari_point_annotation import _MultiAxisViewWidget
from scape_utils import load_napari_rois


def load_projection(analysis_path: Path, proj_tiff_path: Path) -> tuple:
    """
    Loads the max-projection volume and its voxel size, without relying
    on `scape_utils.load_proj`/`load_voxel_size` (which require a
    `sample#_run#` raw-data folder that a combined/merged projection
    doesn't have).

    Parameters
    ----------
    analysis_path : Path
        Folder containing `projection_images.h5` (the same folder as
        `proj_tiff_path`).

    proj_tiff_path : Path
        Path to the `max_proj.ome.tif` file, used only for its embedded
        OME-XML voxel size metadata.

    Returns
    -------
    proj : ndarray
        Max-projection volume, (z, x, y) - `scape_utils`'s own axis
        convention (matching `T_forward`), read straight from the h5
        file rather than the axis-swapped tiff.

    voxel_size : dict of str to float
        `{'z': ..., 'x': ..., 'y': ...}`, in `scape_utils`'s own axis
        naming (see the module docstring).
    """
    with h5py.File(analysis_path / 'projection_images.h5', 'r') as f:
        proj = f['max_proj'][:]

    with tifffile.TiffFile(proj_tiff_path) as tif:
        ome_root = ET.fromstring(tif.ome_metadata)
    ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
    pixels = ome_root.find('.//ome:Pixels', ns)
    voxel_size = {
        'z': float(pixels.attrib['PhysicalSizeZ']),
        'x': float(pixels.attrib['PhysicalSizeX']),
        'y': float(pixels.attrib['PhysicalSizeY']),
    }
    return proj, voxel_size


def load_reference_data(analysis_path: Path) -> tuple:
    """
    Loads the pre-aligned native-resolution reference labels and their
    id -> group/T_forward metadata, saved by align_acc_heart.ipynb.

    Parameters
    ----------
    analysis_path : Path
        Folder containing `pre-aligned_labels.nrrd` and
        `pre-aligned_labels_ids.json` (the SCAPE recording's `analysis`
        folder).

    Returns
    -------
    labels : ndarray of uint32
        Native-resolution reference labels, (z, y, x) (see
        `load_napari_rois`).

    id_to_group : dict of int to str
        Maps each labeled id to its `'{leg}_{segment}'` group name.

    T_forward : (4, 4) ndarray
        Initial rigid+scale transform from `labels`' array-index space
        to the projection's physical space (see align_acc_heart.ipynb).
    """
    labels = load_napari_rois(analysis_path / 'pre-aligned_labels.nrrd')
    with open(analysis_path / 'pre-aligned_labels_ids.json') as f:
        metadata = json.load(f)
    id_to_group = {int(k): v for k, v in metadata['id_to_group'].items()}
    T_forward = np.array(metadata['T_forward'])
    return labels, id_to_group, T_forward


def crop_to_group(labels: np.ndarray, ids: list) -> tuple:
    """
    Crops `labels` to the bounding box of the voxels matching `ids`,
    zeroing out everything else within that box.

    Parameters
    ----------
    labels : ndarray
        Native-resolution reference labels.

    ids : list of int
        Label ids belonging to the segment to extract.

    Returns
    -------
    cropped : ndarray
        `labels`, cropped to the bounding box of `ids` and zeroed
        outside them.

    bbox_start : ndarray of int
        Start index of the crop along each axis, in `labels`' own
        array-index coordinates.
    """
    mask = np.isin(labels, ids)
    coords = np.argwhere(mask)
    bbox_start = coords.min(axis=0)
    bbox_stop = coords.max(axis=0) + 1
    slices = tuple(slice(start, stop) for start, stop in zip(bbox_start, bbox_stop))
    cropped = np.where(mask[slices], labels[slices], 0).astype(labels.dtype)
    return cropped, bbox_start


def build_delta_transform(translate: tuple, rotate_degrees: tuple, scale: float,
                           center: np.ndarray) -> np.ndarray:
    """
    Builds a 4x4 homogeneous transform for a rotation + uniform scale
    (both about `center`) plus a translation, for interactively nudging
    a layer's alignment on top of its initial affine.

    Parameters
    ----------
    translate : tuple of float
        (z, y, x) translation, in the layer's own array-index units.

    rotate_degrees : tuple of float
        (z, y, x) Euler angles, in degrees ('xyz' extrinsic
        convention), rotating about `center`.

    scale : float
        Uniform scale factor, about `center`.

    center : ndarray
        (z, y, x) point to rotate/scale about (the cropped segment's
        own centroid), so small rotations don't fling the segment out
        of view.

    Returns
    -------
    (4, 4) ndarray
        Homogeneous delta transform.
    """
    linear = scale * Rotation.from_euler('xyz', rotate_degrees, degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = linear
    T[:3, 3] = np.array(translate) + center - linear @ center
    return T


def resample_for_display(cropped: np.ndarray, full_affine: np.ndarray,
                          voxel_size: tuple) -> tuple:
    """
    Resamples `cropped` through `full_affine` onto a new grid aligned
    with world axes, at `voxel_size` spacing, sized to just fit the
    transformed data - so it can be displayed with a diagonal
    (rotation-free) affine, which is all napari's 2D slice views fully
    support (see the module docstring). This is for display only: the
    saved output uses `cropped` and `full_affine` unchanged.

    Parameters
    ----------
    cropped : ndarray
        Native-resolution cropped label data, in its own local
        array-index coordinates.

    full_affine : (4, 4) ndarray
        Maps `cropped`'s local array-index coordinates to world
        (physical) space, including any rotation.

    voxel_size : tuple of float
        Physical size of one output voxel, one value per axis (here,
        `proj`'s own voxel size, so the display resample is cheap and
        visually matches `proj`'s resolution).

    Returns
    -------
    resampled : ndarray
        `cropped`, resampled onto the new world-axis-aligned grid.

    display_affine : (4, 4) ndarray
        Diagonal (translate + scale only) affine placing `resampled`
        correctly in world space.
    """
    corners_local = np.array(np.meshgrid(*[[0, s] for s in cropped.shape], indexing='ij')).reshape(3, -1).T
    corners_world = (full_affine[:3, :3] @ corners_local.T).T + full_affine[:3, 3]
    world_min = corners_world.min(axis=0)
    world_max = corners_world.max(axis=0)
    output_shape = tuple(np.maximum(np.ceil((world_max - world_min) / voxel_size).astype(int), 1))

    display_affine = np.eye(4)
    display_affine[:3, :3] = np.diag(voxel_size)
    display_affine[:3, 3] = world_min

    # pull mapping: output-grid index -> `cropped`'s local index
    pull_matrix = np.linalg.inv(full_affine) @ display_affine
    resampled = affine_transform(
        cropped, matrix=pull_matrix[:3, :3], offset=pull_matrix[:3, 3],
        output_shape=output_shape, order=0, mode='constant', cval=0,
    )
    return resampled, display_affine


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f'Usage: python {Path(__file__).name} /path/to/analysis/max_proj.ome.tif')

    proj_tiff_path = Path(sys.argv[1])
    analysis_path = proj_tiff_path.parent

    proj, voxel_sizes = load_projection(analysis_path, proj_tiff_path)
    axes_order = ['z', 'x', 'y']  # matches `proj`'s own array-axis order (see load_projection)
    proj_scale = tuple(voxel_sizes[ax] for ax in axes_order)

    labels, id_to_group, T_forward = load_reference_data(analysis_path)
    ids_by_group = defaultdict(list)
    for seg_id, group_name in id_to_group.items():
        ids_by_group[group_name].append(seg_id)
    group_names = sorted(ids_by_group)

    # needed to allow undocking the 'Other axes' widget (2 extra GL canvases)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

    viewer = napari.Viewer()
    # must attach before adding any layers - it only mirrors layers added
    # after it starts listening (see napari_point_annotation.py, where
    # layers are always added later, via File > Open)
    viewer.window.add_dock_widget(_MultiAxisViewWidget(viewer), name='Other axes', area='right')
    viewer.add_image(proj, name='proj', scale=proj_scale)

    # mutable state shared between the three widgets below, for the
    # single currently-active segment
    state = {'layer': None, 'initial_affine': None, 'full_affine': None,
              'bbox_start': None, 'group_name': None, 'cropped': None}

    def _refresh_display(delta: np.ndarray):
        """Recomputes `state['full_affine']` and the displayed (resampled,
        diagonal-affine) layer for the current delta transform - see the
        module docstring and `resample_for_display`."""
        full_affine = state['initial_affine'] @ delta
        resampled, display_affine = resample_for_display(state['cropped'], full_affine, proj_scale)
        state['full_affine'] = full_affine
        state['layer'].data = resampled
        state['layer'].affine = display_affine

    @magicgui(call_button='Load', group={'choices': group_names, 'label': 'Segment'})
    def load_widget(group: str = group_names[0]):
        if state['layer'] is not None:
            viewer.layers.remove(state['layer'])

        cropped, bbox_start = crop_to_group(labels, ids_by_group[group])
        bbox_translate = np.eye(4)
        bbox_translate[:3, 3] = bbox_start
        initial_affine = T_forward @ bbox_translate

        state.update(layer=None, initial_affine=initial_affine, full_affine=None,
                      bbox_start=bbox_start, group_name=group, cropped=cropped)

        resampled, display_affine = resample_for_display(cropped, initial_affine, proj_scale)
        state['layer'] = viewer.add_labels(resampled, name=group, affine=display_affine)
        state['full_affine'] = initial_affine

        # reset the align widget's sliders for the newly-loaded segment,
        # without triggering align_widget's auto_call: firing 7 rapid
        # .data/.affine reassignments right after this add (itself right
        # after a remove, for every load after the first) desynced
        # _MultiAxisViewWidget's per-model layer bookkeeping, corrupting
        # the viewer - display is already correct from add_labels above,
        # so the reset only needs to update the on-screen slider values
        for name in ('translate_z', 'translate_y', 'translate_x',
                     'rotate_z', 'rotate_y', 'rotate_x'):
            widget = getattr(align_widget, name)
            with widget.changed.blocked():
                widget.value = 0.0
        with align_widget.scale.changed.blocked():
            align_widget.scale.value = 1.0

        print(f'Loaded {group}: {cropped.shape} (bbox start {tuple(bbox_start)}), '
              f'{len(ids_by_group[group])} segments, {np.count_nonzero(cropped)} voxels')

    @magicgui(
        auto_call=True,
        translate_z={'min': -2000, 'max': 2000, 'step': 1, 'label': 'translate z'},
        translate_y={'min': -2000, 'max': 2000, 'step': 1, 'label': 'translate y'},
        translate_x={'min': -2000, 'max': 2000, 'step': 1, 'label': 'translate x'},
        rotate_z={'min': -45, 'max': 45, 'step': 0.5, 'label': 'rotate z'},
        rotate_y={'min': -45, 'max': 45, 'step': 0.5, 'label': 'rotate y'},
        rotate_x={'min': -45, 'max': 45, 'step': 0.5, 'label': 'rotate x'},
        scale={'min': 0.5, 'max': 2.0, 'step': 0.01},
    )
    def align_widget(translate_z: float = 0.0, translate_y: float = 0.0, translate_x: float = 0.0,
                      rotate_z: float = 0.0, rotate_y: float = 0.0, rotate_x: float = 0.0,
                      scale: float = 1.0):
        if state['layer'] is None:
            return
        center = np.array(state['cropped'].shape) / 2
        delta = build_delta_transform(
            translate=(translate_z, translate_y, translate_x),
            rotate_degrees=(rotate_z, rotate_y, rotate_x),
            scale=scale, center=center,
        )
        _refresh_display(delta)

    @magicgui(call_button='Save')
    def save_widget():
        if state['layer'] is None:
            print('Nothing loaded to save.')
            return

        group_name = state['group_name']
        # NOT layer.affine - that's the diagonal display-only affine (see
        # resample_for_display); the true (rotated) transform is tracked
        # separately in state['full_affine']
        final_affine = state['full_affine']

        out_nrrd = analysis_path / f'{group_name}_fully-aligned.nrrd'
        # nrrd files are (x, y, z); `cropped` is (z, y, x) (see
        # load_reference_data), matching load_napari_rois' convention
        nrrd.write(str(out_nrrd), np.transpose(state['cropped'], (2, 1, 0)))

        out_h5 = analysis_path / f'{group_name}_transform.h5'
        with h5py.File(out_h5, 'w') as f:
            f.create_dataset('affine', data=final_affine)
            f.attrs['group_name'] = group_name
            f.attrs['bbox_start'] = state['bbox_start']
            f.attrs['note'] = ('affine maps this nrrd file\'s own (z, y, x) array-index '
                                'coordinates directly to proj\'s physical (um) space')

        print(f'Saved {out_nrrd.name} and {out_h5.name}')

    viewer.window.add_dock_widget(load_widget, area='right', name='Load segment')
    viewer.window.add_dock_widget(align_widget, area='right', name='Align')
    viewer.window.add_dock_widget(save_widget, area='right', name='Save')

    napari.run()


if __name__ == '__main__':
    main()
