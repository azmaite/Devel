"""
Scripts to load and manipulate reference maps (cuticles, muscles, etc.) labelled in webknossos.
Can be in two formats, labels or meshes.

load_labels: load a webknossos label export as a volume of segment ids (the block of
    labelled data only, without its offset in the dataset) and the id -> name mapping
load_labels_in_frame: same, but with the labels placed in the full frame of the dataset
    they were drawn on, so they line up with the image
load_segment_group_names: map segment ids to the name of the group (folder) they are in
load_segment_groups: map segment ids to their enclosing `{leg_id}_{segment}` group
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Sequence, Tuple, Union

import numpy as np
import wkw


def load_labels(labels_dir: Union[str, Path]) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Load a webknossos volume-annotation label export.

    Parameters
    ----------
    labels_dir : str or Path
        Path to a 'Labels' export folder downloaded from webknossos.
        Contains one `.nml` file and a `data_Volume` folder with
        per-magnification subfolders (`1`, `2`, `4`, ...).

    Returns
    -------
    label_volume : np.ndarray
        3D array of `numpy.uint32` segment ids, indexed `[x, y, z]`.
        Unlabelled voxels are `0`.

    id_to_name : dict of int to str
        Maps each named segment id to its webknossos name
        (e.g. `RF_ThCo_1`).
    """
    labels_dir = Path(labels_dir)

    # check that the folder contains a .nml file and a subfolder 'data_Volume' with the right format
    assert labels_dir.is_dir(), f"Input is not a valid directory"
    try:
        nml_path = next(labels_dir.glob('*.nml'))
    except StopIteration:
        raise AssertionError(f"No .nml file found - is this the right folder?")
    assert (labels_dir / 'data_Volume').is_dir(), f"No 'data_Volume' folder found - is this the right folder?"
    assert (labels_dir / 'data_Volume' / '1').is_dir(), f"No '1' subfolder found - is this the right folder?"
    
    segments_elem = ET.parse(nml_path).getroot().find('volume/segments')
    assert segments_elem is not None, f"No <segments> found in '{nml_path}'"
    id_to_name = {
        int(segment.attrib['id']): segment.attrib['name']
        for segment in segments_elem
        if 'name' in segment.attrib
    }

    mag_dir = labels_dir / 'data_Volume' / '1'
    dataset = wkw.Dataset.open(str(mag_dir))
    offset, shape = _get_labelled_block(mag_dir, dataset.header.block_len)

    label_volume = dataset.read(offset, shape)[0]
    dataset.close()

    return label_volume, id_to_name


def _get_labelled_block(mag_dir: Path, block_len: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Get the offset and shape (both `(x, y, z)`, in voxels) of the block of
    wkw files that holds all the labels of a magnification folder.

    Parameters
    ----------
    mag_dir : Path
        The `data_Volume/1` folder.

    block_len : int
        Side length of a wkw block (`dataset.header.block_len`).

    Returns
    -------
    offset : tuple of int
        `(x, y, z)` position of the block's first voxel in the dataset.

    shape : tuple of int
        `(x, y, z)` size of the block.
    """
    indices = [
        (int(p.stem[1:]), int(p.parent.name[1:]), int(p.parent.parent.name[1:]))
        for p in mag_dir.glob('z*/y*/x*.wkw')
    ]
    offset = tuple(min(axis) * block_len for axis in zip(*indices))
    shape = tuple((max(axis) - min(axis) + 1) * block_len for axis in zip(*indices))
    return offset, shape


def load_labels_in_frame(labels_dir: Union[str, Path],
                         shape_xyz: Sequence[int]) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Like `load_labels`, but returns the labels placed in the full frame of the
    dataset they were drawn on, so they line up with the image (`load_labels`
    only returns the block of wkw files holding labels, without its offset).

    Parameters
    ----------
    labels_dir : str or Path
        Same 'Labels' export folder passed to `load_labels`.

    shape_xyz : sequence of int
        `(x, y, z)` shape of the dataset the labels were drawn on.

    Returns
    -------
    label_volume : np.ndarray
        3D array of `numpy.uint32` segment ids with shape `shape_xyz`, indexed
        `[x, y, z]`. Unlabelled voxels are `0`.

    id_to_name : dict of int to str
        Maps each named segment id to its webknossos name.
    """
    labels_dir = Path(labels_dir)
    block, id_to_name = load_labels(labels_dir)
    mag_dir = labels_dir / 'data_Volume' / '1'
    dataset = wkw.Dataset.open(str(mag_dir))
    offset, _ = _get_labelled_block(mag_dir, dataset.header.block_len)
    dataset.close()

    frame = np.zeros(tuple(shape_xyz), dtype=block.dtype)
    inside = tuple(slice(0, min(size, total - start))
                   for size, total, start in zip(block.shape, shape_xyz, offset))
    frame_slices = tuple(slice(start, start + s.stop) for start, s in zip(offset, inside))
    frame[frame_slices] = block[inside]
    if np.count_nonzero(frame) != np.count_nonzero(block):
        raise ValueError(f"Labels in '{labels_dir}' extend outside the given frame "
                         f"{tuple(shape_xyz)} - is it the dataset they were drawn on?")
    return frame, id_to_name


def load_segment_group_names(labels_dir: Union[str, Path]) -> Dict[int, str]:
    """
    Maps each segment id in a webknossos label export to the name of the group
    (folder) it was put in, whatever that name is (unlike `load_segment_groups`,
    which only resolves names matching leg/segment combinations).

    Parameters
    ----------
    labels_dir : str or Path
        Same 'Labels' export folder passed to `load_labels`.

    Returns
    -------
    dict of int to str
        Maps segment id to the name of its (innermost) group. Segments in no
        group are omitted.
    """
    nml_path = next(Path(labels_dir).glob('*.nml'))
    root = ET.parse(nml_path).getroot()
    group_names = {int(group.attrib['id']): group.attrib['name']
                   for group in root.findall('volume/groups//group')}
    segments_elem = root.find('volume/segments')
    assert segments_elem is not None, f"No <segments> found in '{nml_path}'"
    return {int(segment.attrib['id']): group_names[int(segment.attrib['groupId'])]
            for segment in segments_elem if 'groupId' in segment.attrib}


def load_segment_groups(labels_dir: Union[str, Path], leg_ids: Sequence[str],
                         segments: Sequence[str]) -> Dict[int, str]:
    """
    Maps each segment id in a webknossos label export to the name of its
    enclosing `{leg_id}_{segment}` group.

    Segment groups can be nested (e.g. a joint-specific subgroup nested
    inside a leg segment's group) - this walks up each segment's group
    chain until it finds an ancestor whose name matches one of the
    `{leg_id}_{segment}` combinations given, so a segment grouped under a
    nested subgroup still resolves to its containing leg segment.

    Parameters
    ----------
    labels_dir : str or Path
        Same 'Labels' export folder passed to `load_labels`.

    leg_ids : sequence of str
        Valid leg identities, e.g. `('RF', 'LF', 'RM', 'LM')`.

    segments : sequence of str
        Valid segment names, e.g. `('coxa', 'trochanter', 'femur', 'tibia')`.

    Returns
    -------
    dict of int to str
        Maps segment id to its resolved `'{leg_id}_{segment}'` group name.
        Ids with no group, or whose group chain never resolves to one of
        the given `leg_ids`/`segments` combinations (e.g. a `non-muscles`
        or `thorax` group), are omitted.
    """
    labels_dir = Path(labels_dir)
    nml_path = next(labels_dir.glob('*.nml'))
    root = ET.parse(nml_path).getroot()

    valid_names = {f'{leg_id}_{segment}' for leg_id in leg_ids for segment in segments}

    group_name = {}
    group_parent = {}

    def walk_groups(parent_elem, parent_id):
        for group_elem in parent_elem.findall('group'):
            group_id = int(group_elem.attrib['id'])
            group_name[group_id] = group_elem.attrib['name']
            group_parent[group_id] = parent_id
            walk_groups(group_elem, group_id)

    # this is `volume/groups`, not the always-empty top-level `<groups/>`
    # sibling of `<parameters>` - that one is unrelated NML boilerplate
    groups_elem = root.find('volume/groups')
    if groups_elem is not None:
        walk_groups(groups_elem, None)

    def resolve(group_id):
        while group_id is not None:
            name = group_name.get(group_id)
            if name in valid_names:
                return name
            group_id = group_parent.get(group_id)
        return None

    segments_elem = root.find('volume/segments')
    assert segments_elem is not None, f"No <segments> found in '{nml_path}'"

    segment_id_to_group = {}
    for segment_elem in segments_elem:
        if 'groupId' not in segment_elem.attrib:
            continue
        resolved = resolve(int(segment_elem.attrib['groupId']))
        if resolved is not None:
            segment_id_to_group[int(segment_elem.attrib['id'])] = resolved

    return segment_id_to_group
