"""
Scripts to load and manipulate reference maps (cuticles, muscles, etc.) labelled in webknossos.
Can be in two formats, labels or meshes.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple, Union

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
    block_len = dataset.header.block_len

    indices = [
        (int(p.stem[1:]), int(p.parent.name[1:]), int(p.parent.parent.name[1:]))
        for p in mag_dir.glob('z*/y*/x*.wkw')
    ]
    offset = tuple(min(axis) * block_len for axis in zip(*indices))
    shape = tuple((max(axis) - min(axis) + 1) * block_len for axis in zip(*indices))

    label_volume = dataset.read(offset, shape)[0]
    dataset.close()

    return label_volume, id_to_name
