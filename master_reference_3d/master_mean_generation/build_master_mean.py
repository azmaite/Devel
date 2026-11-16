"""
Build the master reference (master mean image and master muscle mask labelled by leg segment)
from the median projections of the SCAPE recordings listed in `recordings.txt`, in `N_ROUNDS`
registration rounds (see `pipeline.md` for the method and `README.md` for the scripts). Each step
is skipped if its output already exists, so the script can be rerun after a crash.

Starts from the median projections of the recordings
(`{sample#_run#}/analysis/projection_images.h5`, computed in the `scape` folder with
`scape/preprocess_scape_exp.py`; `median_proj` only exists for runs of fewer than 75 volumes),
and saves everything in `OUT_PATH`:
1. `hemibodies`: the two isotropic hemibodies of each recording
   (`../registration/hemibodies.py`). The
   reference recording is split with the plane of its earlier minimum-projection split
   (`REFERENCE_SPLIT`), so that its hemibodies are on the grid of its hand-labelled clusters,
2. `segment_masks`: the automatic muscle mask of each hemibody used, labelled by muscle cluster,
   with a scale-1 copy of the image (`../registration/muscle_mask.label_hemibody`),
3. `registered_hemibodies_round1`, `_round2`, ...: `N_ROUNDS` registration rounds
   (`registration_round.py`). Round 1 registers all hemibodies to the reference hemibody
   (`REFERENCE`), whose muscle mask is labelled by leg segment with `reference_segments`; each
   later round registers all of them to the consensus of the previous round
   (`registration_round.template_from_round`),
4. the master, from the last round, saved directly in `OUT_PATH`: `master_mean.tif` (mean of
   the normalised registered images, float32), `master_segments.tif` (master muscle mask
   labelled by leg segment, uint32) and `master_segments.json` (segment names), on the same grid
   (1.5 um voxels; they open together in napari at scale 1), plus a copy of `recordings.txt`.
   New recordings are registered to them with `../register_recording.py`.

The leg segments of the reference come from the hand labelling of the minimum-projection master
(`REFERENCE_LABELS`: `{recording}_segment_ids.json`, the leg segment of each cluster of
`{REFERENCE}_components.tif`). The median mask of the reference is on the same grid, but its
clusters differ, so each of its voxels takes the segment of the nearest hand-labelled voxel
within `registration_round.TEMPLATE_SEGMENT_DISTANCE_UM`.

Run from the terminal: `python build_master_mean.py` (with the `scapeio` conda environment;
settings at the top of the script). It takes many hours: run it in `tmux` when working through
ssh.

read_recordings: read the recording list.
segment_map: segment map of a labelled muscle mask from its segment table.
reference_segments: the reference muscle mask labelled by leg segment.
build_master_mean: run all the steps.
"""

import json
import shutil
import sys
from pathlib import Path

import matplotlib
import numpy as np
import scipy.ndimage as ndi
import tifffile

import registration_round as rr
from common.paths import MASTER_REFERENCE_PATH, SCAPE_PATH

sys.path.insert(0, str(Path(__file__).parents[1] / 'registration'))
import deformation as dfm  # noqa: E402
import hemibodies  # noqa: E402
import muscle_mask  # noqa: E402

RECORDING_LIST = Path(__file__).parent / 'recordings.txt'
OUT_PATH = MASTER_REFERENCE_PATH / 'master_median'
PROJ_TYPE = 'median_proj'
N_ROUNDS = 3
# hemibody (`{experiment}_{sample#_run#}_{side}`) whose leg segments are known: the template of
# round 1
REFERENCE = '20261021_V7xPR_pupa_long_paired_early_sample1_run1_R'
# hand-made leg segments of the reference's minimum-projection clusters, and that split
REFERENCE_LABELS = MASTER_REFERENCE_PATH / 'segment_masks'
REFERENCE_SPLIT = (MASTER_REFERENCE_PATH / 'original_hemibodies'
                   / '20261021_V7xPR_pupa_long_paired_early_sample1_run1_split.json')
VOXEL_SIZE_UM = rr.VOXEL_SIZE_UM


def read_recordings(path: Path = RECORDING_LIST) -> list:
    """
    Read the recording list (see `recordings.txt`).

    Parameters
    ----------
    path : Path
        Text file: one `sample#_run#` folder per line, relative to `SCAPE_PATH`, optionally
        followed by 'L' or 'R' to use only that hemibody; lines starting with '#' are ignored.

    Returns
    -------
    list of tuple
        (recording folder, list of the sides used).
    """
    recordings = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        folder, *sides = line.split()
        recordings.append((SCAPE_PATH / folder, sides or ['L', 'R']))
    return recordings


def segment_map(components: np.ndarray,
                table: dict,
                segment_names: list) -> np.ndarray:
    """
    Segment of each voxel of a labelled muscle mask, from a segment table.

    Parameters
    ----------
    components : np.ndarray
        Muscle clusters (integer labels).
    table : dict
        Segment name -> list of cluster labels.
    segment_names : list of str
        Segments to number, in order (1..); others (e.g. the ignored ones) and unlisted labels
        are 0.

    Returns
    -------
    np.ndarray
        uint32 segment map.
    """
    lookup = np.zeros(int(components.max()) + 1, dtype=np.uint32)
    for index, segment in enumerate(segment_names, start=1):
        labels = [label for label in table.get(segment, []) if label < lookup.size]
        lookup[labels] = index
    return lookup[components]


def reference_segments(mask: np.ndarray,
                       reference: str = REFERENCE,
                       label_folder: Path = REFERENCE_LABELS) -> tuple:
    """
    Label a muscle mask of the reference hemibody by leg segment, from the hand-labelled clusters
    of another mask of it on the same grid (see the module docstring). The head
    (`deformation.IGNORED_SEGMENTS`) is left out.

    Parameters
    ----------
    mask : np.ndarray
        Boolean muscle mask of the reference.
    reference : str
        Reference hemibody name (`{recording}_{side}`).
    label_folder : Path
        Folder with `{recording}_segment_ids.json` and the `{reference}_components.tif` it
        refers to.

    Returns
    -------
    tuple
        (uint32 segment map, same grid as `mask`, 1..; list of segment names, in label order).
    """
    recording, side = reference.rsplit('_', 1)
    table = json.loads((label_folder / f'{recording}_segment_ids.json').read_text())
    all_names = table['segments']
    hand = segment_map(tifffile.imread(label_folder / f'{reference}_components.tif'),
                       table[side], all_names)
    if hand.shape != mask.shape:
        raise ValueError(f'{reference}: the hand-labelled clusters {hand.shape} and the mask '
                         f'{mask.shape} are not on the same grid')
    distance, indices = ndi.distance_transform_edt(hand == 0, sampling=VOXEL_SIZE_UM,
                                                   return_indices=True)
    labels = np.where(mask & (distance <= rr.TEMPLATE_SEGMENT_DISTANCE_UM),
                      hand[tuple(indices)], 0)
    # renumber without the head (its voxels become 0)
    names = [name for name in all_names if name not in dfm.IGNORED_SEGMENTS]
    lookup = np.zeros(len(all_names) + 1, np.uint32)
    for index, name in enumerate(all_names, start=1):
        if name in names:
            lookup[index] = names.index(name) + 1
    return lookup[labels], names


def build_master_mean() -> None:
    """
    Run all the steps (see the module docstring).

    Returns
    -------
    None
    """
    hemibody_folder = OUT_PATH / 'hemibodies'
    segment_folder = OUT_PATH / 'segment_masks'

    # 1-2. hemibodies and their muscle clusters
    names = []
    for recording, sides in read_recordings():
        name = f'{recording.parent.name}_{recording.name}'
        plane_from = REFERENCE_SPLIT if REFERENCE.rsplit('_', 1)[0] == name else None
        hemibodies.split_recording(recording, hemibody_folder, PROJ_TYPE, VOXEL_SIZE_UM,
                                   plane_from=plane_from)
        for side in sides:
            names.append(f'{name}_{side}')
            if not (segment_folder / f'{name}_{side}_components.tif').exists():
                muscle_mask.label_hemibody(hemibody_folder / f'{name}_{side}.ome.tif',
                                           segment_folder, VOXEL_SIZE_UM)
    if REFERENCE not in names:
        raise ValueError(f'REFERENCE {REFERENCE} is not one of the hemibodies of '
                         f'{RECORDING_LIST}')
    others = [name for name in names if name != REFERENCE]

    # 3. registration rounds
    for index in range(1, N_ROUNDS + 1):
        round_folder = OUT_PATH / f'registered_hemibodies_round{index}'
        if (round_folder / 'master_mean.tif').exists():
            print(f'Round {index} already done ({round_folder.name}), skipping.')
            continue
        print(f'\nRound {index} -> {round_folder.name}')
        if index == 1:
            mask = tifffile.imread(segment_folder / f'{REFERENCE}_components.tif') > 0
            template, segment_names = reference_segments(mask)
            rr.registration_round(segment_folder, round_folder, template, segment_names,
                                  REFERENCE, others, reference=REFERENCE)
        else:
            previous = OUT_PATH / f'registered_hemibodies_round{index - 1}'
            template, segment_names = rr.template_from_round(previous)
            rr.registration_round(segment_folder, round_folder, template, segment_names,
                                  f'consensus of {previous.name}', names)

    # 4. the master, from the last round
    last_round = OUT_PATH / f'registered_hemibodies_round{N_ROUNDS}'
    segments, segment_names = rr.template_from_round(last_round)
    mean = tifffile.imread(last_round / 'master_mean.tif').astype(np.float32)
    tifffile.imwrite(OUT_PATH / 'master_mean.tif', mean)
    tifffile.imwrite(OUT_PATH / 'master_segments.tif', segments, compression='zlib')
    (OUT_PATH / 'master_segments.json').write_text(json.dumps(
        {'source_round': str(last_round), 'projection': PROJ_TYPE, 'hemibodies': names,
         'grid_shape': list(segments.shape), 'voxel_size_um': VOXEL_SIZE_UM,
         'array_axes': 'ZXY (as the hemibodies)',
         'id_to_name': {i: name for i, name in enumerate(segment_names, start=1)}}, indent=1))
    shutil.copy(RECORDING_LIST, OUT_PATH / RECORDING_LIST.name)
    print(f'Saved the master in {OUT_PATH} (from {last_round.name})')


if __name__ == '__main__':
    matplotlib.use('Agg')
    build_master_mean()
