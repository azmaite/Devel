"""
Build the master reference from SCAPE recordings, running every step of this folder in order (see
`pipeline.md` for the method and `README.md` for the scripts). Each step is skipped if its output
already exists, so the script can be rerun after adding a recording or after the manual step.

Starts from the projection images of the recordings in `RECORDINGS`
(`{sample#_run#}/analysis/projection_images.h5`, computed in the `scape` folder with
`scape/preprocess_scape_exp.py`), and saves everything in `MASTER_PATH`:
1. `original_hemibodies`: the two isotropic hemibodies of each recording
   (`split_hemibodies.py`),
2. `segment_masks`: the automatic muscle mask of each hemibody, labelled by muscle cluster, with
   a scale-1 copy of the image (`label_mask_components.py`). Existing cluster labels are never
   recomputed, since they may have been edited by hand (e.g. a cluster split in two),
3. manual step: the leg segment of each cluster of the reference hemibody (`REFERENCE`), in
   `segment_masks/{recording}_segment_ids.json`. If it is missing, the script stops here and says
   what to write (look at `{name}_image.tif` and `{name}_components.tif` together in napari),
4. `registered_hemibodies`, `registered_hemibodies_round2`, ...: `N_ROUNDS` rounds of
   registration and averaging (`registration_round.py`),
5. the final master reference, from the last round, saved directly in `MASTER_PATH`:
   `master_mean.tif` (mean of the normalised registered images, float32),
   `master_segments.tif` (master muscle mask labelled by leg segment, uint32: the majority vote
   of the last round's registered muscle masks, see `registration_round.template_from_round`)
   and `master_segments.json` (segment names). Both are on the same grid (1.5 um voxels) and open
   together in napari at scale 1. New recordings are registered to them with
   `register_recording.py`.

Run from the terminal: `python build_master_reference.py` (settings at the top of the script; `--overwrite_rounds`
reruns the registration rounds even if they exist).

build_master_reference: run all the steps.
save_final_master: save the final master reference from a finished round.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile

import label_mask_components
import registration_round
import split_hemibodies

from common.paths import MASTER_REFERENCE_PATH, SCAPE_PATH

# recordings (`sample#_run#` folders) whose minimum projections are used
RECORDINGS = [SCAPE_PATH / '20261016_V7xPR_pupa_long_paired' / 'sample1_run3_short',
              SCAPE_PATH / '20261021_V7xPR_pupa_long_paired_early' / 'sample1_run1',
              SCAPE_PATH / '20261021_V7xPR_pupa_long_scape' / 'sample1_run01_short_20261021_1638']
MASTER_PATH = MASTER_REFERENCE_PATH
# hemibody (`{experiment}_{sample#_run#}_{side}`) whose clusters are assigned to leg segments by
# hand, the template of round 1
REFERENCE = '20261021_V7xPR_pupa_long_paired_early_sample1_run1_R'
N_ROUNDS = 3
VOXEL_SIZE_UM = 1.5


def _round_folder(index: int) -> str:
    """
    Name of the output folder of registration round `index` (1-based).

    Parameters
    ----------
    index : int
        Round number.

    Returns
    -------
    str
        'registered_hemibodies' for round 1, 'registered_hemibodies_round{index}' after.
    """
    return 'registered_hemibodies' if index == 1 else f'registered_hemibodies_round{index}'


def save_final_master(round_folder: Path,
                      master_path: Path = MASTER_PATH) -> None:
    """
    Save the final master reference from a finished registration round, in `master_path` (see
    step 5 of the module docstring).

    Parameters
    ----------
    round_folder : Path
        Output folder of the last round.
    master_path : Path
        Folder to save `master_mean.tif`, `master_segments.tif` and `master_segments.json` in.

    Returns
    -------
    None
    """
    segments, segment_names = registration_round.template_from_round(round_folder)
    mean = tifffile.imread(round_folder / 'master_mean.tif').astype(np.float32)
    tifffile.imwrite(master_path / 'master_mean.tif', mean)
    tifffile.imwrite(master_path / 'master_segments.tif', segments, compression='zlib')
    (master_path / 'master_segments.json').write_text(json.dumps(
        {'source_round': str(round_folder), 'grid_shape': list(segments.shape),
         'voxel_size_um': VOXEL_SIZE_UM, 'array_axes': 'ZXY (as the hemibodies)',
         'id_to_name': {i: name for i, name in enumerate(segment_names, start=1)}}, indent=1))
    print(f'Saved the final master reference in {master_path} (from {round_folder.name})')


def build_master_reference(overwrite_rounds: bool = False) -> None:
    """
    Run all the steps (see the module docstring).

    Parameters
    ----------
    overwrite_rounds : bool
        If True, rerun the registration rounds even if their master already exists.

    Returns
    -------
    None
    """
    hemibody_folder = MASTER_PATH / 'original_hemibodies'
    segment_folder = MASTER_PATH / 'segment_masks'

    # 1. hemibodies
    hemibodies = []
    for recording in RECORDINGS:
        split_hemibodies.split_recording(recording, hemibody_folder,
                                         target_voxel_size=VOXEL_SIZE_UM)
        hemibodies += [f'{recording.parent.name}_{recording.name}_{side}' for side in ('L', 'R')]

    # 2. muscle clusters (never recomputed: they may have been edited by hand)
    for name in hemibodies:
        if (segment_folder / f'{name}_components.tif').exists():
            print(f'Muscle clusters already found for {name}, skipping.')
        else:
            label_mask_components.label_hemibody(hemibody_folder / f'{name}.ome.tif',
                                                 segment_folder, VOXEL_SIZE_UM)

    # 3. manual step: leg segments of the reference's clusters
    if REFERENCE not in hemibodies:
        raise ValueError(f'REFERENCE {REFERENCE} is not one of the hemibodies of RECORDINGS: '
                         f'{hemibodies}')
    recording, side = REFERENCE.rsplit('_', 1)
    table_path = segment_folder / f'{recording}_segment_ids.json'
    has_table = table_path.exists() and side in json.loads(table_path.read_text())
    if not has_table:
        print(f'\nManual step needed: open {segment_folder / f"{REFERENCE}_image.tif"} and '
              f'{segment_folder / f"{REFERENCE}_components.tif"} in napari, and write the leg '
              f'segment of each label in {table_path}, as:\n'
              '{"segments": ["F_coxa/thorax", ..., "head"],\n'
              f' "{side}": {{"F_coxa/thorax": [3], "F_femur": [1], ..., "head": [9, 12]}}}}\n'
              'Then run this script again.')
        return

    # 4. registration rounds
    others = [name for name in hemibodies if name != REFERENCE]
    for index in range(1, N_ROUNDS + 1):
        out_folder = _round_folder(index)
        if (MASTER_PATH / out_folder / 'master_mean.tif').exists() and not overwrite_rounds:
            print(f'Round {index} already done ({out_folder}), skipping.')
            continue
        print(f'\nRound {index} -> {out_folder}')
        registration_round.registration_round(
            MASTER_PATH, REFERENCE, others,
            template_folder=None if index == 1 else Path(_round_folder(index - 1)),
            out_folder=Path(out_folder))
    save_final_master(MASTER_PATH / _round_folder(N_ROUNDS))


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    parser = argparse.ArgumentParser(description='Build the master reference (all steps).')
    parser.add_argument('--overwrite_rounds', action='store_true',
                        help='Rerun the registration rounds even if they exist.')
    args = parser.parse_args()
    build_master_reference(args.overwrite_rounds)
