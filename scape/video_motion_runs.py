"""
Script to make a video of the median projections of the runs of a long SCAPE experiment after
motion correction, to check the alignment by eye: the first long run's median projection and
the other runs' chain-aligned ones (`{run#}_median_aligned_chained.ome.tif`, all on the first
run's grid, made by `motion_across_runs.py` in `{exp}/00_motion-analysis/median_projections`)
are shown one after the other, `RUN_S` seconds each, as maximum-intensity projections along z
of the volume rotating about x (roughly the long axis of the pupa), shown vertical, with y
horizontal (the image of the projections turned 90 degrees clockwise, not mirrored) (the
rotation, contrast and video writing of
`visualization/make_master_videos.py`). Each run is captioned `{run#}_{hours}hAPF` (hours after
puparium formation from the experiment folder name, `common.preanalysis.get_time_0`).

The volumes are resampled to `ISO_UM` isotropic voxels (so the rotation is true to scale) and
cropped to where any of them has data. Gray levels are set per run (99.7th percentile of its
projections), so brightness changes across development do not hide the alignment.

Saved as `{exp}/00_motion-analysis/median_chained_rotation.mp4`. With `original=True`
(`--original`), the same video of the original median projections (`{run#}_median.ome.tif`, no
alignment) is saved as `median_original_rotation.mp4`, to compare.

Run from the terminal: `python video_motion_runs.py exp_folder [--original]`.

video_motion_runs: make the video.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import tifffile

import scape_utils
from common.preanalysis import get_hours_apf, get_time_0
from motion_across_runs import OUT_FOLDER_NAME

sys.path.insert(0, str(Path(__file__).parents[1] / 'visualization'))
import make_master_videos as mmv  # noqa: E402

RUN_S = 1.0   # seconds each run is shown
ISO_UM = 2.0  # isotropic voxel size of the rendered volumes


def video_motion_runs(exp_path: str | Path,
                      original: bool = False) -> None:
    """
    Make `{exp}/00_motion-analysis/median_chained_rotation.mp4`, or with `original`
    `median_original_rotation.mp4` (see module docstring).

    Parameters
    ----------
    exp_path : str | Path
        Experiment folder, with `motion_across_runs.py` already run on it.
    original : bool
        Whether to show the original median projections instead of the chain-aligned ones.

    Returns
    -------
    None
    """
    exp_path = Path(exp_path)
    out_path = exp_path / OUT_FOLDER_NAME
    table = pd.read_csv(out_path / 'motion_across_runs.csv', index_col=0, parse_dates=['start'])
    time_0 = get_time_0(exp_path)
    voxel_size = scape_utils.load_voxel_size(exp_path / table.index[0])
    zoom = (voxel_size['z'] / ISO_UM, voxel_size['x'] / ISO_UM, voxel_size['y'] / ISO_UM)

    captions, volumes = [], []
    for i, (run, start) in enumerate(table['start'].items()):
        run_id = next(t for t in run.split('_') if t.startswith('run'))
        name = f'{run_id}_median' + ('' if i == 0 or original else '_aligned_chained')
        print(f'Loading {name}...')
        volume = tifffile.imread(out_path / 'median_projections' / f'{name}.ome.tif')  # (z, y, x)
        # to (z, x, y reversed): x is the rotation axis (vertical), and the image of (y, x) turned
        # 90 degrees clockwise (transpose + flip, so not mirrored)
        volume = volume.transpose(0, 2, 1)[:, :, ::-1]
        volumes.append(ndi.zoom(volume.astype(np.float32), zoom, order=1))
        captions.append(f'{run_id}_{get_hours_apf(time_0, start.to_pydatetime())}hAPF')
    box = mmv._bbox(np.any([v > 0 for v in volumes], axis=0))
    volumes = [v[box] for v in volumes]

    mips = []
    for caption, volume in zip(captions, volumes):
        print(f'Rendering {caption}...')
        run_mips = mmv._rotated_mips(volume)
        vmax = mmv._vmax(run_mips)
        mips.append([mmv._gray(mip, vmax) for mip in run_mips])

    mmv.CAPTION_LINES = 1
    frames_per_run = round(RUN_S * mmv.FPS)
    frames = (mmv._to_frame(np.repeat(mips[f // frames_per_run][mmv._angle_index(f)][..., None],
                                      3, axis=2),
                            [(captions[f // frames_per_run], (1, 1, 1))])
              for f in range(len(mips) * frames_per_run))
    mmv._write_video(out_path / f'median_{"original" if original else "chained"}_rotation.mp4',
                     frames)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video of the chain-aligned median projections '
                                                 'of the runs of a SCAPE experiment.')
    parser.add_argument('exp_path', help='Experiment folder.')
    parser.add_argument('--original', action='store_true',
                        help='Show the original median projections (no alignment).')
    args = parser.parse_args()
    video_motion_runs(args.exp_path, args.original)
