"""
Script to make a video of the median projections of the runs of a long SCAPE experiment, to
check their motion correction by eye: the first long run in green with each run in turn on top
in magenta (white where they overlap), `RUN_S` seconds each, side by side for the original
median projections (`{run#}_median.ome.tif`, left), the direct alignment to the first run
(`{run#}_median_aligned_direct.ome.tif`, middle) and the chained alignment via the previous runs
(`{run#}_median_aligned_chained.ome.tif`, right), all made by `motion_across_runs.py` in
`{exp}/00_motion-analysis/median_projections` (the aligned ones on the first run's grid).

Each column shows on top the maximum projection over z, with x (roughly the long axis of the
pupa) vertical and y horizontal (the image of the projections turned 90 degrees clockwise, not
mirrored), and below it the maximum projection over x (z vertical, the same y), at `PROJ_UM`
pixels, cropped to where any run has data. Each run is captioned `{run#}_{hours}hAPF` (hours
after puparium formation from the experiment folder name, `common.preanalysis.get_time_0`).
Gray levels are set per run and column (99.7th percentile, as
`visualization/make_master_videos.py`), so brightness changes across development do not hide
the alignment.

Saved as `{exp}/00_motion-analysis/median_overlay.mp4`.

Run from the terminal: `python video_motion_runs.py exp_folder`.

video_motion_runs: make the video.
"""

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import tifffile

import scape_utils
from common.preanalysis import get_hours_apf, get_time_0
from motion_across_runs import OUT_FOLDER_NAME

sys.path.insert(0, str(Path(__file__).parents[1] / 'visualization'))
import make_master_videos as mmv  # noqa: E402

RUN_S = 0.25   # seconds each run is shown
PROJ_UM = 1.0  # pixel size of the projections
GAP_PX = 2     # white lines between the panels
COLUMNS = {'original': '', 'direct': '_aligned_direct', 'chained': '_aligned_chained'}


def video_motion_runs(exp_path: str | Path) -> None:
    """
    Make `{exp}/00_motion-analysis/median_overlay.mp4` (see module docstring).

    Parameters
    ----------
    exp_path : str | Path
        Experiment folder, with `motion_across_runs.py` already run on it.

    Returns
    -------
    None
    """
    exp_path = Path(exp_path)
    out_path = exp_path / OUT_FOLDER_NAME
    table = pd.read_csv(out_path / 'motion_across_runs.csv', index_col=0, parse_dates=['start'])
    time_0 = get_time_0(exp_path)
    voxel_size = scape_utils.load_voxel_size(exp_path / table.index[0])
    # project at full resolution, then resample the 2D images to PROJ_UM pixels
    zoom_top = (voxel_size['x'] / PROJ_UM, voxel_size['y'] / PROJ_UM)     # (x, y)
    zoom_bottom = (voxel_size['z'] / PROJ_UM, voxel_size['y'] / PROJ_UM)  # (z, y)

    captions = []
    tops, bottoms = {c: [] for c in COLUMNS}, {c: [] for c in COLUMNS}  # column -> per run
    for i, (run, start) in enumerate(table['start'].items()):
        run_id = next(t for t in run.split('_') if t.startswith('run'))
        captions.append(f'{run_id}_{get_hours_apf(time_0, start.to_pydatetime())}hAPF')
        print(f'Loading {captions[-1]}...')
        for column, suffix in COLUMNS.items():
            name = f'{run_id}_median' + ('' if i == 0 else suffix)  # the first run is the reference
            volume = tifffile.imread(out_path / 'median_projections' / f'{name}.ome.tif')  # (z, y, x)
            # x vertical, y horizontal reversed (the (y, x) image turned 90 degrees clockwise);
            # below, the max over x, sharing the horizontal y
            top = volume.max(axis=0).T[:, ::-1]
            bottom = volume.max(axis=2)[:, ::-1]
            tops[column].append(ndi.zoom(top.astype(np.float32), zoom_top, order=1))
            bottoms[column].append(ndi.zoom(bottom.astype(np.float32), zoom_bottom, order=1))

    # crop to where any run of any column has data; the same y crop for top and bottom
    box_top = mmv._bbox(np.any([t > 0 for c in COLUMNS for t in tops[c]], axis=0))
    box_bottom = mmv._bbox(np.any([b > 0 for c in COLUMNS for b in bottoms[c]], axis=0))
    y = slice(min(box_top[1].start, box_bottom[1].start), max(box_top[1].stop, box_bottom[1].stop))
    gap = np.ones((GAP_PX, y.stop - y.start), np.float32)
    images = {}  # column -> per run, top and bottom stacked, 0..1
    for column in COLUMNS:
        images[column] = []
        for top, bottom in zip(tops[column], bottoms[column]):
            top, bottom = top[box_top[0], y], bottom[box_bottom[0], y]
            vmax = mmv._vmax([top, bottom])
            images[column].append(np.concatenate([mmv._gray(top, vmax), gap,
                                                  mmv._gray(bottom, vmax)]))

    mmv.CAPTION_LINES = 3
    frames = []
    for r, caption in enumerate(captions):
        columns = [mmv._to_frame(np.clip(images[c][0][..., None] * mmv.GREEN
                                         + images[c][r][..., None] * mmv.MAGENTA, 0, 1),
                                 [(c, (1, 1, 1)), (captions[0], mmv.GREEN),
                                  (caption, mmv.MAGENTA)])
                   for c in COLUMNS]
        separator = np.full((columns[0].shape[0], GAP_PX, 3), 255, np.uint8)
        frames.append(np.concatenate([columns[0], separator, columns[1], separator, columns[2]],
                                     axis=1))

    path = out_path / 'median_overlay.mp4'
    with imageio.get_writer(path, fps=1 / RUN_S, codec='libx264', quality=8, macro_block_size=2,
                            pixelformat='yuv420p') as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f'Saved {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Green-magenta overlay video of the original, '
                                                 'direct- and chain-aligned median projections '
                                                 'of the runs of a SCAPE experiment.')
    parser.add_argument('exp_path', help='Experiment folder.')
    args = parser.parse_args()
    video_motion_runs(args.exp_path)
