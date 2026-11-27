"""
Script to measure how much a sample moves across the runs of a long SCAPE experiment: the median
projection of every `sample#_run#` folder (except `_short` runs, `sample#_combined`, runs whose
projections are not computed yet, and runs whose grid differs from the first long run's, i.e.
processed with or without the flyback frames unlike it, unless their `analysis_dat_old` folder has
the same grid, which is then used) is
rigidly registered to that of the first long run with `register_median_proj.register_median_proj`
(which also saves `median_proj_aligned.*` in each run's analysis folder), and, without saving, to
that of the previous run. The run-to-previous transforms are chained back to the first run and
compared with the direct run-to-first transform, at the corners of the tissue bounding box (the
non-zero voxels) of the first run: if development (growth) biases the direct registration, the
two disagree by more than the slow random-walk drift that chaining noisy steps gives on its own.

Outputs, in `{exp}/00_motion-analysis`:
- `motion_across_runs.csv`: per run, the rotation (deg, about x/y/z in sitk order), the
  translation (um, x/y/z), the correlation with the reference after alignment, and the time the
  registration took (s), to the first long run, and the same with a `prev_` prefix to the previous
  run (except time), and with a `chain_` prefix (except correlation and time) the chained
  transform expressed about the same center as the direct one (so they compare); `disp_um` and `chain_disp_um`: the largest displacement of a bounding box
  corner by the direct and by the chained transform, and `chain_diff_um`: the largest distance
  between where the two send a corner,
- `motion_across_runs.png`: the same over the run start time (from the folder name), the
  direct alignment solid and the chained one dashed (`plot_motion`; replot only with
  `--plot_only`); also copied to `_analysis/motion-analysis/motion_across_runs_YYYYMMDD.png`
  next to the experiment folders (in the SCAPE folder), to compare experiments,
- `median_projections/`: `{run#}_median.ome.tif` (the median projection of each run, a copy),
  `{run#}_median_aligned_direct.ome.tif` (a copy of the direct alignment's
  `median_proj_aligned.ome.tif`) and `{run#}_median_aligned_chained.ome.tif` (resampled onto the
  first run's grid with the chained transform), each with its transform as `.tfm`, to compare
  runs, e.g. with `video_motion_runs.py`.

Can be run from the terminal (`python motion_across_runs.py exp_folder`) or imported.

motion_across_runs: register all runs of an experiment, save and return the table.
plot_motion: plot the saved table.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil
import SimpleITK as sitk
import tifffile

import scape_utils
from common.power_spectrum import COMMON_FOLDER
from register_median_proj import _load_median_proj_sitk, register_median_proj, save_aligned

OUT_FOLDER_NAME = '00_motion-analysis'
COMMON_SUBFOLDER = 'motion-analysis'  # in COMMON_FOLDER, for the plots of every experiment
# y limits of the 4 panels of `plot_motion`, the same for every fly to compare them: they cover
# the runs before eclosion of the first 4 experiments (larger shifts near eclosion fall off)
YLIMS = ((-40, 40), (-2, 8), (0.7, 1.0), (0, 130))


prev_keys = ('rot_x', 'rot_y', 'rot_z', 'tx', 'ty', 'tz', 'correlation')


def _motion_row(rigid, correlation: float) -> dict:
    """
    Rotation (deg), translation (um) and correlation of one registration, keyed as `prev_keys`.
    """
    angles = np.degrees((rigid.GetAngleX(), rigid.GetAngleY(), rigid.GetAngleZ()))
    return dict(zip(prev_keys, (*angles, *rigid.GetTranslation(), correlation)))


def _as_euler(transform: sitk.Transform,
              center: tuple) -> sitk.Euler3DTransform:
    """
    Express a rigid transform (e.g. a composite of rigid transforms) as one Euler transform
    about `center`, so its angles and translation compare with those of another transform with
    that center.
    """
    offset = np.array(transform.TransformPoint((0., 0., 0.)))
    matrix = np.column_stack([np.array(transform.TransformPoint(tuple(e))) - offset
                              for e in np.eye(3)])
    euler = sitk.Euler3DTransform()
    euler.SetCenter(center)
    euler.SetMatrix(matrix.ravel().tolist())
    euler.SetTranslation(tuple(np.array(transform.TransformPoint(center)) - np.array(center)))
    return euler


def plot_motion(exp_path: str | Path) -> None:
    """
    Plot `{exp}/00_motion-analysis/motion_across_runs.csv` as `motion_across_runs.png`: the
    rotation and translation of the direct (solid) and chained (dashed) alignment to the first
    run, the correlations, and the largest tissue bounding box corner displacements, with the
    same y limits for every experiment (`YLIMS`).

    Parameters
    ----------
    exp_path : str | Path
        Experiment folder.

    Returns
    -------
    None
    """
    exp_path = Path(exp_path)
    out_path = exp_path / OUT_FOLDER_NAME
    table = pd.read_csv(out_path / 'motion_across_runs.csv', index_col=0, parse_dates=['start'])
    fixed_name = table.index[0]
    hours = (table['start'] - table['start'].iloc[0]).dt.total_seconds() / 3600
    fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
    for ax, cols, unit in ((axes[0], ('tx', 'ty', 'tz'), 'translation (um)'),
                           (axes[1], ('rot_x', 'rot_y', 'rot_z'), 'rotation (deg)')):
        for col, colour in zip(cols, ('tab:blue', 'tab:orange', 'tab:green')):
            ax.plot(hours, table[col], 'o-', color=colour, label=f'{col[-1]} direct')
            ax.plot(hours, table[f'chain_{col}'], 'o--', color=colour, mfc='none',
                    label=f'{col[-1]} chained')
        ax.set_ylabel(unit)
        ax.axhline(0, color='grey', lw=0.5)
        ax.legend(title='axis', ncol=3, fontsize='small')
    axes[2].plot(hours, table['correlation'], 'ko-', label=f'with {fixed_name}')
    axes[2].plot(hours, table['prev_correlation'], 'o--', color='grey', label='with previous run')
    axes[2].set_ylabel('correlation after\nalignment')
    axes[2].legend()
    axes[3].plot(hours, table['disp_um'], 'ko-', label=f'direct to {fixed_name}')
    axes[3].plot(hours, table['chain_disp_um'], 'o--', color='grey', label='chained via previous runs')
    axes[3].plot(hours, table['chain_diff_um'], 'o-', color='tab:red', label='direct vs chained')
    axes[3].set_ylabel('max tissue corner\ndisplacement (um)')
    axes[3].legend()
    axes[3].set_xlabel(f'hours since {fixed_name}')
    for ax, ylim in zip(axes, YLIMS):
        ax.set_ylim(ylim)
    fig.suptitle(f'{exp_path.name}: rigid motion of the median projection')
    fig.tight_layout()
    fig.savefig(out_path / 'motion_across_runs.png', dpi=150)
    plt.close(fig)
    print(f'Saved motion_across_runs.png to {out_path}')
    common = exp_path.parent / COMMON_FOLDER / COMMON_SUBFOLDER
    common.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out_path / 'motion_across_runs.png',
                    common / f'motion_across_runs_{exp_path.name[:8]}.png')
    print(f'Copied it to {common}')


def motion_across_runs(exp_path: str | Path) -> pd.DataFrame:
    """
    Rigidly register the median projection of every long run of `exp_path` to the first long
    run and to the previous run, and save the transforms, correlations and a plot in
    `exp_path/00_motion-analysis`.

    Parameters
    ----------
    exp_path : str | Path
        Experiment folder containing `sample#_run#_yyyymmdd_hhmm` folders.

    Returns
    -------
    pd.DataFrame
        One row per run (see module docstring), indexed by run folder name.
    """
    exp_path = Path(exp_path)
    runs = sorted(p for p in exp_path.glob('sample*_run*')
                  if p.is_dir() and 'short' not in p.name)
    # don't compute missing projections here (slow, from the raw data over the network)
    missing = [p for p in runs
               if not (scape_utils.analysis_path_for(p) / scape_utils.PROJECTION_FILE_NAME).exists()]
    if missing:
        print(f'Skipping runs without projections: {[p.name for p in missing]}')
    runs = [p for p in runs if p not in missing]
    # runs processed with and without the flyback frames have different grids: for those, use
    # the older analysis folder if it has the reference run's grid, else skip them
    # shortcut: only needed until all runs are processed the same way
    def median_shape(analysis_path: Path) -> tuple:
        return tifffile.TiffFile(analysis_path / 'median_proj.ome.tif').series[0].shape
    ref_shape = median_shape(scape_utils.analysis_path_for(runs[0]))
    sources = {}  # run -> folder passed to `register_median_proj`
    for p in runs:
        sources[p] = p
        if median_shape(scape_utils.analysis_path_for(p)) != ref_shape:
            old = p / 'analysis_dat_old'
            if (old / 'median_proj.ome.tif').exists() and median_shape(old) == ref_shape:
                sources[p] = old
                print(f'Using {old.name} for {p.name} (same grid as {runs[0].name})')
            else:
                del sources[p]
                print(f'Skipping {p.name}: different grid than {runs[0].name}')
    runs = list(sources)
    assert runs, f'No long runs found in {exp_path}'
    fixed = runs[0]

    # tissue bounding box corners of the first run, in sitk physical (x, y, z) um, as in
    # `register_median_proj._load_median_proj_sitk` (array (z, y, x), origin 0)
    voxel_size = scape_utils.load_voxel_size(fixed)
    nonzero = np.argwhere(scape_utils.load_proj(fixed, 'median_proj').transpose(0, 2, 1) > 0)
    lo, hi = nonzero.min(0), nonzero.max(0)
    corners = np.array([(x * voxel_size['x'], y * voxel_size['y'], z * voxel_size['z'])
                        for z in (lo[0], hi[0]) for y in (lo[1], hi[1]) for x in (lo[2], hi[2])])
    chain = []  # run-to-previous transforms, latest first (CompositeTransform applies the last first)
    out_path = exp_path / OUT_FOLDER_NAME
    proj_path = out_path / 'median_projections'
    proj_path.mkdir(parents=True, exist_ok=True)
    fixed_image = _load_median_proj_sitk(sources[fixed])

    def run_id(p: Path) -> str:
        return next(t for t in p.name.split('_') if t.startswith('run'))

    rows = []
    for run in runs:
        start = datetime.strptime('_'.join(run.name.split('_')[-2:]), '%Y%m%d_%H%M')
        row = dict(run=run.name, start=start)
        analysis = scape_utils.analysis_path_for(run) if sources[run] == run else sources[run]
        shutil.copy(analysis / 'median_proj.ome.tif', proj_path / f'{run_id(run)}_median.ome.tif')
        if run == fixed:
            row.update({k: 0. for k in prev_keys}, correlation=1., seconds=0.)
            row.update({f'prev_{k}': v for k, v in row.items() if k in prev_keys})
            row.update(disp_um=0., chain_disp_um=0., chain_diff_um=0.)
            row.update({f'chain_{k}': 0. for k in prev_keys if k != 'correlation'})
            rows.append(row)
            continue
        t0 = time.perf_counter()
        direct, correlation = register_median_proj(sources[run], sources[fixed])
        row.update(_motion_row(direct, correlation))
        row['seconds'] = time.perf_counter() - t0
        print(f'  took {row["seconds"]:.0f} s')
        for ext in ('.ome.tif', '.tfm'):
            shutil.copy(analysis / f'median_proj_aligned{ext}',
                        proj_path / f'{run_id(run)}_median_aligned_direct{ext}')
        if run != runs[1]:  # the previous run of runs[1] is the first run, already done
            prev, correlation = register_median_proj(sources[run],
                                                     sources[runs[runs.index(run) - 1]],
                                                     save=False)
        else:
            prev = direct
        row.update({f'prev_{k}': v for k, v in _motion_row(prev, correlation).items()})
        # each transform maps points of its fixed run to its moving run, so chaining goes forward
        chain.insert(0, prev)
        chained_transform = sitk.CompositeTransform(chain)
        aligned = sitk.Resample(_load_median_proj_sitk(sources[run]), fixed_image,
                                chained_transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        save_aligned(aligned, chained_transform,
                     proj_path / f'{run_id(run)}_median_aligned_chained')
        chained = np.array([chained_transform.TransformPoint(tuple(c)) for c in corners])
        direct_corners = np.array([direct.TransformPoint(tuple(c)) for c in corners])
        row['disp_um'] = np.linalg.norm(direct_corners - corners, axis=1).max()
        row['chain_disp_um'] = np.linalg.norm(chained - corners, axis=1).max()
        row['chain_diff_um'] = np.linalg.norm(chained - direct_corners, axis=1).max()
        chain_euler = _as_euler(chained_transform, direct.GetCenter())
        row.update({f'chain_{k}': v for k, v in _motion_row(chain_euler, np.nan).items()
                    if k != 'correlation'})
        print(f'  max corner displacement: direct {row["disp_um"]:.2f} um, chained '
              f'{row["chain_disp_um"]:.2f} um, difference {row["chain_diff_um"]:.2f} um')
        rows.append(row)
    table = pd.DataFrame(rows).set_index('run')

    table.to_csv(out_path / 'motion_across_runs.csv')
    print(f'Saved motion_across_runs.csv to {out_path}')

    plot_motion(exp_path)
    return table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rigidly register the median projection of '
                                                 'every run of a SCAPE experiment to the first '
                                                 'long run and plot the motion.')
    parser.add_argument('exp_path', help='Experiment folder.')
    parser.add_argument('--plot_only', action='store_true',
                        help='Only replot the saved motion_across_runs.csv.')
    args = parser.parse_args()
    if args.plot_only:
        plot_motion(args.exp_path)
    else:
        motion_across_runs(args.exp_path)
