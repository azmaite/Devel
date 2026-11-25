"""
Script to measure how much a sample moves across the runs of a long SCAPE experiment: the median
projection of every `sample#_run#` folder (except `_short` runs and `sample#_combined`) is
rigidly registered to that of the first long run with `register_median_proj.register_median_proj`
(which also saves `median_proj_aligned.*` in each run's analysis folder).

Outputs, in `{exp}/00_motion-analysis`:
- `motion_across_runs.csv`: per run, the rotation (deg, about x/y/z in sitk order), the
  translation (um, x/y/z), the correlation with the reference after alignment, and the time the
  registration took (s),
- `motion_across_runs.png`: the same over the run start time (from the folder name).

Can be run from the terminal (`python motion_across_runs.py exp_folder`) or imported.

motion_across_runs: register all runs of an experiment, save and return the table.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from register_median_proj import register_median_proj

OUT_FOLDER_NAME = '00_motion-analysis'


def motion_across_runs(exp_path: str | Path) -> pd.DataFrame:
    """
    Rigidly register the median projection of every long run of `exp_path` to the first long
    run, and save the transforms, correlations and a plot in `exp_path/00_motion-analysis`.

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
    assert runs, f'No long runs found in {exp_path}'
    fixed = runs[0]

    rows = []
    for run in runs:
        start = datetime.strptime('_'.join(run.name.split('_')[-2:]), '%Y%m%d_%H%M')
        if run == fixed:
            rows.append(dict(run=run.name, start=start, rot_x=0., rot_y=0., rot_z=0.,
                             tx=0., ty=0., tz=0., correlation=1., seconds=0.))
            continue
        t0 = time.perf_counter()
        rigid, correlation = register_median_proj(run, fixed)
        seconds = time.perf_counter() - t0
        print(f'  took {seconds:.0f} s')
        angles = np.degrees((rigid.GetAngleX(), rigid.GetAngleY(), rigid.GetAngleZ()))
        rows.append(dict(run=run.name, start=start,
                         rot_x=angles[0], rot_y=angles[1], rot_z=angles[2],
                         tx=rigid.GetTranslation()[0], ty=rigid.GetTranslation()[1],
                         tz=rigid.GetTranslation()[2],
                         correlation=correlation, seconds=seconds))
    table = pd.DataFrame(rows).set_index('run')

    out_path = exp_path / OUT_FOLDER_NAME
    out_path.mkdir(exist_ok=True)
    table.to_csv(out_path / 'motion_across_runs.csv')

    hours = (table['start'] - table['start'].iloc[0]).dt.total_seconds() / 3600
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for col in ('tx', 'ty', 'tz'):
        axes[0].plot(hours, table[col], 'o-', label=col[1])
    axes[0].set_ylabel('translation (um)')
    for col in ('rot_x', 'rot_y', 'rot_z'):
        axes[1].plot(hours, table[col], 'o-', label=col[-1])
    axes[1].set_ylabel('rotation (deg)')
    axes[2].plot(hours, table['correlation'], 'ko-')
    axes[2].set_ylabel(f'correlation with\n{fixed.name}')
    axes[2].set_xlabel(f'hours since {fixed.name}')
    for ax in axes[:2]:
        ax.axhline(0, color='grey', lw=0.5)
        ax.legend(title='axis')
    fig.suptitle(f'{exp_path.name}: rigid motion of the median projection')
    fig.tight_layout()
    fig.savefig(out_path / 'motion_across_runs.png', dpi=150)
    plt.close(fig)
    print(f'Saved motion_across_runs.csv and .png to {out_path}')
    return table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rigidly register the median projection of '
                                                 'every run of a SCAPE experiment to the first '
                                                 'long run and plot the motion.')
    parser.add_argument('exp_path', help='Experiment folder.')
    args = parser.parse_args()
    motion_across_runs(args.exp_path)
