"""
Pre-analysis plots of a long recording (one pupa recorded many times over several days), shared
by the FlyTrack and SCAPE preprocessing (`flytrack/preprocess_flytrack.py`,
`scape/preprocess_scape_exp.py`), saved in `<experiment folder>/PREANALYSIS_FOLDER`: grids of all
the recordings' activity traces and of their max and median projections, to check a long
recording at a glance.

The experiment folder name ends with the puparium formation time (`..._MM-DD_HHhMM`, optionally
followed by a suffix such as `_FAIL`), from which each recording's hours after puparium formation
(hAPF) is computed.

A trace is a dict with keys 'title' (e.g. '72 hAPF'), 'activity' (1D array of the mean
fluorescence, evenly sampled), 'fs' (sampling rate in Hz), and optionally 'hours' (hours APF)
and 'name' (recording folder, relative to the experiment folder).

get_time_0: puparium formation time from the experiment folder name.
get_hours_apf: hours after puparium formation of a recording.
save_activity_summary: grids of all the activity traces (shared and own y scales).
save_image_grid: grids of images (e.g. every recording's max projection), with a shared
    contrast.
"""

import datetime
import os
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

PREANALYSIS_FOLDER = '00_preanalysis'         # in the experiment (main) folder
ACT_SUMMARY_NAME = 'activity_{}.png'          # in PREANALYSIS_FOLDER; {} is 'shared' or 'own'
ACT_SUMMARY_COLS = 3
GRID_MAX_ROWS = 4                             # image grids: at most this many rows and
GRID_MAX_COLS = 12                            # columns per figure
CONTRAST_PCT = (0.2, 99.8)                    # per-image low/high contrast percentiles
# image grid contrast: vmin/vmax are these percentiles (across images) of the per-image
# CONTRAST_PCT low/high percentiles
GRID_CONTRAST_PCT = (10, 90)


def get_time_0(folder: str | os.PathLike) -> datetime.datetime | None:
    """
    Extract the puparium formation time from the name of an experiment folder.

    Uses the last `MM-DD_HHhMM` in the folder name, so a suffix after it (e.g. `_FAIL`) is
    allowed. The year is not encoded, so it is left as 1900 and set per recording in
    `get_hours_apf()`.

    Parameters
    ----------
    folder : str or os.PathLike
        Experiment folder whose name ends with the puparium formation time.

    Returns
    -------
    datetime.datetime or None
        Puparium formation time, or None if the folder name does not match.
    """
    matches = re.findall(r'(?<!\d)\d{2}-\d{2}_\d{2}h\d{2}(?!\d)', Path(folder).name)
    try:
        return datetime.datetime.strptime(matches[-1], '%m-%d_%Hh%M')
    except (IndexError, ValueError):
        return None


def get_hours_apf(time_0: datetime.datetime | None,
                  time_t: datetime.datetime | None) -> int | None:
    """
    Calculate the hours after puparium formation (APF) of a recording.

    `time_0` carries no year, so it is given the recording's year (or the year before, if that
    would put it after the recording, e.g. across New Year).

    Parameters
    ----------
    time_0 : datetime.datetime or None
        Puparium formation time, see `get_time_0()`.
    time_t : datetime.datetime or None
        Start time of the recording.

    Returns
    -------
    int or None
        Hours APF, rounded to the nearest hour, or None if either time is unknown.
    """
    if time_0 is None or time_t is None:
        return None
    t0 = time_0.replace(year=time_t.year)
    if t0 > time_t:
        t0 = t0.replace(year=time_t.year - 1)
    return int(np.round((time_t - t0).total_seconds() / 3600))


def save_activity_summary(traces: list[dict],
                          folder: str | os.PathLike,
                          suptitle: str = '',
                          ylabel: str = 'Mean fluorescence',
                          linewidth: float = 0.5) -> None:
    """
    Save grids of activity traces in `folder`.

    Every trace is plotted (time in minutes) in a grid of `ACT_SUMMARY_COLS` columns, in the
    given order, each subplot titled with its title. Two figures are saved (see
    `ACT_SUMMARY_NAME`): 'shared', where every subplot has the same x and y limits and the
    subplots in a row touch so the row reads as one continuous trace (y ticks on the first
    column only), and 'own', where each subplot keeps its own y limits and the y scale is hidden.

    Parameters
    ----------
    traces : list of dict
        The traces (see module docstring), in time order.
    folder : str or os.PathLike
        Output folder (created if needed).
    suptitle : str
        Figure title, e.g. the experiment name.
    ylabel : str
        Y axis label of the 'shared' figure.
    linewidth : float
        Line width of the traces (thinner for longer traces).

    Returns
    -------
    None
    """
    if len(traces) == 0:
        return
    matplotlib.use('Agg')  # force a non-interactive backend
    print(f'Saving activity summary of {len(traces)} recordings in {folder}...')
    times = [np.arange(len(tr['activity'])) / tr['fs'] / 60 for tr in traces]
    n_rows = int(np.ceil(len(traces) / ACT_SUMMARY_COLS))
    t_max = max(t[-1] for t in times)
    y_min = min(np.min(tr['activity']) for tr in traces)
    y_max = max(np.max(tr['activity']) for tr in traces)
    y_pad = 0.03 * (y_max - y_min)

    for scale in ('shared', 'own'):
        fig, axes = plt.subplots(n_rows, ACT_SUMMARY_COLS, figsize=(18, 1.6 * n_rows),
                                 squeeze=False)
        for i, ax in enumerate(axes.flat):
            if i >= len(traces):
                ax.axis('off')
                continue
            ax.plot(times[i], traces[i]['activity'], linewidth=linewidth)
            ax.set_title(traces[i]['title'], fontsize=9, pad=2)
            if scale == 'shared':
                ax.set_xlim(0, t_max)
                ax.set_ylim(y_min - y_pad, y_max + y_pad)
                if i % ACT_SUMMARY_COLS != 0:  # y ticks on the first column only
                    ax.tick_params(left=False, labelleft=False)
            else:
                ax.set_xlim(0, times[i][-1])
                ax.tick_params(left=False, labelleft=False)
            if i + ACT_SUMMARY_COLS < len(traces):  # x labels on the last plot of each column
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel('Time (min)', fontsize=8)
            ax.tick_params(labelsize=7)

        if scale == 'shared':
            axes[n_rows // 2, 0].set_ylabel(ylabel)
        fig.suptitle(suptitle, y=1.0)
        fig.tight_layout()
        if scale == 'shared':  # subplots in a row touch: one continuous trace
            fig.subplots_adjust(wspace=0)
        os.makedirs(folder, exist_ok=True)
        fig.savefig(os.path.join(folder, ACT_SUMMARY_NAME.format(scale)), dpi=200,
                    bbox_inches='tight')
        plt.close(fig)


def save_image_grid(images: list[np.ndarray],
                    titles: list[str],
                    save_path: str | os.PathLike,
                    aspect: float = 1.0,
                    suptitle: str = '') -> None:
    """
    Save grids of 2D images (e.g. every recording's max projection), each titled.

    Each figure holds at most `GRID_MAX_ROWS` rows x `GRID_MAX_COLS` columns (fewer rows if there
    are few images); if there are more images, several figures are saved (`save_path` with
    `_1`, `_2`, ... before the extension). All images share one contrast: from each image's
    `CONTRAST_PCT` low and high percentiles, vmin is the `GRID_CONTRAST_PCT[0]` percentile of
    the lows and vmax the `GRID_CONTRAST_PCT[1]` percentile of the highs.

    Parameters
    ----------
    images : list of np.ndarray
        The 2D images, in display order.
    titles : list of str
        One title per image.
    save_path : str or os.PathLike
        Path of the png (its folder is created if needed).
    aspect : float
        Pixel aspect ratio (height / width of a pixel, e.g. the row spacing over the column
        spacing in um), passed to `imshow`.
    suptitle : str
        Figure title, e.g. the experiment name.

    Returns
    -------
    None
    """
    if len(images) == 0:
        return
    matplotlib.use('Agg')  # force a non-interactive backend
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Saving {save_path.stem} of {len(images)} recordings...')

    # shared contrast from the per-image low/high percentiles
    lows, highs = zip(*(np.nanpercentile(img, CONTRAST_PCT) for img in images))
    vmin = float(np.percentile(lows, GRID_CONTRAST_PCT[0]))
    vmax = float(np.percentile(highs, GRID_CONTRAST_PCT[1]))

    # grid layout (same for all figures) and split of the images into figures
    cols = min(GRID_MAX_COLS, int(np.ceil(len(images) / GRID_MAX_ROWS)))
    rows = min(GRID_MAX_ROWS, int(np.ceil(len(images) / cols)))
    n_per_fig = rows * cols
    fig_chunks = [range(i, min(i + n_per_fig, len(images)))
                  for i in range(0, len(images), n_per_fig)]

    for k, chunk in enumerate(fig_chunks):
        fig, axes = plt.subplots(rows, cols, figsize=(22, 12 * rows / GRID_MAX_ROWS),
                                 squeeze=False)
        axes = axes.flatten()
        for ax in axes:
            ax.axis('off')
        for i, idx in enumerate(chunk):
            axes[i].imshow(images[idx], cmap='gray', vmin=vmin, vmax=vmax, aspect=aspect,
                           interpolation='none')
            axes[i].set_title(titles[idx], fontsize=10)
        if suptitle:
            fig.suptitle(suptitle)
        fig.tight_layout()
        path = save_path
        if len(fig_chunks) > 1:  # figure number in the name if there are several figures
            path = save_path.with_name(f'{save_path.stem}_{k + 1}{save_path.suffix}')
        fig.savefig(path, dpi=300)
        plt.close(fig)
