"""
Frequency analysis of the activity traces (mean fluorescence over time) of a long recording,
shared by the FlyTrack and SCAPE analyses (`analysis/flytrack/`, `analysis/scape/`): which
frequencies the activity is made of, e.g. transients at 3 Hz give a peak at 3 Hz.

Each trace is converted to dF/F (baseline: its `BASELINE_PCT` percentile), and its power spectral
density (PSD) is estimated with Welch's method (`scipy.signal.welch`): the spectra of windows of
`WINDOW_S` seconds (overlapping by half) are averaged. The spectrum goes from 1 / `WINDOW_S` up
to the Nyquist frequency (half the frame rate), so each setup keeps its full range. To reduce
the noise of the estimate, the spectrum (not the trace) is then smoothed over frequency:
averaged in `BINS_PER_DECADE` log-spaced bins per decade for a log frequency axis, or with a
moving average `SMOOTH_HZ` wide for a linear one.

A trace is a dict (see `common.preanalysis`) with keys 'activity' (1D array, evenly sampled),
'fs' (sampling rate in Hz), 'title', and optionally 'hours' (hours APF, used to color the
spectra and as the x axis of the band power) and 'name' (recording folder).

compute_power_spectrum: dF/F power spectrum of one trace, optionally smoothed.
save_power_spectrum_summary: save the power spectra of many traces overlaid on one plot, colored
    by hours APF, with a log or linear frequency axis.
save_band_power: save the power in each of several frequency bands against hours APF, one line
    per band.
save_relative_heatmap: save the normalized power spectra of many traces as a heatmap (hours APF
    against frequency), each frequency relative to its mean over the traces.
save_frequency_analysis: save all of these for the recordings of a long recording before
    eclosion, in `<experiment folder>/FREQUENCY_FOLDER`.
"""

import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, signal

FREQUENCY_FOLDER = '00_frequency_analysis'  # in the experiment (main) folder
PSD_SUMMARY_NAME = 'power_spectrum_{}{}.png'  # {} are the frequency scale and an optional suffix
WINDOW_S = 60.0       # Welch window length (s); also sets the lowest frequency (1 / WINDOW_S)
BASELINE_PCT = 10     # dF/F baseline percentile
BINS_PER_DECADE = 20  # spectrum smoothing for a log frequency axis
SMOOTH_HZ = 0.05      # spectrum smoothing (moving average width, Hz) for a linear frequency axis
CMAP = 'viridis'  # hours APF in the power spectra
# band power: a different sequential colormap than the hours APF one of the power spectra,
# light (slowest band) to dark (fastest band), within this range of the colormap so the lightest
# band stays visible
BAND_CMAP = 'Oranges'
BAND_CMAP_RANGE = (0.35, 1.0)
NORMALIZED_SUFFIX = 'normalized'  # file name suffix of the normalized plots
BAND_POWER_NAME = 'band_power{}.png'  # {} is an optional suffix
HEATMAP_NAME = 'power_spectrum_heatmap_relative{}.png'  # {} is an optional suffix
HEATMAP_CMAP = 'RdBu_r'  # diverging: below / above the mean over the recordings
# frequency bands (Hz) for save_band_power: log-spaced (each 4-5x wide), so the slow and fast
# activity each get their own bands, with one centered on 1 Hz; bands above a trace's Nyquist
# frequency are cut there (or left out)
BANDS = [(0.02, 0.1), (0.1, 0.5), (0.5, 2), (2, 8), (8, 20)]


def compute_power_spectrum(activity: np.ndarray,
                           fs: float,
                           window_s: float = WINDOW_S,
                           smooth: Literal['log', 'linear'] | None = None,
                           normalize: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the power spectral density of the dF/F of an activity trace (Welch's method).

    Parameters
    ----------
    activity : np.ndarray
        1D activity trace (mean fluorescence), evenly sampled.
    fs : float
        Sampling rate (Hz).
    window_s : float
        Welch window length (s), overlapping by half.
    smooth : 'log', 'linear' or None
        Smooth the spectrum over frequency: 'log' averages it in `BINS_PER_DECADE` log-spaced
        bins per decade (for a log frequency axis), 'linear' with a moving average `SMOOTH_HZ`
        wide (for a linear one). None: no smoothing.
    normalize : bool
        Divide the spectrum by the total power (its integral over frequency, before smoothing),
        so it shows how the power is spread over frequency rather than how much there is.

    Returns
    -------
    f : np.ndarray
        Frequencies (Hz), from 1 / `window_s` to the Nyquist frequency (0 Hz left out).
    psd : np.ndarray
        Power spectral density of the dF/F (1/Hz), or the fraction of the total power per Hz if
        `normalize` (1/Hz).
    """
    activity = np.asarray(activity, dtype=float)
    f0 = np.percentile(activity, BASELINE_PCT)
    dff = (activity - f0) / f0
    nperseg = min(len(dff), int(round(window_s * fs)))
    f, psd = signal.welch(dff, fs=fs, window='hann', nperseg=nperseg, noverlap=nperseg // 2,
                          detrend='linear', scaling='density')
    f, psd = f[1:], psd[1:]
    if normalize:
        psd = psd / np.trapezoid(psd, f)

    if smooth == 'log':
        n_bins = int(np.ceil(np.log10(f[-1] / f[0]) * BINS_PER_DECADE)) + 1
        bin_idx = np.digitize(f, np.geomspace(f[0], f[-1] * 1.0001, n_bins)) - 1
        bins = np.unique(bin_idx)
        f = np.array([np.exp(np.log(f[bin_idx == i]).mean()) for i in bins])
        psd = np.array([psd[bin_idx == i].mean() for i in bins])
    elif smooth == 'linear':
        width = int(round(SMOOTH_HZ / (f[1] - f[0])))
        if width > 1:
            psd = ndimage.uniform_filter1d(psd, width, mode='nearest')
    return f, psd


def save_power_spectrum_summary(traces: list[dict],
                                folder: str | os.PathLike,
                                scale: Literal['log', 'linear'] = 'log',
                                window_s: float = WINDOW_S,
                                suptitle: str = '',
                                suffix: str = '',
                                normalize: bool = False) -> str:
    """
    Save the power spectra of many traces (e.g. every recording of a long recording) overlaid on
    one plot, colored by hours APF (or by order, if any trace has no 'hours'), as
    `folder/PSD_SUMMARY_NAME`.

    The power axis is always logarithmic, as the power spans orders of magnitude; `scale` sets the
    frequency axis, and the matching smoothing of the spectra (see `compute_power_spectrum()`).

    Parameters
    ----------
    traces : list of dict
        The traces (see module docstring), in time order.
    folder : str or os.PathLike
        Output folder (created if needed).
    scale : 'log' or 'linear'
        Frequency axis scale.
    window_s : float
        Welch window length (s), see `compute_power_spectrum()`.
    suptitle : str
        Figure title, e.g. the experiment name.
    suffix : str
        Added to the file name (after an underscore), e.g. to tell apart window lengths.
    normalize : bool
        Divide each spectrum by its total power (see `compute_power_spectrum()`), adding
        `NORMALIZED_SUFFIX` to the file name.

    Returns
    -------
    str
        Path of the saved figure.
    """
    spectra = [compute_power_spectrum(tr['activity'], tr['fs'], window_s, smooth=scale,
                                      normalize=normalize) for tr in traces]
    if normalize:
        suffix = '_'.join(filter(None, [NORMALIZED_SUFFIX, suffix]))
    f_min = min(f[0] for f, _ in spectra)
    f_max = max(f[-1] for f, _ in spectra)
    p_min = np.percentile(np.concatenate([p for _, p in spectra]), 0.5)
    p_max = max(p.max() for _, p in spectra)

    hours = [tr.get('hours') for tr in traces]
    by_hours = all(h is not None for h in hours)
    values = np.array(hours if by_hours else range(len(traces)), dtype=float)
    norm = plt.Normalize(values.min(), values.max())
    cmap = plt.get_cmap(CMAP)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for (f, psd), v in zip(spectra, values):
        ax.plot(f, psd, color=cmap(norm(v)), linewidth=0.8, alpha=0.85)
    ax.set_yscale('log')
    ax.set_ylim(p_min, p_max * 1.5)
    ax.set_xscale(scale)
    ax.set_xlim(f_min if scale == 'log' else 0, f_max)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power / total power (1/Hz)' if normalize else 'Power (dF/F$^2$/Hz)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, which='major', color='0.9', linewidth=0.6)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 label='Hours APF' if by_hours else 'Recording', pad=0.02)
    kind = 'normalized power spectrum' if normalize else 'power spectrum'
    fig.suptitle(f'{suptitle}\n{kind} of dF/F ({window_s:g} s windows)', y=1.0, fontsize=11)

    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, PSD_SUMMARY_NAME.format(scale, f'_{suffix}' if suffix else ''))
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return save_path


def save_band_power(traces: list[dict],
                    folder: str | os.PathLike,
                    bands: list[tuple[float, float]] = BANDS,
                    suptitle: str = '',
                    suffix: str = '') -> str:
    """
    Save the power of each frequency band against hours APF, one line per band, as
    `folder/BAND_POWER_NAME`.

    The band power is the integral of the power spectrum (see `compute_power_spectrum()`, 60 s
    windows, not smoothed) over the band, i.e. the variance of the dF/F in that band. Bands
    reaching above the Nyquist frequency of a trace are cut there, and left out for that trace if
    they start above it.

    Parameters
    ----------
    traces : list of dict
        The traces (see module docstring), each with 'hours' (or they are numbered in order).
    folder : str or os.PathLike
        Output folder (created if needed).
    bands : list of (float, float)
        Frequency bands (Hz), low to high.
    suptitle : str
        Figure title, e.g. the experiment name.
    suffix : str
        Added to the file name (after an underscore).

    Returns
    -------
    str
        Path of the saved figure.
    """
    hours = [tr.get('hours') for tr in traces]
    by_hours = all(h is not None for h in hours)
    x = np.array(hours if by_hours else range(len(traces)), dtype=float)
    power = np.full((len(traces), len(bands)), np.nan)
    for i, tr in enumerate(traces):
        f, psd = compute_power_spectrum(tr['activity'], tr['fs'])
        for j, (lo, hi) in enumerate(bands):
            inside = (f >= lo) & (f < hi)
            if inside.sum() > 1:
                power[i, j] = np.trapezoid(psd[inside], f[inside])

    nyquist = max(tr['fs'] for tr in traces) / 2
    cmap = plt.get_cmap(BAND_CMAP)
    lo_c, hi_c = BAND_CMAP_RANGE
    colors = [cmap(lo_c + (hi_c - lo_c) * j / max(len(bands) - 1, 1)) for j in range(len(bands))]
    fig, ax = plt.subplots(figsize=(9, 5))
    for j, (lo, hi) in enumerate(bands):
        if np.all(np.isnan(power[:, j])):
            continue
        label = f'{lo:.2g}-{min(hi, nyquist):.3g} Hz'  # bands are cut at the Nyquist frequency
        ax.plot(x, power[:, j], marker='o', markersize=3, linewidth=1.2,
                color=colors[j], label=label)
    ax.set_yscale('log')
    ax.set_xlabel('Hours APF' if by_hours else 'Recording')
    ax.set_ylabel('Band power (dF/F$^2$)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, which='major', color='0.9', linewidth=0.6)
    ax.legend(title='Frequency band', fontsize=8, title_fontsize=8, frameon=False,
              loc='center left', bbox_to_anchor=(1.01, 0.5))
    fig.suptitle(f'{suptitle}\npower of dF/F per frequency band', y=1.0, fontsize=11)

    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, BAND_POWER_NAME.format(f'_{suffix}' if suffix else ''))
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return save_path


def save_relative_heatmap(traces: list[dict],
                          folder: str | os.PathLike,
                          suptitle: str = '',
                          suffix: str = '') -> str:
    """
    Save the normalized power spectra of many traces as a heatmap, as `folder/HEATMAP_NAME`:
    one column per trace at its hours APF (each column spans half the gap to its neighbours,
    so gaps between recordings show), frequency on a log y axis in narrow bands (the
    `BINS_PER_DECADE` log bins of the smoothed spectra), and as color, each frequency relative
    to its mean over the traces.

    Each spectrum is normalized by its total power (see `compute_power_spectrum()`), so the
    amount of activity doesn't matter, only how it is spread over frequency. The color is
    log10 of the normalized power divided by its geometric mean over all the traces at that
    frequency: red where a recording puts more of its power at that frequency than the average
    recording, blue where it puts less. The reference is the mean of the traces given, so it
    depends on which recordings are included. A trace with a lower Nyquist frequency than the
    others is left blank above it.

    Parameters
    ----------
    traces : list of dict
        The traces (see module docstring), in time order, each with 'hours' (or they are
        numbered in order).
    folder : str or os.PathLike
        Output folder (created if needed).
    suptitle : str
        Figure title, e.g. the experiment name.
    suffix : str
        Added to the file name (after an underscore).

    Returns
    -------
    str
        Path of the saved figure.
    """
    spectra = [compute_power_spectrum(tr['activity'], tr['fs'], smooth='log', normalize=True)
               for tr in traces]
    hours = [tr.get('hours') for tr in traces]
    by_hours = all(h is not None for h in hours)
    x = np.array(hours if by_hours else range(len(traces)), dtype=float)

    # common frequency axis: that of the trace reaching highest (all the same within a setup;
    # 5% tolerance, as frame rates estimated from timestamps differ slightly between recordings)
    f_ref = max((f for f, _ in spectra), key=lambda f: f[-1])
    log_power = np.full((len(f_ref), len(traces)), np.nan)
    for i, (f, psd) in enumerate(spectra):
        inside = (f_ref >= f[0] / 1.05) & (f_ref <= f[-1] * 1.05)
        log_power[inside, i] = np.interp(np.log(f_ref[inside]), np.log(f), np.log10(psd))
    relative = log_power - np.nanmean(log_power, axis=1, keepdims=True)

    # column edges halfway between recordings (half a median gap at both ends), row edges
    # halfway (in log) between frequencies
    gap = np.median(np.diff(x)) if len(x) > 1 else 1.0
    x_edges = np.concatenate([[x[0] - gap / 2], (x[:-1] + x[1:]) / 2, [x[-1] + gap / 2]])
    f_edges = np.concatenate([[f_ref[0] ** 1.5 / f_ref[1] ** 0.5],
                              np.sqrt(f_ref[:-1] * f_ref[1:]),
                              [f_ref[-1] ** 1.5 / f_ref[-2] ** 0.5]])
    lim = np.nanpercentile(np.abs(relative), 99)

    fig, ax = plt.subplots(figsize=(10, 5))
    mesh = ax.pcolormesh(x_edges, f_edges, relative, cmap=HEATMAP_CMAP, vmin=-lim, vmax=lim,
                         shading='flat', rasterized=True)
    ax.set_yscale('log')
    ax.set_ylim(f_edges[0], f_edges[-1])
    ax.set_xlabel('Hours APF' if by_hours else 'Recording')
    ax.set_ylabel('Frequency (Hz)')
    fig.colorbar(mesh, ax=ax, pad=0.02,
                 label='log$_{10}$ (normalized power / its mean over the recordings)')
    fig.suptitle(f'{suptitle}\nnormalized power spectrum of dF/F relative to the mean over the '
                 f'recordings', y=1.0, fontsize=11)

    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, HEATMAP_NAME.format(f'_{suffix}' if suffix else ''))
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return save_path


def save_frequency_analysis(exp_folder: str | os.PathLike,
                            traces: list[dict],
                            pick_eclosion: bool = False) -> str | None:
    """
    Save the frequency analysis of a long recording, using only the recordings before
    eclosion, in `<exp_folder>/FREQUENCY_FOLDER`: the overlaid power spectra with a log and a
    linear frequency axis, and normalized with a log one (see `save_power_spectrum_summary()`),
    the band power against hours APF (see `save_band_power()`), and the heatmap of the normalized spectra relative to their mean (see
    `save_relative_heatmap()`).

    The eclosion recording is read from the experiment's eclosion file (see
    `common.eclosion`); if there is none (or `pick_eclosion`), the user is asked to pick it.
    The eclosion recording and every later one are left out.

    Parameters
    ----------
    exp_folder : str or os.PathLike
        The experiment (main) folder of the long recording.
    traces : list of dict
        Its recordings' traces (see module docstring, each with a 'name'), in time order.
    pick_eclosion : bool
        Ask for the eclosion recording even if it was already picked.

    Returns
    -------
    str or None
        The output folder, or None if nothing was saved (cancelled, or no recording before
        eclosion).
    """
    from common.eclosion import get_eclosion_index

    if len(traces) == 0:
        print(f'No preprocessed recordings found in {exp_folder}.')
        return None
    index = get_eclosion_index(exp_folder, traces, pick=pick_eclosion)
    if index is None:
        print('No eclosion recording picked, frequency analysis cancelled.')
        return None
    if index == 0:
        print('The fly eclosed in the first recording: nothing to analyse.')
        return None
    before = traces[:index]
    if index < len(traces):
        label = f'before eclosion ({traces[index]["title"]})'
    else:
        label = 'no eclosion'
    print(f'Frequency analysis of {len(before)} recordings, {label}...')
    suptitle = f'{os.path.basename(os.path.normpath(exp_folder))} - {label}'
    folder = os.path.join(exp_folder, FREQUENCY_FOLDER)
    for scale in ('log', 'linear'):
        save_power_spectrum_summary(before, folder, scale, suptitle=suptitle)
    save_power_spectrum_summary(before, folder, 'log', suptitle=suptitle, normalize=True)
    save_band_power(before, folder, suptitle=suptitle)
    save_relative_heatmap(before, folder, suptitle=suptitle)
    print(f'Saved the frequency analysis in {folder}')
    return folder
