#!/usr/bin/env python3
"""
Shared helpers for the FlyTrack .mkv preprocessing (`preprocess_flytrack.py`).

Holds the constants, the single-video decode/projection routine, the
per-recording processing (`_process_mkv()`: decode a video -> save
``projection_images.h5`` (median, min/max projections, cumulative mean/mean_sq,
whole-video ``std`` and per-frame mean fluorescence ``act``), ``metadata.txt``,
the activity plot, the ``median.png``, ``max_proj.png``, ``min_proj.png`` and
``std.png`` preview images, and a half-resolution/compressed
``summary_video.mp4`` preview), and the interactive selection parser. Also holds
`tif2mkv()`, which converts a folder of raw FlyTrack tifs (numbered RGBA tifs
plus per-tif metadata csvs) into the ``_raw_tiff.mkv`` and ``_tif_metadata.h5``
pair used here, for recordings that were never exported.

Preview contrast: every preview PNG is auto-contrasted on its own between its
0.2th and 99.8th percentiles. The summary video uses one fixed contrast for the
whole video, the 0.2th-99.8th percentiles of ~`VIDEO_CONTRAST_SAMPLES` evenly
spaced frames read by seeking before the main decode (`_video_contrast()`), and
plays back at the acquisition frame rate read from ``config_concatenated.yaml``
(`_get_framerate()`).

Optional within-recording rigid-translation drift correction
(`_process_mkv(..., motion_correct=True)`, run by ``preprocess_flytrack.py -m``):
drift is estimated by phase correlation on spatially high-pass-filtered frames
so the changing GCaMP brightness does not drive the alignment. The video is
decoded once to build the uncorrected (drift-blurred) projections, the median is
sharpened into a template, and the video is decoded a second time to align every
frame to it (roughly doubling the decode time). The corrected images are saved
to ``projection_images.h5`` under the usual names, the uncorrected ones with a
``_raw`` suffix (``median_raw``, ``max_proj_raw``, ...), and the per-frame drift
as ``shifts`` (also plotted as ``drift.png``). The corrected preview video is
``summary_video.mp4`` and the uncorrected one ``summary_video_raw.mp4`` (an
uncorrected ``summary_video.mp4`` from a first run is renamed to it); the PNGs
show the corrected images, with
``_raw`` versions of them written alongside.
"""

import os
import re
import csv
import sys
import yaml
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import av
import cv2
import h5py
import tqdm
from scipy import ndimage
from skimage.registration import phase_cross_correlation

import matplotlib
matplotlib.use('Agg')  # force a non-interactive backend
import matplotlib.pyplot as plt

# put this file's folder and its parent (which holds utils.py) on sys.path so
# `import utils` works no matter how this module is launched (as a script, via
# `-m`, or imported as FlyTrack.<module> from a notebook) or what the working
# directory is
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import utils

MAIN_PATH = '/mnt/labserver/data/MA/Development_project'

# saving names
PROJ_H5_NAME = 'projection_images.h5'        # per-recording, next to each .mkv
METADATA_NAME = 'metadata.txt'               # per-recording, next to each .mkv
ACT_SAVE_NAME = 'activity.png'               # per-recording mean fluorescence plot
DRIFT_SAVE_NAME = 'drift.png'                # per-recording estimated drift plot
MEDIAN_SAVE_NAME = 'median.png'              # per-recording median image
STD_SAVE_NAME = 'std.png'                    # per-recording std image
MAX_PROJ_SAVE_NAME = 'max_proj.png'          # per-recording max projection
MIN_PROJ_SAVE_NAME = 'min_proj.png'          # per-recording min projection
SUMMARY_NAME = 'summary_video.mp4'           # per-recording downsampled preview
SUMMARY_RAW_NAME = 'summary_video_raw.mp4'  # per-recording, uncorrected (after motion correction)
MKV_PATHS_NAME = 'mkv_paths.json'            # in the analysis folder
GROUPED_H5_NAME = 'projection_images_grouped.h5'  # in the analysis folder
CONFIG_NAME = 'config_concatenated.yaml'     # acquisition config, next to each .mkv

# datasets required in projection_images.h5 for a recording to count as done.
# `std` is included so recordings processed before it was added (which cannot be
# backfilled) are reprocessed rather than skipped.
PROJ_DATASETS = ('median', 'min_proj', 'max_proj', 'mean', 'mean_sq', 'std',
                 'frame_count', 'act')

# datasets that, after motion correction, hold the corrected images and are also
# saved uncorrected under the same name plus RAW_SUFFIX (e.g. `median_raw`)
CORRECTED_DATASETS = ('median', 'min_proj', 'max_proj', 'mean', 'mean_sq',
                      'std', 'act')
RAW_SUFFIX = '_raw'

# number of evenly-spaced frames kept to estimate the median (memory bounded;
# the estimate is approximate but close to the true median)
MEDIAN_SAMPLES = 3000

# preview contrast: low/high percentiles mapped to black/white in every preview
# PNG (per image) and in the summary video (fixed for the whole video)
CONTRAST_PCT = (0.2, 99.8)

# motion-correction (rigid-translation drift) parameters. Drift is estimated by
# phase correlation on spatially high-pass-filtered images so that structure
# (not the changing GCaMP brightness) drives the alignment. HP_SIGMA is the
# Gaussian sigma (px) of the low-pass that is subtracted to build the high-pass
# image; UPSAMPLE sets the subpixel precision (1/UPSAMPLE px); REFINE_ITERS is
# how many times the template is re-sharpened from the aligned median subsample
# before the full correction pass, using at most REFINE_SAMPLES of the median
# subsample frames (evenly spaced; bounds the memory and time of refinement).
#
# DOWNSAMPLE speeds up the per-frame registration: the shift is estimated on a
# DOWNSAMPLE-times smaller high-pass image (both the reference and each frame are
# downsampled first) and then scaled back to full resolution. This cuts the
# high-pass filter and FFT cost by roughly DOWNSAMPLE**2 at a sub-pixel precision
# of DOWNSAMPLE/UPSAMPLE px. The frames themselves are still shifted at full
# resolution, so only the drift estimate is affected. Set to 1 to disable.
HP_SIGMA = 8
UPSAMPLE = 10
REFINE_ITERS = 2
REFINE_SAMPLES = 1500
DOWNSAMPLE = 2

# summary preview video: every frame is streamed straight to a compressed mp4
# (so nothing is held in memory), spatially reduced by averaging 2x2 pixel blocks
# (half resolution in each direction), and mapped to 8-bit with one fixed
# contrast (CONTRAST_PCT percentiles of VIDEO_CONTRAST_SAMPLES evenly spaced
# container frames, i.e. 4x as many real frames, read by seeking before the
# decode). It plays back at the acquisition frame rate from CONFIG_NAME.
VIDEO_CONTRAST_SAMPLES = 100

# raw tif -> .mkv conversion (`tif2mkv()`): FlyTrack saves each recording as
# numbered RGBA tifs (4 real frames per tif) plus a `<n>_metadata.csv` per tif.
# These are packed into a lossless ffv1 .mkv at 1/TIF_DURATION container fps,
# matching flymuscle-control's export/export_exp_as_mkv.py.
TIF_DURATION = 0.04
FIXED_TIF_DIR = 'fixed_tif'                  # temporary, inside the tif folder
TIF_LIST_NAME = 'list_tif.txt'               # temporary ffmpeg concat list
FFMPEG_LOG_NAME = 'ffmpeg_tif_export.log'    # kept only if ffmpeg fails


def _parse_selection(selection: str, n: int) -> list[int]:
    """
    Parse a selection string like ``1,3,5-7`` into sorted 0-based indices.

    Numbers and ``a-b`` ranges may be separated by commas and/or spaces. Values
    are 1-based on input; anything out of ``1..n`` or unparseable is reported and
    ignored.

    Parameters
    ----------
    selection : str
        The user's selection string.
    n : int
        Number of listed items (valid 1-based indices are ``1..n``).

    Returns
    -------
    list of int
        Sorted, de-duplicated 0-based indices.
    """

    indices: set[int] = set()
    for tok in selection.replace(',', ' ').split():
        if '-' in tok:
            parts = tok.split('-')
            if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
                print(f"  Ignoring invalid range '{tok}'.")
                continue
            a, b = int(parts[0]), int(parts[1])
            for k in range(min(a, b), max(a, b) + 1):
                if 1 <= k <= n:
                    indices.add(k - 1)
                else:
                    print(f'  Ignoring out-of-range index {k}.')
        elif tok.isdigit():
            k = int(tok)
            if 1 <= k <= n:
                indices.add(k - 1)
            else:
                print(f'  Ignoring out-of-range index {k}.')
        else:
            print(f"  Ignoring invalid entry '{tok}'.")

    return sorted(indices)


def _outputs_exist(dir: str) -> bool:
    """
    Check whether a recording has already been processed.

    Returns True only if a `projection_images.h5` containing all the datasets in
    `PROJ_DATASETS` is present in `dir` (whether or not it was motion-corrected).

    Parameters
    ----------
    dir : str
        Folder containing the .mkv recording.

    Returns
    -------
    bool
        Whether the recording can be skipped.
    """

    h5_path = os.path.join(dir, PROJ_H5_NAME)
    if not os.path.exists(h5_path):
        return False
    try:
        with h5py.File(h5_path, 'r') as f:
            return all(key in f for key in PROJ_DATASETS)
    except OSError:
        return False


def _is_corrected(dir: str) -> bool:
    """
    Check whether a recording's `projection_images.h5` is motion-corrected.

    Parameters
    ----------
    dir : str
        Folder containing the .mkv recording.

    Returns
    -------
    bool
        True if `projection_images.h5` exists and holds a `shifts` dataset.
    """

    h5_path = os.path.join(dir, PROJ_H5_NAME)
    if not os.path.exists(h5_path):
        return False
    try:
        with h5py.File(h5_path, 'r') as f:
            return 'shifts' in f
    except OSError:
        return False


def _get_framerate(mkv_file: str,
                   fallback: float) -> float:
    """
    Get the acquisition frame rate (real frames per second) of a recording.

    Read from ``arduino: framerate`` in the ``config_concatenated.yaml`` saved
    next to the .mkv file (a concatenation of several yaml documents, of which
    the main config is the one holding the ``arduino`` section). If the config
    is missing or has no frame rate, `fallback` is used instead.

    Parameters
    ----------
    mkv_file : str
        Path to the .mkv video file.
    fallback : float
        Frame rate to use if it can't be read from the config, e.g. the one
        estimated from the timestamps by `utils.load_timestamps()`.

    Returns
    -------
    float
        Frame rate in Hz.
    """

    config_path = os.path.join(os.path.dirname(mkv_file), CONFIG_NAME)
    if os.path.exists(config_path):
        with open(config_path) as f:
            for doc in yaml.safe_load_all(f):
                if isinstance(doc, dict) and 'framerate' in (doc.get('arduino') or {}):
                    return float(doc['arduino']['framerate'])
    print(f'  No frame rate found in {CONFIG_NAME}, using {fallback:.1f} Hz '
          f'from the timestamps.')
    return fallback


def _highpass(img: np.ndarray, sigma: float = HP_SIGMA) -> np.ndarray:
    """
    Return a spatially high-pass-filtered copy of `img` as float64.

    The high-pass image is `img` minus a Gaussian low-pass of it, which
    suppresses the smooth, changing GCaMP brightness and keeps the fixed
    structural edges. Registering on this image makes drift estimation robust to
    activity changes.

    Parameters
    ----------
    img : np.ndarray
        Single-frame image.
    sigma : float, optional
        Gaussian sigma (px) of the low-pass that is subtracted.

    Returns
    -------
    np.ndarray
        High-pass image as float64.
    """

    img_f = img.astype(np.float64)
    return img_f - ndimage.gaussian_filter(img_f, sigma)


def _downsample(img: np.ndarray, factor: int) -> np.ndarray:
    """
    Return `img` bilinearly downsampled by `factor` as float64.

    A factor of 1 returns a float64 copy unchanged. Used to shrink images before
    registration so the high-pass filter and FFT are cheaper.

    Parameters
    ----------
    img : np.ndarray
        Image to downsample.
    factor : int
        Integer downsampling factor.

    Returns
    -------
    np.ndarray
        Downsampled image as float64.
    """

    img_f = img.astype(np.float64)
    if factor == 1:
        return img_f
    return ndimage.zoom(img_f, 1.0 / factor, order=1)


def _reference_hp(template: np.ndarray,
                  hp_sigma: float = HP_SIGMA,
                  downsample: int = DOWNSAMPLE) -> np.ndarray:
    """
    Build the downsampled high-pass reference a frame is registered against.

    The template is downsampled by `downsample` and high-pass filtered at the
    correspondingly reduced sigma, so it matches the images produced for each
    frame in `_estimate_shift()`.

    Parameters
    ----------
    template : np.ndarray
        Full-resolution reference image (e.g. the sharpened median template).
    hp_sigma : float, optional
        Full-resolution high-pass sigma; scaled by `1 / downsample`.
    downsample : int, optional
        Integer downsampling factor for registration.

    Returns
    -------
    np.ndarray
        Downsampled high-pass reference image as float64.
    """

    return _highpass(_downsample(template, downsample), hp_sigma / downsample)


def _estimate_shift(ref_hp: np.ndarray,
                    img: np.ndarray,
                    hp_sigma: float = HP_SIGMA,
                    upsample: int = UPSAMPLE,
                    downsample: int = DOWNSAMPLE) -> np.ndarray:
    """
    Estimate the (dy, dx) shift that aligns `img` to a reference.

    `img` is downsampled by `downsample` and high-pass filtered, then phase
    correlation against `ref_hp` (the matching downsampled high-pass reference
    from `_reference_hp()`) gives the shift in downsampled pixels, which is scaled
    back to full-resolution pixels before being returned.

    Parameters
    ----------
    ref_hp : np.ndarray
        Downsampled high-pass reference from `_reference_hp()`.
    img : np.ndarray
        Full-resolution frame to align.
    hp_sigma : float, optional
        Full-resolution high-pass sigma; scaled by `1 / downsample`.
    upsample : int, optional
        Subpixel upsampling factor for `phase_cross_correlation()`.
    downsample : int, optional
        Integer downsampling factor for registration.

    Returns
    -------
    np.ndarray
        Full-resolution (dy, dx) shift.
    """

    img_hp = _highpass(_downsample(img, downsample), hp_sigma / downsample)
    # normalization=None is plain (sub-pixel) cross-correlation. The default
    # 'phase' normalization whitens the spectrum and, on noisy fluorescence
    # high-pass images, collapses the peak to (0, 0) -> no drift is ever found.
    shift, _, _ = phase_cross_correlation(
        ref_hp, img_hp, upsample_factor=upsample, normalization=None)
    return np.asarray(shift) * downsample


def _refine_template(samples: np.ndarray,
                     template: np.ndarray,
                     hp_sigma: float = HP_SIGMA,
                     upsample: int = UPSAMPLE,
                     iters: int = REFINE_ITERS,
                     downsample: int = DOWNSAMPLE) -> np.ndarray:
    """
    Sharpen a drift-blurred template using the median frame subsample.

    Each round, every subsampled frame is aligned to the current template (by
    high-pass phase correlation) and a new template is taken as the median of
    the aligned frames. This runs entirely on the in-memory subsample (no extra
    video decode) and gives a crisper template for the full correction pass.
    Only up to `REFINE_SAMPLES` evenly spaced frames of `samples` are used.

    Parameters
    ----------
    samples : np.ndarray
        Stack of subsampled frames, shape (n_samples, H, W).
    template : np.ndarray
        Initial (drift-blurred) template, e.g. the uncorrected median.
    hp_sigma : float, optional
        High-pass sigma passed to `_highpass()`.
    upsample : int, optional
        Subpixel upsampling factor for `phase_cross_correlation()`.
    iters : int, optional
        Number of refinement rounds.
    downsample : int, optional
        Integer downsampling factor for registration (see `_estimate_shift()`).

    Returns
    -------
    np.ndarray
        Refined template as float64, shape (H, W).
    """

    # evenly spaced subset of at most REFINE_SAMPLES frames
    if samples.shape[0] > REFINE_SAMPLES:
        idx = np.linspace(0, samples.shape[0] - 1, REFINE_SAMPLES).round().astype(int)
        samples = samples[idx]

    template = template.astype(np.float64)
    k = samples.shape[0] // 2
    for it in range(iters):
        ref_hp = _reference_hp(template, hp_sigma, downsample)
        # aligned frames kept as uint16 (rounded) to halve the memory
        aligned = np.empty(samples.shape, dtype=np.uint16)
        for i, s in enumerate(tqdm.tqdm(samples,
                                        desc=f'Refining template {it+1}/{iters}',
                                        unit='frame')):
            shift = _estimate_shift(ref_hp, s, hp_sigma, upsample, downsample)
            shifted = ndimage.shift(s.astype(np.float64), shift,
                                    order=1, mode='nearest')
            aligned[i] = np.clip(np.round(shifted), 0, 65535)
        template = np.partition(aligned, k, axis=0)[k].astype(np.float64)
    return template


def _bin2x2(img: np.ndarray) -> np.ndarray:
    """
    Average 2x2 pixel blocks, halving the resolution in each direction.

    Any odd last row/column is dropped so the dimensions are even.

    Parameters
    ----------
    img : np.ndarray
        2-D image.

    Returns
    -------
    np.ndarray
        The 2x2-averaged image, shape (H//2, W//2).
    """

    height, width = img.shape
    h2, w2 = height - height % 2, width - width % 2
    return img[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2).mean(axis=(1, 3))


def _video_contrast(mkv_file: str,
                    n_samples: int = VIDEO_CONTRAST_SAMPLES) -> tuple[float, float]:
    """
    Get the fixed contrast range of a recording's summary video.

    Seeks to `n_samples` evenly spaced container frames (cheap, as ffv1 is
    intra-only so every frame is a keyframe), takes all 4 real frames of each,
    reduces them to half resolution as in the summary video (`_bin2x2()`), and
    returns the `CONTRAST_PCT` percentiles of their pooled pixels (every other
    pixel in each direction, to bound memory). If seeking fails, the first
    `n_samples` container frames are used instead.

    Parameters
    ----------
    mkv_file : str
        Absolute path to the source .mkv video file.
    n_samples : int, optional
        Number of container frames to sample.

    Returns
    -------
    tuple[float, float]
        (low, high) intensities mapped to black and white in the summary video.
    """

    frames = []
    container = av.open(mkv_file)
    stream = container.streams.video[0]
    try:
        # container.duration is in av.time_base units (microseconds)
        for t in np.linspace(0, container.duration * 0.99, n_samples):
            container.seek(int(t))
            frames.append(next(container.decode(stream)).to_ndarray(format='rgba64le'))
    except Exception as e:  # e.g. no duration or a file that can't be seeked
        print(f'  Could not seek in {os.path.basename(mkv_file)} ({e}); using '
              f'the first {n_samples} frames for the summary video contrast.')
        container.close()
        container = av.open(mkv_file)
        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format='rgba64le'))
            if len(frames) >= n_samples:
                break
    container.close()

    pixels = np.concatenate([_bin2x2(f[..., j])[::2, ::2].ravel().astype(np.float32)
                             for f in frames for j in range(f.shape[2])])
    lo, hi = np.percentile(pixels, CONTRAST_PCT)
    if hi <= lo:
        hi = lo + 1
    return float(lo), float(hi)


def _decode_projections(mkv_file: str,
                        n_frames: int,
                        template: np.ndarray | None = None,
                        hp_sigma: float = HP_SIGMA,
                        upsample: int = UPSAMPLE,
                        downsample: int = DOWNSAMPLE,
                        summary_path: str | None = None,
                        summary_fps: float = 30.0,
                        video_contrast: tuple[float, float] = (0, 600)) -> dict:
    """
    Decode one .mkv once and accumulate its projection images.

    Decodes the video (4 real frames packed per container frame, 16-bit) and
    computes the min and max projections, the cumulative mean and mean of
    squares and the whole-video std image built from them, the per-frame mean
    fluorescence, and a subsample of `MEDIAN_SAMPLES` evenly spaced frames used
    to estimate the median.

    If `template` is given, each real frame is drift-corrected before being
    accumulated: its rigid-translation shift is estimated by `_estimate_shift()`
    (downsampled high-pass phase correlation against the template) and applied
    with linear interpolation (edges filled by replication). This removes rigid
    drift so the resulting projections are sharp; the per-frame shifts are
    returned too.

    Parameters
    ----------
    mkv_file : str
        Absolute path to the source .mkv video file.
    n_frames : int
        Expected number of real frames (the timestamp count; .mkv does not
        expose a frame count). Sets the median subsample positions and the
        progress-bar total.
    template : np.ndarray or None, optional
        Full-resolution reference image to align every frame to. If None, no
        correction is done and the projections are built from the raw frames.
    hp_sigma : float, optional
        High-pass sigma passed to `_highpass()`.
    upsample : int, optional
        Subpixel upsampling factor for `phase_cross_correlation()`.
    downsample : int, optional
        Integer downsampling factor for registration (see `_estimate_shift()`).
    summary_path : str or None, optional
        If given, a half-resolution (2x2-averaged), compressed preview mp4 of the
        (corrected, if `template` is set) frames is streamed to this path, one
        entry per frame.
    summary_fps : float, optional
        Playback frame rate of the preview mp4 (use the real recording fps).
    video_contrast : tuple of float, optional
        (low, high) intensities mapped to black and white in the preview mp4,
        see `_video_contrast()`.

    Returns
    -------
    dict
        Keys: 'median', 'min_proj', 'max_proj' (uint16), 'mean', 'mean_sq',
        'std' (float64), 'act' (float64, per-frame mean), 'samples' (uint16
        subsample stack, for template refinement), 'shifts' ((n, 2) float64
        (dy, dx) or None if uncorrected), 'n', 'width', 'height'.
    """

    correct = template is not None

    # downsampled high-pass reference every frame is registered against
    ref_hp = _reference_hp(template, hp_sigma, downsample) if correct else None

    # get the frame shape from the first decoded frame
    container = av.open(mkv_file)
    frame_0 = next(container.decode(video=0)).to_ndarray(format='rgba64le')[..., 0]
    container.close()
    height, width = frame_0.shape

    # projections accumulate in float64 because drift-corrected frames are
    # interpolated (non-integer); they are cast back to uint16 at the end
    max_proj = np.zeros((height, width), dtype=np.float64)
    min_proj = np.full((height, width), fill_value=65535.0, dtype=np.float64)
    sum_ = np.zeros((height, width), dtype=np.float64)
    sum_sq = np.zeros((height, width), dtype=np.float64)

    # mean fluorescence of each frame over time (activity) and per-frame drift
    act: list[float] = []
    shifts: list[np.ndarray] = []

    # evenly spaced subsample of frames for the (approximate) median, at
    # positions set from the expected frame count (preallocated buffer)
    n_samples = max(1, min(MEDIAN_SAMPLES, n_frames))
    sample_idx = np.unique(np.linspace(0, n_frames - 1, n_samples).round().astype(int))
    sample_pos = {int(i): k for k, i in enumerate(sample_idx)}
    samples = np.zeros((len(sample_idx), height, width), dtype=np.uint16)
    n_sampled = 0

    # optional preview video: every frame is streamed to a compressed mp4 (half
    # resolution via 2x2 averaging), so nothing is held in memory
    summary = summary_path is not None
    if summary:
        sh, sw = (height - height % 2) // 2, (width - width % 2) // 2
        summary_writer = cv2.VideoWriter(
            summary_path, cv2.VideoWriter_fourcc(*'mp4v'),
            summary_fps, (sw, sh))
        lo, hi = video_contrast

    n = 0  # real-frame counter
    container = av.open(mkv_file)
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'

    # the bar counts real frames and is updated once per decoded packet (which
    # holds 4 of them)
    desc = 'Correcting + decoding' if correct else 'Decoding'
    pbar = tqdm.tqdm(total=n_frames, desc=desc, unit='frame')
    for frame in container.decode(stream):
        img = frame.to_ndarray(format='rgba64le')  # (H, W, 4): 4 real frames
        for j in range(img.shape[2]):
            img_i = img[..., j]

            if correct:
                shift = _estimate_shift(ref_hp, img_i, hp_sigma,
                                        upsample, downsample)
                img_f = ndimage.shift(img_i.astype(np.float64), shift,
                                      order=1, mode='nearest')
                shifts.append(shift)
            else:
                img_f = img_i.astype(np.float64)

            np.maximum(max_proj, img_f, out=max_proj)
            np.minimum(min_proj, img_f, out=min_proj)
            sum_ += img_f
            sum_sq += img_f * img_f
            act.append(img_f.mean())

            if n in sample_pos:
                samples[sample_pos[n]] = np.clip(np.round(img_f), 0, 65535)
                n_sampled += 1

            if summary:
                small = (_bin2x2(img_f) - lo) / (hi - lo)
                small = np.clip(small, 0, 1) * 255
                summary_writer.write(
                    cv2.cvtColor(small.astype(np.uint8), cv2.COLOR_GRAY2BGR))

            n += 1

        pbar.update(img.shape[2])

    pbar.close()
    container.close()
    print(f'Decoded {n} real frames.')

    # cumulative statistics and the whole-video std: std = sqrt(E[x^2] - E[x]^2)
    mean = sum_ / n
    mean_sq = sum_sq / n
    std = np.sqrt(np.clip(mean_sq - mean ** 2, 0, None))

    # approximate median from the evenly spaced subsample (only the positions
    # that were reached, in case the video has fewer frames than expected)
    samples = samples[:n_sampled]
    k = samples.shape[0] // 2
    median_frame = np.partition(samples, k, axis=0)[k]
    print(f'Estimated median from {samples.shape[0]} subsampled frames.')

    # finalize the preview video
    if summary:
        summary_writer.release()
        print(f'Saved summary video ({sw}x{sh}) to {summary_path}')

    return {
        'median': median_frame,
        'min_proj': np.clip(np.round(min_proj), 0, 65535).astype(np.uint16),
        'max_proj': np.clip(np.round(max_proj), 0, 65535).astype(np.uint16),
        'mean': mean,
        'mean_sq': mean_sq,
        'std': std,
        'act': np.array(act),
        'samples': samples,
        'shifts': np.array(shifts) if correct else None,
        'n': n,
        'width': int(width),
        'height': int(height),
    }


def _autocontrast_8bit(img: np.ndarray,
                       pct: tuple[float, float] = CONTRAST_PCT) -> np.ndarray:
    """
    Map an image to 8-bit, auto-contrasted between two of its percentiles.

    The image is linearly rescaled so its `pct[0]` percentile maps to 0 and its
    `pct[1]` percentile maps to 255 (values outside are clipped), which keeps a
    few very dark or bright outlier pixels from washing out the visible range.
    If the high percentile is not above the low one, the true min and max are
    used instead.

    Parameters
    ----------
    img : np.ndarray
        Image to rescale.
    pct : tuple of float, optional
        (low, high) percentiles mapped to 0 and 255.

    Returns
    -------
    np.ndarray
        The auto-contrasted image as uint8.
    """

    img_f = img.astype(np.float64)
    lo, hi = np.percentile(img_f, pct)
    if hi <= lo:
        lo, hi = img_f.min(), img_f.max()
    scaled = np.clip((img_f - lo) / (hi - lo + 1e-9), 0, 1) * 255
    return scaled.astype(np.uint8)


def _write_pngs(path: str) -> None:
    """
    Save auto-contrasted 8-bit preview PNGs next to a projection-images h5.

    Reads the `median`, `max_proj`, `min_proj`, and `std` datasets from a
    projection-images h5 (those that are present; the grouped h5 has no
    `min_proj`) and writes each as an auto-contrasted 8-bit PNG in the same
    folder (see `_autocontrast_8bit()`): `median.png`, `max_proj.png`,
    `min_proj.png`, and `std.png`. If the h5 is motion-corrected and also holds
    the uncorrected images (`median_raw`, ...), those are written too, as
    `median_raw.png`, ....

    Parameters
    ----------
    path : str
        Either a `projection_images.h5` / `projection_images_grouped.h5` file, or
        the folder containing one of them (the file is found automatically).

    Raises
    ------
    FileNotFoundError
        If the folder does not exist, or contains neither
        `projection_images.h5` nor `projection_images_grouped.h5`.
    """

    # accept either a direct h5 file path or the folder that holds one of them
    if os.path.isdir(path):
        dir = path
        h5_path = next((os.path.join(dir, name)
                        for name in (PROJ_H5_NAME, GROUPED_H5_NAME)
                        if os.path.exists(os.path.join(dir, name))), None)
        if h5_path is None:
            raise FileNotFoundError(
                f'Folder does not contain {PROJ_H5_NAME} or {GROUPED_H5_NAME}: '
                f'{dir}')
    else:
        h5_path = path
        dir = os.path.dirname(h5_path)
        if not os.path.isdir(dir):
            raise FileNotFoundError(f'Folder does not exist: {dir}')
        if not os.path.exists(h5_path):
            raise FileNotFoundError(
                f'Folder does not contain {os.path.basename(h5_path)}: {dir}')

    png_names = {'median': MEDIAN_SAVE_NAME, 'max_proj': MAX_PROJ_SAVE_NAME,
                 'min_proj': MIN_PROJ_SAVE_NAME, 'std': STD_SAVE_NAME}
    with h5py.File(h5_path, 'r') as f:
        for key, name in png_names.items():
            for suffix in ('', RAW_SUFFIX):
                if key + suffix in f:
                    img = f[key + suffix][:]  # type: ignore[index]
                    png_path = os.path.join(dir, name.replace('.png', suffix + '.png'))
                    cv2.imwrite(png_path, _autocontrast_8bit(img))


def _plot_activity(mkv_file: str,
                   act: np.ndarray,
                   act_path: str) -> None:
    """
    Plot the mean fluorescence of a recording over time and save it as a png.

    Parameters
    ----------
    mkv_file : str
        Path to the .mkv video file (its timestamps give the time axis).
    act : np.ndarray
        Mean fluorescence of every real frame.
    act_path : str
        Save path of the plot; its folder is created if needed.
    """

    timestamps, _ = utils.load_timestamps(mkv_file)
    if len(timestamps) != len(act):
        print(f'Warning: number of timestamps ({len(timestamps)}) does not match '
              f'number of frames ({len(act)}), LOST FRAMES.')
    n = min(len(timestamps), len(act))
    plt.figure(figsize=(10, 3))
    plt.plot(timestamps[:n] / 60, act[:n], linewidth=0.5)  # in minutes
    plt.xlim(0, timestamps[n - 1] / 60)
    plt.xlabel('Time (min)')
    plt.ylabel('Mean fluorescence')
    plt.title(os.path.basename(mkv_file))
    os.makedirs(os.path.dirname(act_path), exist_ok=True)
    plt.savefig(act_path, dpi=300)
    plt.close()


def _process_mkv(mkv_file: str,
                 act_path: str | None = None,
                 motion_correct: bool = False) -> tuple[list[int], int, float]:
    """
    Decode one .mkv and write its projection images and metadata.

    Builds the projection images with `_decode_projections()` and saves them to
    `projection_images.h5` and `metadata.txt` in the same folder as the video,
    plus the mean-fluorescence plot at `act_path`, the preview PNGs (see
    `_write_pngs()`), and the half-resolution `summary_video.mp4` (fixed
    contrast from `_video_contrast()`, at the frame rate from
    `_get_framerate()`).

    If `motion_correct` is True, rigid-translation drift is removed too: the
    video is decoded once to build the uncorrected projections (also writing
    the uncorrected preview video as `summary_video_raw.mp4`, unless it or an
    uncorrected `summary_video.mp4` from a first run exists; the latter is
    renamed to `summary_video_raw.mp4`), their median is sharpened into
    a template with `_refine_template()`, and the video is decoded a second time
    to build drift-corrected projections. This roughly doubles the decode time.
    The corrected images are saved under the usual dataset names, the
    uncorrected ones with a `_raw` suffix, and the per-frame drift as `shifts`
    (also plotted as `drift.png`); the corrected preview video is
    `summary_video.mp4`, and the activity plot is made from the
    corrected frames.

    Parameters
    ----------
    mkv_file : str
        Absolute path to the source .mkv video file.
    act_path : str or None, optional
        Save path of the activity plot. If None, `activity.png` next to the
        video.
    motion_correct : bool, optional
        Whether to correct rigid-translation drift.

    Returns
    -------
    tuple[list[int], int, float]
        (size as [width, height], number of real frames, fps).
    """

    dir = os.path.dirname(mkv_file)
    if act_path is None:
        act_path = os.path.join(dir, ACT_SAVE_NAME)

    # timestamps from the matching _tif_metadata.h5 file; their count is the
    # expected frame count. The fps comes from the acquisition config.
    timestamps, ts_fps = utils.load_timestamps(mkv_file)
    n_frames = len(timestamps)
    fps = _get_framerate(mkv_file, fallback=ts_fps)

    print('Sampling frames for the summary video contrast...')
    contrast = _video_contrast(mkv_file)
    summary_path = os.path.join(dir, SUMMARY_NAME)
    raw_summary_path = os.path.join(dir, SUMMARY_RAW_NAME)

    if motion_correct:
        # pass 1: uncorrected projections, which also seed the alignment template.
        # An existing summary_video.mp4 (without a _raw one) is the uncorrected
        # video of a first run, so it is kept as the _raw one instead of redone.
        print('Motion correction: building uncorrected projections and template...')
        raw_summary = None if (os.path.exists(raw_summary_path)
                               or os.path.exists(summary_path)) else raw_summary_path
        raw = _decode_projections(mkv_file, n_frames,
                                  summary_path=raw_summary, summary_fps=fps,
                                  video_contrast=contrast)
        template = _refine_template(raw.pop('samples'), raw['median'])
        if not os.path.exists(raw_summary_path):
            os.replace(summary_path, raw_summary_path)
        # pass 2: align every frame to the sharpened template
        proj = _decode_projections(mkv_file, n_frames, template=template,
                                   summary_path=summary_path,
                                   summary_fps=fps, video_contrast=contrast)
    else:
        raw = None
        proj = _decode_projections(mkv_file, n_frames,
                                   summary_path=summary_path, summary_fps=fps,
                                   video_contrast=contrast)

    n = proj['n']
    width, height = proj['width'], proj['height']

    # save projection images, cumulative stats, and mean fluorescence (plus the
    # uncorrected versions and the drift when motion-correcting)
    h5_path = os.path.join(dir, PROJ_H5_NAME)
    with h5py.File(h5_path, 'w') as f:
        for key in CORRECTED_DATASETS:
            f.create_dataset(key, data=proj[key])
        f.create_dataset('frame_count', data=n)
        if raw is not None:
            for key in CORRECTED_DATASETS:
                f.create_dataset(key + RAW_SUFFIX, data=raw[key])
            f.create_dataset('shifts', data=proj['shifts'])

    # save the preview PNGs and the mean fluorescence plot
    _write_pngs(h5_path)
    _plot_activity(mkv_file, proj['act'], act_path)

    # plot the estimated drift over time and save as a png
    if proj['shifts'] is not None:
        dy, dx = proj['shifts'][:, 0], proj['shifts'][:, 1]
        m = min(len(timestamps), len(dy))
        plt.figure(figsize=(10, 3))
        plt.plot(timestamps[:m] / 60, dx[:m], linewidth=0.5, label='x')
        plt.plot(timestamps[:m] / 60, dy[:m], linewidth=0.5, label='y')
        plt.xlim(0, timestamps[m - 1] / 60)
        plt.xlabel('Time (min)')
        plt.ylabel('Estimated drift (px)')
        plt.title(os.path.basename(mkv_file))
        plt.legend()
        plt.savefig(os.path.join(dir, DRIFT_SAVE_NAME), dpi=300)
        plt.close()

    # save metadata as a human-readable txt file
    size = [int(width), int(height)]
    with open(os.path.join(dir, METADATA_NAME), 'w') as f:
        f.write(f'mkv_path: {mkv_file}\n')
        f.write(f'size (width, height): {width} x {height}\n')
        f.write(f'num_frames: {n}\n')
        f.write(f'fps: {fps:.4f}\n')
        f.write(f'motion_corrected: {motion_correct}\n')

    return size, int(n), float(fps)


def _read_existing_metadata(mkv_file: str) -> tuple[list[int], int, float]:
    """
    Read size, frame count, and fps for an already-processed recording.

    Size and frame count are read from `projection_images.h5`; fps from the
    acquisition config (or the timestamps), see `_get_framerate()`.

    Parameters
    ----------
    mkv_file : str
        Absolute path to the source .mkv video file.

    Returns
    -------
    tuple[list[int], int, float]
        (size as [width, height], number of real frames, fps).
    """

    h5_path = os.path.join(os.path.dirname(mkv_file), PROJ_H5_NAME)
    with h5py.File(h5_path, 'r') as f:
        height, width = f['max_proj'].shape  # type: ignore[index]
        num_frames = int(f['frame_count'][()])  # type: ignore[index]

    _, ts_fps = utils.load_timestamps(mkv_file)
    fps = _get_framerate(mkv_file, fallback=ts_fps)

    return [int(width), int(height)], num_frames, float(fps)


def tif2mkv(tif_folder: str,
            overwrite: bool = False,
            delete_tifs: bool = False) -> str:
    """
    Convert a folder of raw FlyTrack tifs into a `_raw_tiff.mkv` and its metadata.

    Adapted from `export_tifs_as_mkv()` and `store_frame_metadata()` in
    flymuscle-control's ``export/export_exp_as_mkv.py``, but without moving the
    outputs to another folder (and deleting the raw tifs only on request, see
    `delete_tifs`). Use it for recordings
    whose tifs were never exported to .mkv, so the .mkv pipeline can pick them up.

    `tif_folder` must hold numbered RGBA tifs (``0.tif``, ``4.tif``, ...; each
    holds 4 real 16-bit frames, one per channel) and a ``<n>_metadata.csv`` per
    tif (one row per real frame: frame_id, timestamp in us, is_kinematic, and
    optionally stage_position). Each tif is first re-saved through `cv2` into a
    temporary `FIXED_TIF_DIR` subfolder (as in the original script) and these
    are then encoded losslessly by ``ffmpeg`` (ffv1, 16-bit RGBA). Two files are
    written to `tif_folder`, named ``<exp_name>_raw_tiff.mkv`` and
    ``<exp_name>_tif_metadata.h5``, where `exp_name` joins the last three folder
    names of `tif_folder` with underscores (e.g. ``<experiment>_Fly_001_001``).
    The temporary subfolder and concat list are removed afterwards; the ffmpeg
    log is kept only if encoding fails.

    If `delete_tifs` is True, the numbered tifs and their ``_metadata.csv``
    files are deleted once the conversion is verified: the .mkv must hold one
    container frame per tif and the h5 must hold 4 metadata rows per tif.
    Otherwise nothing is deleted and a warning is printed.

    Parameters
    ----------
    tif_folder : str
        Recording folder holding the numbered tifs and metadata csvs.
    overwrite : bool, optional
        If False (default) and the .mkv already exists, skip the conversion.
    delete_tifs : bool, optional
        If True, delete the raw tifs and metadata csvs after a verified
        conversion. Default False. Not applied when the conversion is skipped.

    Returns
    -------
    str
        Path to the `.mkv` file.
    """

    tif_folder = Path(tif_folder)
    exp_name = '_'.join(tif_folder.resolve().parts[-3:])
    mkv_path = tif_folder / f'{exp_name}_raw_tiff.mkv'
    h5_path = tif_folder / f'{exp_name}_tif_metadata.h5'

    if mkv_path.exists() and not overwrite:
        print(f'{mkv_path.name} already exists, skip.')
        return str(mkv_path)

    tif_files = sorted(tif_folder.glob('[0-9]*.tif'), key=lambda x: int(x.stem))
    assert len(tif_files) > 0, f'No numbered .tif files found in {tif_folder}'

    # re-save every tif through cv2 so ffmpeg can read them
    fixed_tif_folder = tif_folder / FIXED_TIF_DIR
    fixed_tif_folder.mkdir(exist_ok=True)
    with ThreadPoolExecutor() as executor:
        results = list(tqdm.tqdm(
            executor.map(_resave_tif, tif_files, [fixed_tif_folder] * len(tif_files)),
            total=len(tif_files), desc='Re-saving tifs'))
    failed = [r for r in results if r is not None]
    if failed:
        shutil.rmtree(fixed_tif_folder, ignore_errors=True)
        raise RuntimeError(f'{len(failed)} tifs could not be re-saved, e.g. {failed[0]}')

    # write the ffmpeg concat list (one entry per tif, each shown for TIF_DURATION s)
    list_path = tif_folder / TIF_LIST_NAME
    with open(list_path, 'w') as f:
        for tif_file in tif_files:
            f.write(f"file '{fixed_tif_folder / tif_file.name}'\n")
            f.write(f'duration {TIF_DURATION}\n')

    ffmpeg_command = [
        'ffmpeg',
        '-hwaccel', 'none',
        '-y',                    # overwrite output file if it exists
        '-f', 'concat',
        '-safe', '0',            # allow absolute paths
        '-i', str(list_path),
        '-c:v', 'ffv1',          # lossless codec
        '-pix_fmt', 'rgba64le',
        '-an',
        str(mkv_path),
    ]
    log_path = tif_folder / FFMPEG_LOG_NAME
    returncode = _run_ffmpeg_with_log(ffmpeg_command, log_path, len(tif_files))

    list_path.unlink()
    shutil.rmtree(fixed_tif_folder, ignore_errors=True)
    if returncode != 0:
        raise RuntimeError(f'ffmpeg failed (code {returncode}), see {log_path}')
    log_path.unlink()

    # gather the per-tif metadata csvs into a single h5
    n_meta = _store_tif_metadata(tif_files, h5_path)
    if len(tif_files) * 4 != n_meta:
        print(f'Warning: {len(tif_files)} tif files found, but {n_meta} metadata entries')

    print(f'Saved {mkv_path}')

    # delete the raw tifs and csvs, only if the .mkv and metadata are complete
    if delete_tifs:
        with av.open(str(mkv_path)) as container:
            n_mkv = sum(1 for packet in container.demux(video=0) if packet.size > 0)
        if n_mkv != len(tif_files) or n_meta != len(tif_files) * 4:
            print(f'Warning: {n_mkv} mkv frames and {n_meta} metadata entries for '
                  f'{len(tif_files)} tifs, NOT deleting the raw tifs.')
        else:
            for tif_file in tqdm.tqdm(tif_files, desc='Deleting tifs'):
                tif_file.with_name(tif_file.stem + '_metadata.csv').unlink(missing_ok=True)
                tif_file.unlink()
            print(f'Deleted {len(tif_files)} raw tifs and their metadata csvs')

    return str(mkv_path)


def _resave_tif(path: Path,
                output_dir: Path) -> str | None:
    """
    Re-save one RGBA tif into `output_dir` with `cv2`, for `tif2mkv()`.

    Parameters
    ----------
    path : Path
        Source tif.
    output_dir : Path
        Folder to write the re-saved tif to (same file name).

    Returns
    -------
    str or None
        None on success, otherwise a message describing the failure.
    """

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return f'Failed to read {path.name}'
    if img.ndim != 3 or img.shape[2] != 4:
        return f'Unexpected shape {img.shape} in {path.name}'
    if not cv2.imwrite(str(output_dir / path.name), img):
        return f'Failed to write {path.name}'
    return None


def _run_ffmpeg_with_log(ffmpeg_command: list[str],
                         log_path: Path,
                         n_frames: int) -> int:
    """
    Run ``ffmpeg``, logging its stderr to `log_path` and showing a progress bar.

    Parameters
    ----------
    ffmpeg_command : list of str
        Full ffmpeg command.
    log_path : Path
        File to write ffmpeg's stderr to (overwritten).
    n_frames : int
        Expected number of output (container) frames, for the progress bar.

    Returns
    -------
    int
        ffmpeg's return code.
    """

    with open(log_path, 'w') as log, subprocess.Popen(
            ffmpeg_command, start_new_session=True, stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL, text=True, bufsize=1) as process:
        progress_bar = tqdm.tqdm(total=n_frames, unit='frame', desc='Encoding mkv')
        for line in process.stderr:
            log.write(line)
            match = re.search(r'frame=\s*(\d+)', line)
            if match:
                progress_bar.update(int(match.group(1)) - progress_bar.n)
        progress_bar.close()
        process.wait()
    return process.returncode


def _store_tif_metadata(tif_files: list[Path],
                        h5_path: Path) -> int:
    """
    Merge the per-tif ``<n>_metadata.csv`` files into one h5, for `tif2mkv()`.

    Each csv row is one real frame (frame_id, timestamp, is_kinematic, and
    optionally stage_position). Values are stored as strings, as in
    flymuscle-control, which `utils.load_timestamps()` expects.

    Parameters
    ----------
    tif_files : list of Path
        Tif paths, in recording order.
    h5_path : Path
        Output ``_tif_metadata.h5`` path.

    Returns
    -------
    int
        Number of metadata rows (real frames) written.
    """

    frame_ids, timestamps, is_kinematic, stage_positions = [], [], [], []
    for tif_file in tif_files:
        csv_path = tif_file.with_name(tif_file.stem + '_metadata.csv')
        assert csv_path.exists(), f'Metadata file {csv_path} does not exist'
        with open(csv_path) as f:
            for row in csv.reader(f, delimiter=','):
                frame_ids.append(row[0])
                timestamps.append(row[1])
                is_kinematic.append(row[2])
                if len(row) > 3:
                    stage_positions.append(row[3])

    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('frame_ids', data=frame_ids)
        f.create_dataset('timestamps', data=timestamps)
        f.create_dataset('is_kinematic', data=is_kinematic)
        if stage_positions:
            f.create_dataset('stage_positions', data=stage_positions)
    return len(frame_ids)
