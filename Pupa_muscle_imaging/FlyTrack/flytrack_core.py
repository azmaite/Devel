#!/usr/bin/env python3
"""
Shared helpers for the FlyTrack .mkv preprocessing and motion-correction scripts.

Holds the constants, the single-video decode/projection routine, the
per-recording processing (decode a video -> save ``projection_images.h5``
(including a whole-video ``std`` image computed from the cumulative mean/mean_sq),
``metadata.txt``, ``activity.png``, ``median.png``, ``max_proj.png`` and
``std.png`` preview images, a half-resolution/compressed ``summary_video.mp4``
preview, and, when motion-correcting, ``drift.png`` and a
``summary_video_motioncorrected.mp4``), and the interactive terminal selection
helpers. These are imported by both
`preprocess_flytrack.py` (builds the grouped projections across recordings) and
`motion_correct_flytrack.py` (removes within-recording rigid drift), so neither
script depends on the other and either can be run first.

Optional within-recording rigid-translation drift correction (used only by
`motion_correct_flytrack.py`, via `_process_mkv(..., motion_correct=True)`):
drift is estimated by phase correlation on spatially high-pass-filtered frames
so the changing GCaMP brightness does not drive the alignment. The video is
decoded once to build a blurred median, that median is sharpened into a
template, and the video is decoded a second time to align every frame to it
(roughly doubling the decode time). The per-frame drift is saved to
``projection_images.h5`` as ``shifts`` and plotted as ``drift.png``.
"""

import os
import sys
from pathlib import Path

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
SUMMARY_NAME = 'summary_video.mp4'           # per-recording downsampled preview
SUMMARY_MC_NAME = 'summary_video_motioncorrected.mp4'  # per-recording, corrected
MKV_PATHS_NAME = 'mkv_paths.json'            # in the analysis folder
GROUPED_H5_NAME = 'projection_images_grouped.h5'  # in the analysis folder

# datasets required in projection_images.h5 for a recording to count as done.
# `std` is included so recordings processed before it was added (which cannot be
# backfilled) are reprocessed rather than skipped.
PROJ_DATASETS = ('median', 'min_proj', 'max_proj', 'mean', 'mean_sq', 'std',
                 'frame_count', 'act')

# number of evenly-spaced frames kept to estimate the median (memory bounded;
# the estimate is approximate but close to the true median)
MEDIAN_SAMPLES = 1500

# motion-correction (rigid-translation drift) parameters. Drift is estimated by
# phase correlation on spatially high-pass-filtered images so that structure
# (not the changing GCaMP brightness) drives the alignment. HP_SIGMA is the
# Gaussian sigma (px) of the low-pass that is subtracted to build the high-pass
# image; UPSAMPLE sets the subpixel precision (1/UPSAMPLE px); REFINE_ITERS is
# how many times the template is re-sharpened from the aligned median subsample
# before the full correction pass.
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
DOWNSAMPLE = 2

# summary preview video: every frame is streamed straight to a compressed mp4
# (so nothing is held in memory), spatially reduced by averaging 2x2 pixel blocks
# (half resolution in each direction), and mapped to 8-bit by clipping
# intensities to SUMMARY_CLIP. It plays back at the recording's real frame rate.
SUMMARY_CLIP = 600


def _select_mkv_paths(input_folder: str) -> list[str]:
    """
    List the input folder's subfolders and let the user pick which to include.

    The immediate subfolders of `input_folder` are listed in the terminal, each
    with a number and its count of top-level .mkv files. Analysis folders (any
    subfolder containing a `mkv_paths.json`) and subfolders with no top-level
    .mkv files are excluded from the list. The user picks which to include with a
    comma/space-separated list of numbers and ranges (e.g. ``1,3,5-7``), ``0`` or
    ``all`` to include every listed subfolder, or ``q`` to cancel. The top-level
    .mkv files of every chosen subfolder are collected, de-duplicated while
    preserving order, and returned.

    Parameters
    ----------
    input_folder : str
        Folder whose immediate subfolders are offered for selection.

    Returns
    -------
    list of str
        Ordered, de-duplicated list of selected .mkv file paths.
    """

    # immediate subfolders, excluding analysis folders (those with mkv_paths.json)
    candidates = sorted(
        p for p in Path(input_folder).iterdir()
        if p.is_dir() and not (p / MKV_PATHS_NAME).exists())

    # keep only subfolders that actually contain top-level .mkv files
    subfolders = []
    mkv_per_folder = []
    for sub in candidates:
        mkvs = sorted(str(p) for p in sub.glob('*.mkv'))
        if mkvs:
            subfolders.append(sub)
            mkv_per_folder.append(mkvs)

    if len(subfolders) == 0:
        print(f'No subfolders with .mkv files found in {input_folder}.')
        return []

    # list each subfolder with its top-level .mkv count
    print(f'\nSubfolders in {input_folder}:')
    print('  [0] all')
    for i, (sub, mkvs) in enumerate(zip(subfolders, mkv_per_folder)):
        print(f'  [{i + 1}] {sub.name}  ({len(mkvs)} .mkv)')

    # ask which subfolders to include
    selection = input("\nWhich subfolders to include? "
                      "e.g. '1,3,5-7', '0' or 'all', or 'q' to cancel: ").strip().lower()
    if selection in ('q', 'quit', ''):
        return []
    if selection == 'all' or '0' in selection.replace(',', ' ').split():
        chosen = list(range(len(subfolders)))
    else:
        chosen = _parse_selection(selection, len(subfolders))

    # collect top-level .mkv files from every chosen subfolder
    mkv_paths: list[str] = []
    for idx in chosen:
        mkv_paths.extend(mkv_per_folder[idx])

    # de-duplicate while preserving order
    seen: set[str] = set()
    unique = [p for p in mkv_paths if not (p in seen or seen.add(p))]
    return unique


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


def _outputs_exist(dir: str, motion_correct: bool = False) -> bool:
    """
    Check whether a recording's per-folder outputs already exist.

    Returns True only if both `metadata.txt` and a `projection_images.h5`
    containing all required datasets are present in `dir`. When `motion_correct`
    is requested, the `shifts` dataset must also be present, so that a recording
    processed without correction is reprocessed rather than skipped.

    Parameters
    ----------
    dir : str
        Folder containing the .mkv recording.
    motion_correct : bool, optional
        Whether motion correction is requested for this run.

    Returns
    -------
    bool
        Whether the recording can be skipped.
    """

    metadata_path = os.path.join(dir, METADATA_NAME)
    h5_path = os.path.join(dir, PROJ_H5_NAME)
    if not (os.path.exists(metadata_path) and os.path.exists(h5_path)):
        return False

    required = PROJ_DATASETS + (('shifts',) if motion_correct else ())
    try:
        with h5py.File(h5_path, 'r') as f:
            return all(key in f for key in required)
    except OSError:
        return False


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

    template = template.astype(np.float64)
    k = samples.shape[0] // 2
    for it in range(iters):
        ref_hp = _reference_hp(template, hp_sigma, downsample)
        aligned = np.empty(samples.shape, dtype=np.float32)
        for i, s in enumerate(tqdm.tqdm(samples,
                                        desc=f'Refining template {it+1}/{iters}',
                                        unit='frame')):
            shift = _estimate_shift(ref_hp, s, hp_sigma, upsample, downsample)
            aligned[i] = ndimage.shift(s.astype(np.float64), shift,
                                       order=1, mode='nearest')
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


def _decode_projections(mkv_file: str,
                        template: np.ndarray | None = None,
                        hp_sigma: float = HP_SIGMA,
                        upsample: int = UPSAMPLE,
                        downsample: int = DOWNSAMPLE,
                        n_frames: int | None = None,
                        summary_path: str | None = None,
                        summary_fps: float = 30.0) -> dict:
    """
    Decode one .mkv once and accumulate its projection images.

    Decodes the video (4 real frames packed per container frame, 16-bit) and
    computes the min and max projections, the cumulative mean and mean of
    squares (for the std image), the per-frame mean fluorescence, and an
    evenly-spaced frame subsample used to estimate the median.

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
    template : np.ndarray or None, optional
        Full-resolution reference image to align every frame to. If None, no
        correction is done and the projections are built from the raw frames.
    hp_sigma : float, optional
        High-pass sigma passed to `_highpass()`.
    upsample : int, optional
        Subpixel upsampling factor for `phase_cross_correlation()`.
    downsample : int, optional
        Integer downsampling factor for registration (see `_estimate_shift()`).
    n_frames : int or None, optional
        Expected number of real frames, used only as the progress-bar total
        (.mkv does not expose a frame count). If None, the bar shows no ETA.
    summary_path : str or None, optional
        If given, a half-resolution (2x2-averaged), compressed preview mp4 of the
        (corrected, if `template` is set) frames is streamed to this path, one
        entry per frame.
    summary_fps : float, optional
        Playback frame rate of the preview mp4 (use the real recording fps).

    Returns
    -------
    dict
        Keys: 'median', 'min_proj', 'max_proj' (uint16), 'mean', 'mean_sq'
        (float64), 'act' (float64, per-frame mean), 'samples' (uint16 subsample
        stack, for template refinement), 'shifts' ((n, 2) float64 (dy, dx) or
        None if uncorrected), 'n', 'width', 'height'.
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

    # evenly-spaced subsample of frames for the (approximate) median. We keep
    # every `median_stride`-th frame; whenever the buffer grows too large we
    # halve it and double the stride, keeping the sample roughly even.
    median_samples: list[np.ndarray] = []
    median_stride = 1

    # optional preview video: every frame is streamed to a compressed mp4 (half
    # resolution via 2x2 averaging), so nothing is held in memory
    summary = summary_path is not None
    if summary:
        sh, sw = (height - height % 2) // 2, (width - width % 2) // 2
        summary_writer = cv2.VideoWriter(
            summary_path, cv2.VideoWriter_fourcc(*'mp4v'),
            summary_fps, (sw, sh))

    n = 0  # real-frame counter
    container = av.open(mkv_file)
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'

    # .mkv (Matroska) does not expose a frame count, so tqdm's total comes from
    # the caller's `n_frames` hint (the timestamp count); the bar counts real
    # frames and is updated once per decoded packet (which holds 4 of them)
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

            if n % median_stride == 0:
                sample = np.clip(np.round(img_f), 0, 65535).astype(np.uint16)
                median_samples.append(sample)
                if len(median_samples) >= 2 * MEDIAN_SAMPLES:
                    median_samples = median_samples[::2]
                    median_stride *= 2

            if summary:
                small = _bin2x2(img_f)
                small = np.clip(small, 0, SUMMARY_CLIP) / SUMMARY_CLIP * 255
                summary_writer.write(
                    cv2.cvtColor(small.astype(np.uint8), cv2.COLOR_GRAY2BGR))

            n += 1

        pbar.update(img.shape[2])

    pbar.close()
    container.close()
    print(f'Decoded {n} real frames.')

    # cumulative statistics
    mean = sum_ / n
    mean_sq = sum_sq / n

    # approximate median from the evenly-spaced subsample
    samples = np.stack(median_samples)
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
        'act': np.array(act),
        'samples': samples,
        'shifts': np.array(shifts) if correct else None,
        'n': n,
        'width': int(width),
        'height': int(height),
    }


def _autocontrast_8bit(img: np.ndarray, hi_pct: float = 99.8) -> np.ndarray:
    """
    Map an image to 8-bit, auto-contrasted between its minimum and a percentile.

    The image is linearly rescaled so its minimum maps to 0 and its `hi_pct`
    percentile maps to 255 (values above are clipped), which keeps bright
    outliers from washing out the visible range. If the percentile is not above
    the minimum, the true maximum is used instead.

    Parameters
    ----------
    img : np.ndarray
        Image to rescale.
    hi_pct : float, optional
        Percentile mapped to 255.

    Returns
    -------
    np.ndarray
        The auto-contrasted image as uint8.
    """

    img_f = img.astype(np.float64)
    lo = img_f.min()
    hi = np.percentile(img_f, hi_pct)
    if hi <= lo:
        hi = img_f.max()
    scaled = np.clip((img_f - lo) / (hi - lo + 1e-9), 0, 1) * 255
    return scaled.astype(np.uint8)


def _write_pngs(path: str) -> None:
    """
    Save auto-contrasted 8-bit preview PNGs next to a projection-images h5.

    Reads the `median`, `max_proj`, and `std` datasets from a projection-images
    h5 and writes three auto-contrasted 8-bit PNGs in the same folder:
    `median.png` (the median image), `max_proj.png` (the max projection), and
    `std.png` (the std image). The `std` dataset must
    already exist (it is computed during preprocessing); it is not rebuilt here,
    so re-run preprocessing if it is missing.

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

    with h5py.File(h5_path, 'r') as f:
        median = f['median'][:]       # type: ignore[index]
        max_proj = f['max_proj'][:]   # type: ignore[index]
        std = f['std'][:]             # type: ignore[index]

    # median image, auto-contrasted min..99.8th percentile
    cv2.imwrite(os.path.join(dir, MEDIAN_SAVE_NAME), _autocontrast_8bit(median))

    # max projection, auto-contrasted min..99.8th percentile
    cv2.imwrite(os.path.join(dir, MAX_PROJ_SAVE_NAME),
                _autocontrast_8bit(max_proj))

    # std image, auto-contrasted min..99.8th percentile
    cv2.imwrite(os.path.join(dir, STD_SAVE_NAME), _autocontrast_8bit(std))



def _process_mkv(mkv_file: str,
                 motion_correct: bool = False) -> tuple[list[int], int, float]:
    """
    Decode one .mkv and write its projection images and metadata.

    Builds the projection images with `_decode_projections()` and saves them to
    `projection_images.h5` and `metadata.txt` in the same folder as the video,
    plus the `activity.png` mean-fluorescence plot. The whole-video std image is
    rebuilt from the cumulative mean/mean_sq and stored in the h5 as `std`, and
    two auto-contrasted preview PNGs are written next to the video: `median.png`
    (the median image) and `max_proj.png` (the max projection).

    If `motion_correct` is True, rigid-translation drift is removed first: the
    video is decoded once to build a (drift-blurred) median, that median is
    sharpened into a template with `_refine_template()`, and the video is
    decoded a second time to build drift-corrected projections. This roughly
    doubles the decode time. The per-frame drift is then also saved to the h5 as
    `shifts` and plotted as `drift.png`.

    Parameters
    ----------
    mkv_file : str
        Absolute path to the source .mkv video file.
    motion_correct : bool, optional
        Whether to correct rigid-translation drift before building projections.

    Returns
    -------
    tuple[list[int], int, float]
        (size as [width, height], number of real frames, fps).
    """

    dir = os.path.dirname(mkv_file)

    # timestamps and fps from the matching _tif_metadata.h5 file; the timestamp
    # count is the expected frame count used for the decode progress-bar totals
    timestamps, fps = utils.load_timestamps(mkv_file)
    n_frames = len(timestamps)

    # the summary video is written from the frames the outputs are built from
    # (the corrected frames when motion-correcting), with a distinct name
    summary_name = SUMMARY_MC_NAME if motion_correct else SUMMARY_NAME
    summary_path = os.path.join(dir, summary_name)

    if motion_correct:
        # pass 1: build a drift-blurred median to seed the alignment template
        print('Motion correction: building initial template...')
        seed = _decode_projections(mkv_file, n_frames=n_frames)
        template = _refine_template(seed['samples'], seed['median'])
        # pass 2: align every frame to the sharpened template
        proj = _decode_projections(mkv_file, template=template,
                                   n_frames=n_frames, summary_path=summary_path,
                                   summary_fps=fps)
    else:
        proj = _decode_projections(mkv_file, n_frames=n_frames,
                                   summary_path=summary_path, summary_fps=fps)

    n = proj['n']
    width, height = proj['width'], proj['height']

    # whole-video std image from the cumulative statistics: std = sqrt(E[x^2] - E[x]^2)
    std = np.sqrt(np.clip(proj['mean_sq'] - proj['mean'] ** 2, 0, None))

    # save projection images, cumulative stats, and mean fluorescence
    h5_path = os.path.join(dir, PROJ_H5_NAME)
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('median', data=proj['median'])
        f.create_dataset('min_proj', data=proj['min_proj'])
        f.create_dataset('max_proj', data=proj['max_proj'])
        f.create_dataset('mean', data=proj['mean'])
        f.create_dataset('mean_sq', data=proj['mean_sq'])
        f.create_dataset('std', data=std)
        f.create_dataset('frame_count', data=n)
        f.create_dataset('act', data=proj['act'])
        if proj['shifts'] is not None:
            f.create_dataset('shifts', data=proj['shifts'])

    # save the median / max projection / std preview PNGs
    _write_pngs(h5_path)

    # plot the mean fluorescence over time and save as a png
    act = proj['act']
    if len(timestamps) != len(act):
        print(f'Warning: number of timestamps ({len(timestamps)}) does not match '
              f'number of frames ({len(act)}), LOST FRAMES.')
        timestamps = timestamps[:len(act)]
    plt.figure(figsize=(10, 3))
    plt.plot(timestamps / 60, act, linewidth=0.5)  # in minutes
    plt.xlim(0, timestamps[-1] / 60)
    plt.xlabel('Time (min)')
    plt.ylabel('Mean fluorescence')
    plt.title(os.path.basename(mkv_file))
    plt.savefig(os.path.join(dir, ACT_SAVE_NAME), dpi=300)
    plt.close()

    # plot the estimated drift over time and save as a png
    if proj['shifts'] is not None:
        dy, dx = proj['shifts'][:, 0], proj['shifts'][:, 1]
        plt.figure(figsize=(10, 3))
        plt.plot(timestamps / 60, dx[:len(timestamps)], linewidth=0.5, label='x')
        plt.plot(timestamps / 60, dy[:len(timestamps)], linewidth=0.5, label='y')
        plt.xlim(0, timestamps[-1] / 60)
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

    Size and frame count are read from `projection_images.h5`; fps is derived
    from the matching timestamps file via `utils.load_timestamps()`.

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

    _, fps = utils.load_timestamps(mkv_file)

    return [int(width), int(height)], num_frames, float(fps)
