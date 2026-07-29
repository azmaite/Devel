"""
Load frames from FlyTrack's chunked raw `.tif` recordings.

This is another raw-recording format, alongside the `.mkv`-based pipeline in
`flytrack_core.py` and the chunked `.dat` format in `load_dat_files.py`: a
recording folder holds several `muscle_<N>` batch subfolders (e.g. `muscle_1`,
`muscle_4001`, `muscle_8001`, ...), each with 1000 `.tif` files named by the
(1-based) index of the first real frame they hold (`1.tif`, `5.tif`, `9.tif`,
...). Each `.tif` is a `height x width x 4` 16-bit image that packs 4
sequential real frames, one per channel (R, G, B, A in that order) -- the same
packing convention as the FlyTrack `.mkv` format's `rgba64le` frames (see
`flytrack_core.py`'s `_decode_projections()`).

IMPORTANT: these tifs must be read with `tifffile`, not `PIL`/`Pillow` -- PIL
silently misreads the 16-bit 4-channel data as 8-bit RGBA (values end up
byte-split and wrong), corrupting every pixel.

Also includes `tif_batch_to_mp4()`, for making an 8-bit mp4 preview of one
recording, and `make_preview_videos()`, for building previews of every Nth
recording in a fly's folder into a shared `videos` subfolder.
"""

import os
import re
import glob

import cv2
import numpy as np
import tifffile

MUSCLE_GLOB = 'muscle_*'                         # batch subfolders, in a recording folder
TIF_GLOB = '*.tif'                               # per real-frame-group tif, in a muscle_* folder
RECORDING_DIR_RE = re.compile(r'^\d{8}_\d{6}$')  # e.g. '20260724_165203'
VIDEOS_DIRNAME = 'videos'

# Intensity ceiling for the preview mp4s below, mapped to 8-bit 255. 400
# (based on a small early sample topping out around 300-350 counts) turned out
# to clip too much of the real range and looked oversaturated; 600 was chosen
# instead to give more headroom.
TIF_PREVIEW_CLIP = 600.0


def _list_recording_tifs(recording_folder: str) -> list[str]:
    """
    Every `.tif` in `recording_folder`'s `muscle_*` batch subfolders, in order.

    Parameters
    ----------
    recording_folder : str
        Path to one timestamped recording folder (holds `muscle_*` subfolders).

    Returns
    -------
    list of str
        `.tif` paths, sorted by the frame-group index encoded in each
        filename (e.g. `81.tif` before `101.tif`), across all `muscle_*`
        subfolders.
    """

    tif_paths = glob.glob(os.path.join(recording_folder, MUSCLE_GLOB, TIF_GLOB))
    return sorted(tif_paths,
                 key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))


def tif_batch_to_mp4(recording_folder: str,
                     output_path: str,
                     fps: float = 20.0,
                     clip: float = TIF_PREVIEW_CLIP) -> str:
    """
    Convert one chunked `.tif` recording to an 8-bit greyscale mp4 preview.

    Every `.tif` across the recording's `muscle_*` subfolders is read in order
    with `tifffile` (see the module docstring for why not `PIL`), and its 4
    packed real frames are streamed straight to the writer, so the whole
    recording is never held in memory at once. Since raw frames are 16-bit but
    mp4 only supports 8-bit, each frame is linearly rescaled from `[0, clip]`
    to `[0, 255]` (clipping brighter pixels).

    Parameters
    ----------
    recording_folder : str
        Path to one timestamped recording folder (holds `muscle_*` subfolders).
    output_path : str
        Path to write the `.mp4` to; its parent folder is created if needed.
    fps : float, optional
        Playback frame rate of the output mp4.
    clip : float, optional
        Raw intensity mapped to 255; values above are clipped. Use
        `TIF_PREVIEW_CLIP` as a starting point (see its definition for how it
        was picked).

    Returns
    -------
    str
        Path to the written `.mp4` file (`output_path`).

    Raises
    ------
    FileNotFoundError
        If `recording_folder` has no `muscle_*/*.tif` files.
    """

    tif_paths = _list_recording_tifs(recording_folder)
    if not tif_paths:
        raise FileNotFoundError(
            f'No muscle_*/*.tif files found in {recording_folder}')

    first_frame = tifffile.imread(tif_paths[0])
    height, width = first_frame.shape[:2]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for tif_path in tif_paths:
        img = tifffile.imread(tif_path).astype(np.float64)
        for channel in range(img.shape[2]):
            frame_8bit = np.clip(img[..., channel], 0, clip) / clip * 255
            writer.write(
                cv2.cvtColor(frame_8bit.astype(np.uint8), cv2.COLOR_GRAY2BGR))

    writer.release()

    return output_path


def make_preview_videos(fly_folder: str,
                        stride: int = 5,
                        fps: float = 20.0,
                        clip: float = TIF_PREVIEW_CLIP) -> list[str]:
    """
    Build preview mp4s for every `stride`-th recording in a fly's folder.

    `fly_folder`'s immediate subfolders are filtered to those named like a
    recording timestamp (`YYYYMMDD_HHMMSS`) and sorted chronologically; every
    `stride`-th one (starting from the first) is converted with
    `tif_batch_to_mp4()` into a shared `videos` subfolder of `fly_folder`
    (created if needed), named after its recording's timestamp.

    Parameters
    ----------
    fly_folder : str
        Path to one fly's folder, holding timestamped recording subfolders.
    stride : int, optional
        Keep 1 in every `stride` recordings.
    fps : float, optional
        Playback frame rate of the output mp4s.
    clip : float, optional
        Passed to `tif_batch_to_mp4()`.

    Returns
    -------
    list of str
        Paths to the written `.mp4` files, in chronological order.
    """

    recording_names = sorted(
        name for name in os.listdir(fly_folder)
        if RECORDING_DIR_RE.match(name)
        and os.path.isdir(os.path.join(fly_folder, name)))

    videos_dir = os.path.join(fly_folder, VIDEOS_DIRNAME)

    output_paths = []
    for name in recording_names[::stride]:
        output_path = os.path.join(videos_dir, f'{name}.mp4')
        tif_batch_to_mp4(os.path.join(fly_folder, name), output_path,
                         fps=fps, clip=clip)
        output_paths.append(output_path)

    return output_paths
