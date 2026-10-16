"""
Load frames from FlyTrack's chunked raw `.dat` recordings.

This is a new raw-recording format being tested as a replacement for the
`.mkv`-based FlyTrack pipeline (see `flytrack_core.py`); it holds the same
kind of data (single-channel greyscale video frames with per-frame
timestamps), just written directly to disk instead of encoded as video.

A recording folder holds, for each ~10-minute chunk:
    - `chunk_XXXX.dat`             : real frames, back-to-back, no header.
      Each frame is `height x width` pixels of `bit_depth`-bit unsigned
      integers (little-endian), row-major.
    - `chunk_XXXX_timestamps.csv`  : one row per frame in the matching
      `.dat`, with columns `frame_id`, `camera_timestamp_us`,
      `host_wall_clock_ns`.
and, once, a single `metadata.json` describing the recording (`resolution`,
`bit_depth`, `framerate_hz`, plus acquisition bookkeeping).

Because the recording may still be in progress, the last chunk can be
partially written; frame counts are always taken from each chunk file's
current size rather than any nominal per-chunk frame count.

Also includes `move_dat_chunks_to_raw_data()`, for tidying a recording
folder by moving its `chunk_*.dat`/`_timestamps.csv` files into a `raw_data`
subfolder, and `dat_to_mp4()`, for making an 8-bit mp4 preview of one chunk
(e.g. to spot-check a long recording without decoding every chunk); given
just a `chunk_*.dat` path, it picks the right `videos` output folder
depending on whether the chunk lives in a `raw_data` subfolder or not.
"""

import os
import csv
import glob
import json
import shutil

import cv2
import numpy as np

METADATA_NAME = 'metadata.json'        # in the recording folder
CHUNK_GLOB = 'chunk_*.dat'             # raw frame chunks, in the recording folder
TIMESTAMPS_SUFFIX = '_timestamps.csv'  # per chunk, next to each chunk_*.dat

# Rough intensity ceiling for the preview mp4s below: sampled frames across
# several chunks of a PCO Panda 4.2 recording topped out around 300-320
# counts, so 400 gives some headroom before clipping to 8-bit.
DAT_PREVIEW_CLIP = 400.0


def load_dat_metadata(folder: str) -> dict:
    """
    Read a `.dat` recording's `metadata.json` into the fields needed to
    decode its frames.

    Parameters
    ----------
    folder : str
        Path to a chunked `.dat` recording folder, containing `metadata.json`.

    Returns
    -------
    dict
        - 'width', 'height' : int, frame size in pixels.
        - 'dtype' : `np.dtype`, the raw (little-endian, unsigned) pixel type.
        - 'fps' : float, the recording's framerate.
    """

    with open(os.path.join(folder, METADATA_NAME), 'r') as f:
        meta = json.load(f)

    return {
        'width': int(meta['resolution']['width']),
        'height': int(meta['resolution']['height']),
        'dtype': np.dtype(f"<u{meta['bit_depth'] // 8}"),
        'fps': float(meta['framerate_hz']),
    }


def _list_chunks(folder: str) -> list[str]:
    """
    Return every `chunk_*.dat` in `folder`, sorted in recording order.

    Parameters
    ----------
    folder : str
        Path to a chunked `.dat` recording folder.

    Returns
    -------
    list of str
        Sorted `chunk_*.dat` paths.
    """

    return sorted(glob.glob(os.path.join(folder, CHUNK_GLOB)))


def _chunk_frame_count(chunk_path: str, frame_bytes: int) -> int:
    """
    Number of whole frames currently written to one chunk file.

    Uses the file's current size rather than any nominal per-chunk frame
    count, since the last chunk of an in-progress recording may be partially
    written; any trailing partial frame is dropped by the integer division.

    Parameters
    ----------
    chunk_path : str
        Path to a `chunk_*.dat` file.
    frame_bytes : int
        Size of one frame in bytes (`width * height * dtype.itemsize`).

    Returns
    -------
    int
        Number of complete frames in `chunk_path`.
    """

    return os.path.getsize(chunk_path) // frame_bytes


def _open_chunk(chunk_path: str, meta: dict) -> np.memmap:
    """
    Memory-map one chunk file as an `(n_frames, height, width)` array.

    Frames are stored back-to-back with no header, so this is a plain
    reshape of the raw bytes; nothing is read from disk until a frame is
    actually indexed.

    Parameters
    ----------
    chunk_path : str
        Path to a `chunk_*.dat` file.
    meta : dict
        Result of `load_dat_metadata()`.

    Returns
    -------
    np.memmap
        Read-only memory-mapped view, shape `(n_frames, height, width)`.
    """

    frame_bytes = meta['width'] * meta['height'] * meta['dtype'].itemsize
    n_frames = _chunk_frame_count(chunk_path, frame_bytes)
    return np.memmap(chunk_path, dtype=meta['dtype'], mode='r',
                     shape=(n_frames, meta['height'], meta['width']))


def get_dat_frame_count(folder: str, meta: dict | None = None) -> int:
    """
    Total number of frames currently written across all chunks in `folder`.

    Parameters
    ----------
    folder : str
        Path to a chunked `.dat` recording folder.
    meta : dict or None, optional
        Result of `load_dat_metadata(folder)`. Loaded automatically if None.

    Returns
    -------
    int
        Number of complete frames currently on disk.
    """

    if meta is None:
        meta = load_dat_metadata(folder)

    frame_bytes = meta['width'] * meta['height'] * meta['dtype'].itemsize
    return sum(_chunk_frame_count(p, frame_bytes) for p in _list_chunks(folder))


def iter_dat_frames(folder: str, meta: dict | None = None):
    """
    Yield every frame of a chunked `.dat` recording, in order.

    Mirrors decoding an FlyTrack `.mkv` frame by frame (e.g.
    `container.decode(stream)` in `flytrack_core._decode_projections`): each
    chunk is memory-mapped in turn and its frames are yielded one at a time,
    so at most one chunk is ever paged into memory.

    Parameters
    ----------
    folder : str
        Path to a chunked `.dat` recording folder.
    meta : dict or None, optional
        Result of `load_dat_metadata(folder)`. Loaded automatically if None.

    Yields
    ------
    tuple of (int, np.ndarray)
        (global frame index, frame), where frame has shape `(height, width)`
        and dtype `meta['dtype']`.
    """

    if meta is None:
        meta = load_dat_metadata(folder)

    frame_idx = 0
    for chunk_path in _list_chunks(folder):
        chunk = _open_chunk(chunk_path, meta)
        for i in range(chunk.shape[0]):
            yield frame_idx, np.array(chunk[i])
            frame_idx += 1


def load_dat_frame(folder: str,
                   frame_index: int,
                   meta: dict | None = None) -> np.ndarray:
    """
    Load a single frame from a chunked `.dat` recording by global index.

    Only the chunk containing `frame_index` is memory-mapped, so this does
    not read the rest of the recording. For reading many frames in order,
    prefer `iter_dat_frames()`, which avoids rescanning earlier chunks.

    Parameters
    ----------
    folder : str
        Path to a chunked `.dat` recording folder.
    frame_index : int
        0-based index of the frame to load, over the whole recording
        (across chunks, in chunk order).
    meta : dict or None, optional
        Result of `load_dat_metadata(folder)`. Loaded automatically if None.

    Returns
    -------
    np.ndarray
        The frame, shape `(height, width)`, dtype `meta['dtype']`.

    Raises
    ------
    IndexError
        If `frame_index` is beyond the frames currently written to `folder`.
    """

    if meta is None:
        meta = load_dat_metadata(folder)

    frame_bytes = meta['width'] * meta['height'] * meta['dtype'].itemsize
    remaining = frame_index
    for chunk_path in _list_chunks(folder):
        n_frames = _chunk_frame_count(chunk_path, frame_bytes)
        if remaining < n_frames:
            chunk = _open_chunk(chunk_path, meta)
            return np.array(chunk[remaining])
        remaining -= n_frames

    total_frames = frame_index - remaining
    raise IndexError(f'frame_index {frame_index} is beyond the {total_frames} '
                     f'frames currently written in {folder}')


def load_dat_timestamps(folder: str) -> tuple[np.ndarray, float]:
    """
    Load every chunk's per-frame timestamps, concatenated in chunk order.

    Timestamps are read from each chunk's `camera_timestamp_us` column and
    converted to SECONDS relative to the first frame, mirroring
    `utils.load_timestamps()` for the `.mkv` format. Row order within each
    `_timestamps.csv` is assumed to match its `.dat`'s frame order, so
    timestamp index `i` corresponds to `load_dat_frame(folder, i)`.

    Parameters
    ----------
    folder : str
        Path to a chunked `.dat` recording folder.

    Returns
    -------
    tuple of (np.ndarray, float)
        (timestamps in seconds, fps).
    """

    timestamps_us: list[int] = []
    for chunk_path in _list_chunks(folder):
        ts_path = chunk_path[:-len('.dat')] + TIMESTAMPS_SUFFIX
        with open(ts_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            timestamps_us.extend(int(row['camera_timestamp_us']) for row in reader)

    timestamps = np.asarray(timestamps_us, dtype=np.int64)
    timestamps = timestamps - timestamps[0]
    timestamps = timestamps / 1e6  # convert to seconds

    fps = 1 / np.median(np.diff(timestamps))

    return timestamps, float(fps)


def move_dat_chunks_to_raw_data(folder: str, dest_dirname: str = 'raw_data') -> str:
    """
    Move every `chunk_*.dat` and its `_timestamps.csv` into a subfolder.

    `metadata.json` is copied (not moved) into the destination too, so it
    stays self-sufficient for `load_dat_metadata()` and friends, while a copy
    also remains in `folder` for quick browsing.

    Parameters
    ----------
    folder : str
        Path to a chunked `.dat` recording folder.
    dest_dirname : str, optional
        Name of the subfolder to create under `folder` and move chunks into.

    Returns
    -------
    str
        Path to the destination subfolder.
    """

    dest = os.path.join(folder, dest_dirname)
    os.makedirs(dest, exist_ok=True)

    for chunk_path in _list_chunks(folder):
        ts_path = chunk_path[:-len('.dat')] + TIMESTAMPS_SUFFIX
        shutil.move(chunk_path, os.path.join(dest, os.path.basename(chunk_path)))
        if os.path.exists(ts_path):
            shutil.move(ts_path, os.path.join(dest, os.path.basename(ts_path)))

    shutil.copy2(os.path.join(folder, METADATA_NAME),
                 os.path.join(dest, METADATA_NAME))

    return dest


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
        The 2x2-averaged image, shape `(H // 2, W // 2)`.
    """

    height, width = img.shape
    h2, w2 = height - height % 2, width - width % 2
    return img[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2).mean(axis=(1, 3))


def dat_to_mp4(chunk_path: str,
               meta: dict | None = None,
               clip: float = DAT_PREVIEW_CLIP,
               downsample: bool = False) -> str:
    """
    Convert a `chunk_*.dat` file to an 8-bit greyscale mp4 preview.

    Uses the same `cv2.VideoWriter` / `mp4v` approach as the summary preview
    video in `flytrack_core.py`'s `_decode_projections()`. Frames are
    memory-mapped and streamed straight to the writer, so the whole chunk is
    never held in memory at once. Since raw frames are 16-bit but mp4 only
    supports 8-bit, each frame is linearly rescaled from `[0, clip]` to
    `[0, 255]` (clipping brighter pixels). (The .mkv summary video in
    `flytrack_core.py` instead uses a contrast estimated from sampled frames.)

    The output path is picked from `chunk_path`'s location: if `chunk_path`
    sits directly inside a `raw_data` folder (as left by
    `move_dat_chunks_to_raw_data()`), the mp4 is written to a `videos` folder
    next to `raw_data` (i.e. in its parent, the recording folder); otherwise
    it's written to a `videos` folder inside `chunk_path`'s own parent
    folder. Either way, `videos` is created if it doesn't already exist.

    Parameters
    ----------
    chunk_path : str
        Path to the `chunk_*.dat` file to convert.
    meta : dict or None, optional
        Result of `load_dat_metadata()`. If None, loaded automatically from
        the recording folder (`chunk_path`'s parent, or grandparent if
        `chunk_path` is inside `raw_data`).
    clip : float, optional
        Raw intensity mapped to 255; values above are clipped. Use
        `DAT_PREVIEW_CLIP` as a starting point (see its definition for how it
        was picked).
    downsample : bool, optional
        If True, 2x2-average frames to half resolution before encoding.

    Returns
    -------
    str
        Path to the written `.mp4` file.
    """

    chunk_dir = os.path.dirname(os.path.abspath(chunk_path))
    if os.path.basename(chunk_dir) == 'raw_data':
        recording_folder = os.path.dirname(chunk_dir)
    else:
        recording_folder = chunk_dir

    if meta is None:
        meta = load_dat_metadata(recording_folder)

    videos_dir = os.path.join(recording_folder, 'videos')
    os.makedirs(videos_dir, exist_ok=True)

    output_path = os.path.join(
        videos_dir, os.path.basename(chunk_path)[:-len('.dat')] + '.mp4')

    chunk = _open_chunk(chunk_path, meta)
    height, width = meta['height'], meta['width']
    if downsample:
        height, width = (height - height % 2) // 2, (width - width % 2) // 2

    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*'mp4v'), meta['fps'], (width, height))

    for i in range(chunk.shape[0]):
        frame = chunk[i].astype(np.float64)
        if downsample:
            frame = _bin2x2(frame)
        frame_8bit = np.clip(frame, 0, clip) / clip * 255
        writer.write(cv2.cvtColor(frame_8bit.astype(np.uint8), cv2.COLOR_GRAY2BGR))

    writer.release()

    return output_path
