#!/usr/bin/env python3
"""
Build small synthetic FlyTrack fixtures from a real recording folder.

The real .mkv recordings are ~10 GB each: 6000 container frames, every one
holding 4 real 16-bit frames packed as the 4 channels of an ffv1/gbrap16le
video (24000 real frames, matching the 24000 rows of the `_tif_metadata.h5`).
This script copies a couple of such recordings into a tiny fixture tree by
keeping only `N_KEEP` real frames, evenly interspaced across each video, and
subsampling the metadata rows to the exact same frame indices so frames and
timestamps stay aligned.

`N_KEEP` is a multiple of 4 so the kept frames repack into whole container
frames (`N_KEEP // 4` of them) and the pipeline decodes exactly `N_KEEP` real
frames against `N_KEEP` timestamps (no LOST FRAMES mismatch).

For each source subfolder only three files are reproduced: the `_raw_tiff.mkv`
(repacked, same name), the `_tif_metadata.h5` (subsampled, same name), and
`config_concatenated.yaml` (copied verbatim). The output tree is:

    _synthetic-test-data/
      20260701_test-flytrack-data_prepup_06-29_10h55/
        001/  <mkv> <metadata.h5> config_concatenated.yaml
        002/  ...
"""

import os
import shutil

import numpy as np
import av
import h5py

# source recording folders (each an immediate subfolder holding one .mkv) and
# the fixture folders they map to, all under MAIN below
MAIN = '/mnt/labserver/data/MA/Development_project/Pupa_muscle_long_recordings'
SRC_ROOT = os.path.join(MAIN, '20260422_V7_MHC-GCaMP8m_SCAPE', 'Fly_001')
DST_ROOT = os.path.join(MAIN, '_synthetic-test-data',
                        '20260701_test-flytrack-data_prepup_06-29_10h55')
SUBFOLDERS = ['001', '002']

# names of the three files reproduced per subfolder (config copied verbatim)
CONFIG_NAME = 'config_concatenated.yaml'

# real frames kept per recording (a multiple of 4 so they repack into whole
# 4-real-frames-per-container frames)
N_KEEP = 12


def _find_one(folder: str, suffix: str) -> str:
    """
    Return the single file in `folder` whose name ends with `suffix`.

    Parameters
    ----------
    folder : str
        Folder to search (top level only).
    suffix : str
        Filename suffix to match, e.g. '_raw_tiff.mkv'.

    Returns
    -------
    str
        Absolute path to the matching file.
    """

    matches = [f for f in os.listdir(folder) if f.endswith(suffix)]
    assert len(matches) == 1, f'Expected one *{suffix} in {folder}, got {matches}'
    return os.path.join(folder, matches[0])


def _extract_frames(mkv_file: str,
                    real_indices: list[int]) -> np.ndarray:
    """
    Decode the given real-frame indices from a packed .mkv recording.

    Each container frame packs 4 real frames as its 4 rgba64le channels, so real
    frame `r` is channel `r % 4` of container frame `r // 4`. The needed
    container frames are seeked to individually (ffv1 is all-intra, so seeking is
    frame-accurate and cheap); the seek lands at or just before the target, then
    we step forward to the exact pts.

    Parameters
    ----------
    mkv_file : str
        Path to the source .mkv.
    real_indices : list of int
        Real-frame indices to extract, in ascending order.

    Returns
    -------
    np.ndarray
        Stack of the extracted 16-bit frames, shape (len(real_indices), H, W).
    """

    container = av.open(mkv_file)
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'
    rate = float(stream.average_rate)
    tb = float(stream.time_base)
    pts_per_frame = int(round((1.0 / rate) / tb))  # container-frame pts step

    out: list[np.ndarray] = []
    for r in real_indices:
        cframe, channel = r // 4, r % 4
        target_pts = cframe * pts_per_frame
        # seek a little before the target, then decode forward to the exact frame
        container.seek(max(target_pts - pts_per_frame, 0), stream=stream)
        chosen = None
        for frame in container.decode(stream):
            idx = int(round(frame.pts * tb * rate))
            if idx == cframe:
                chosen = frame.to_ndarray(format='rgba64le')[..., channel]
                break
            if idx > cframe:
                raise RuntimeError(f'Overshot container frame {cframe} (got {idx})')
        assert chosen is not None, f'Container frame {cframe} not found'
        out.append(chosen.copy())

    container.close()
    return np.stack(out)


def _write_packed_mkv(frames: np.ndarray,
                      out_path: str) -> None:
    """
    Repack real frames into an ffv1/gbrap16le .mkv, 4 real frames per container.

    Groups the frames in order into container frames of 4 channels each (channel
    `c` of container `k` is `frames[k * 4 + c]`) and encodes them losslessly,
    matching the source recordings' codec and pixel format so the pipeline reads
    them back identically via `to_ndarray('rgba64le')`.

    Parameters
    ----------
    frames : np.ndarray
        Stack of 16-bit frames, shape (N, H, W); N must be a multiple of 4.
    out_path : str
        Destination .mkv path.
    """

    n, height, width = frames.shape
    assert n % 4 == 0, f'Frame count {n} must be a multiple of 4'

    out = av.open(out_path, 'w')
    stream = out.add_stream('ffv1', rate=25)
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'gbrap16le'

    for k in range(n // 4):
        packed = np.stack([frames[k * 4 + c] for c in range(4)], axis=-1)
        vframe = av.VideoFrame.from_ndarray(packed.astype(np.uint16),
                                            format='rgba64le')
        vframe = vframe.reformat(format='gbrap16le')
        for pkt in stream.encode(vframe):
            out.mux(pkt)
    for pkt in stream.encode():  # flush
        out.mux(pkt)
    out.close()


def _subsample_metadata(src_h5: str,
                        dst_h5: str,
                        real_indices: list[int]) -> None:
    """
    Copy a `_tif_metadata.h5`, keeping only the given rows of every dataset.

    Every dataset (`frame_ids`, `is_kinematic`, `timestamps`) has one
    variable-length-bytes row per real frame; the same `real_indices` are kept
    so the metadata still lines up one-to-one with the repacked frames.

    Parameters
    ----------
    src_h5 : str
        Source metadata h5.
    dst_h5 : str
        Destination metadata h5.
    real_indices : list of int
        Row indices to keep.
    """

    idx = np.asarray(real_indices)
    with h5py.File(src_h5, 'r') as fin, h5py.File(dst_h5, 'w') as fout:
        for key in fin.keys():
            dset = fin[key]
            fout.create_dataset(key, data=dset[:][idx], dtype=dset.dtype)


def main() -> None:
    """
    Generate the fixture tree for every subfolder in `SUBFOLDERS`.
    """

    for sub in SUBFOLDERS:
        src_dir = os.path.join(SRC_ROOT, sub)
        dst_dir = os.path.join(DST_ROOT, sub)
        os.makedirs(dst_dir, exist_ok=True)

        mkv_src = _find_one(src_dir, '_raw_tiff.mkv')
        meta_src = _find_one(src_dir, '_tif_metadata.h5')

        # number of real frames = number of metadata rows; pick N_KEEP evenly
        # interspaced indices across the whole recording (0, stride, 2*stride...)
        with h5py.File(meta_src, 'r') as f:
            n_real = f['timestamps'].shape[0]
        stride = n_real // N_KEEP
        real_indices = [i * stride for i in range(N_KEEP)]
        print(f'{sub}: {n_real} real frames -> keeping {real_indices}')

        # repacked .mkv (same filename), subsampled metadata (same filename)
        frames = _extract_frames(mkv_src, real_indices)
        _write_packed_mkv(frames, os.path.join(dst_dir, os.path.basename(mkv_src)))
        _subsample_metadata(meta_src, os.path.join(dst_dir,
                                                   os.path.basename(meta_src)),
                            real_indices)

        # config copied verbatim
        shutil.copy2(os.path.join(src_dir, CONFIG_NAME),
                     os.path.join(dst_dir, CONFIG_NAME))
        print(f'{sub}: wrote fixture to {dst_dir}')


if __name__ == '__main__':
    main()
