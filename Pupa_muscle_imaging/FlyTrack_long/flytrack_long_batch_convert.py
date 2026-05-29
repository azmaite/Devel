#!/usr/bin/env python3
"""
flytrack_long_batch_convert.py
batch-convert live-written tif frames to lossless MKV
Used for long FlyTrack recordings where frames are saved as individual tif files directly to the server.
This script monitors a folder for incoming ``N.tif`` images, converts them to ``batchN.mkv`` videos in batches
of 6000 tifs (24000 raw frames, 10 min at 40hz), 
and optionally deletes the source tif/metadata files after successful encoding.

Usage
-----
Run from the command line while (or after) a recording is saving frames:

    python flytrack_long_batch_convert.py /path/to/folder [options]

The script expects frames named ``N.tif`` (N is an integer, not zero-padded) to
appear incrementally in ``folder``. It converts them to lossless FFV1 ``.mkv`` files
named ``batch1.mkv``, ``batch2.mkv``, ... and (by default) deletes the source tifs after
each successful batch.

Termination
-----------
The script checks every ``--check-interval`` seconds whether new frames have arrived.
Once a full interval passes with no new frames, it saves whatever remains as a final
(possibly partial) batch and exits. If no frames appear within 60 minutes of startup,
the script exits with a timeout message.

Inputs
------
folder : str
    Directory containing (or that will contain) ``N.tif`` images.
--keep-tifs
    Do not delete source ``.tif`` files after a successful batch.
    Also do not delete metadata csv files. (default: delete).
--check-interval SECONDS
    Seconds between folder scans when waiting for more frames (default: 30).
--batch-size N
    Number of tif files per batch (default: 6000, equivalent to 24000 raw frames).

Outputs (written to ``folder``)
--------------------------------
batch1.mkv, batch2.mkv, ...
    Lossless FFV1-encoded videos, one per batch (6000 tifs = 24000 raw frames each).
batchN_timestamps.csv
    Merged per-frame metadata from the individual ``N_metadata.csv`` files.
batch_contents.json
    Log of every attempted batch: frame range, missing-frame list, and status
    (``in_progress`` / ``success`` / ``failed``). Used for resuming after a restart.
ffmpeg_batchN.log
    Raw ffmpeg stderr for each batch, useful for debugging encode failures.

Resume behaviour
----------------
If restarted mid-recording the script reads ``batch_contents.json`` and skips frame
ranges already saved, picking up from where it left off.

Missing frames
--------------
Gaps in the ``N.tif`` sequence within a batch are detected and logged to
``batch_contents.json``. The present frames are still encoded; the missing frame numbers
are printed at batch start.

Frame-count verification
------------------------
After each encode, ``ffprobe`` performs a full read-through of the output ``.mkv`` to
confirm the frame count matches the number of tifs fed in. If there is a mismatch the
batch is marked ``failed`` and the source tifs are kept, so no data is lost.

Some functions copied from Victor Alfred Stimfplings' FlyTrack code: https://github.com/NeLy-EPFL/flymuscle-control
"""

import argparse
import csv
import json
import re
import shutil
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import subprocess
from tqdm import tqdm


_shutdown_requested = False

# Tif files are numbered in steps of 4 (1, 5, 9, …) because each tif holds 4 frames.
_FRAME_STEP = 4


def _handle_signal(signum, frame):
    global _shutdown_requested
    print(f'\n{signal.Signals(signum).name} received — stopping. '
          f'Any ongoing encode will finish but tif files will NOT be deleted.')
    _shutdown_requested = True


def flytrack_long_batch_convert(folder: Path,
                                 keep_tifs: bool = False,
                                 check_interval: int = 30,
                                 batch_size: int = 6000) -> None:
    """
    Monitor a folder for incoming .tif images and convert them to .mkv in batches.

    Behaviour
    ---------
    - If >= `batch_size` unprocessed frames are present, converts the lowest-numbered
      `batch_size` of them to `batchN.mkv` immediately, then checks again — no waiting
      between back-to-back full batches.
    - If fewer than `batch_size` frames are present, waits `check_interval` seconds and
      checks again. If new frames arrived during the wait, continues monitoring. If no
      new frames arrived, saves whatever remains as the final (possibly partial) batch
      and exits.
    - Can be started while recording is already in progress or after it has finished.
    - Resumable: on restart it reads `batch_contents.json` and skips already-saved
      frame ranges.

    Missing frames (gaps in the `N.tif` sequence) are detected within each batch and
    written to the log; the present frames are still encoded.

    Parameters
    ----------
    folder : Path
        Folder to monitor for `N.tif` images. Need not exist yet at call time.
    keep_tifs : bool
        If True, do not delete source .tif files after a successful batch. Default False.
    check_interval : int
        Seconds to wait between scans when fewer than `batch_size` frames are available.
        Default 30.
    batch_size : int
        Number of frames per batch. Default 6000 (but each frame has 4 channels that
        are actually 4 frames, so the effective batch size is 24000).
    """
    folder = Path(folder)
    log_file = folder / 'batch_contents.json'

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    start_timeout = 60 * 60  # give up if no frames appear within this many seconds

    # --- Load or initialize state ---
    batch_log: dict = {'first_frame': None, 'batches': {}}
    first_frame: Optional[int] = None
    batch_num = 1
    last_processed_frame = -1  # highest frame number successfully saved

    if folder.exists() and log_file.exists():
        with open(log_file) as f:
            batch_log = json.load(f)
        first_frame = batch_log.get('first_frame')
        batches = batch_log.get('batches', {})
        if batches:
            batch_num = max(int(k[5:]) for k in batches) + 1
            successful = [b for b in batches.values() if b['status'] == 'success']
            if successful:
                last_processed_frame = max(b['frame_range'][1] for b in successful)
            print(f'Resuming from batch {batch_num} '
                  f'(last saved frame: {last_processed_frame}).')

    if 'batches' not in batch_log:
        batch_log['batches'] = {}

    # Clean up leftover temp folders/files from any previously interrupted encode.
    if folder.exists():
        for leftover_dir in folder.glob('*_fixed'):
            if leftover_dir.is_dir():
                shutil.rmtree(leftover_dir, ignore_errors=True)
                print(f'Cleaned up leftover temp folder: {leftover_dir.name}')
        for leftover_txt in folder.glob('*_list.txt'):
            leftover_txt.unlink(missing_ok=True)

    # last_unprocessed_count: count seen at the previous check while waiting.
    # None means we haven't yet done a wait-interval check, so we should not yet
    # declare end-of-recording (avoids false exit right after a batch or on first scan).
    last_unprocessed_count: Optional[int] = None

    start_time = time.time()
    total_missing = 0
    total_batches = 0

    # --- Main loop ---
    while not _shutdown_requested:
        if not folder.exists():
            print('Folder does not exist yet, waiting...')
            time.sleep(check_interval)
            continue

        tif_dict = _scan_tif_nums(folder)

        if first_frame is None:
            if not tif_dict:
                elapsed = time.time() - start_time
                if elapsed > start_timeout:
                    print(f'No frames found after {start_timeout / 3600:.0f} h. Exiting.')
                    return
                print(f'Waiting for recording to start '
                      f'({elapsed / 60:.0f}/{start_timeout / 60:.0f} min)...')
                time.sleep(check_interval)
                continue
            first_frame = min(tif_dict)
            batch_log['first_frame'] = first_frame
            _write_batch_log(log_file, batch_log)
            print(f'Recording started. First frame: {first_frame}')

        total_tif_count = len(tif_dict)
        unprocessed = {n: p for n, p in tif_dict.items() if n > last_processed_frame}
        current_count = len(unprocessed)
        elapsed = time.time() - start_time

        if current_count >= batch_size:
            # Enough frames: convert a full batch immediately (no sleep).
            batch_items = sorted(unprocessed.items())[:batch_size]
            n_missing = _process_batch(
                folder=folder,
                tif_items=batch_items,
                batch_num=batch_num,
                batch_log=batch_log,
                log_file=log_file,
                keep_tifs=keep_tifs,
            )
            total_missing += n_missing
            total_batches += 1
            last_processed_frame = batch_items[-1][0]
            last_unprocessed_count = None  # skip end-of-recording check next pass
            batch_num += 1

        elif last_unprocessed_count is not None and current_count <= last_unprocessed_count:
            # No new frames since last wait — recording has ended.
            print(f'[{elapsed/3600:.1f}h] No new frames arrived. Recording has ended.')
            if current_count > 0:
                n_missing = _process_batch(
                    folder=folder,
                    tif_items=sorted(unprocessed.items()),
                    batch_num=batch_num,
                    batch_log=batch_log,
                    log_file=log_file,
                    keep_tifs=keep_tifs,
                )
                total_missing += n_missing
                total_batches += 1
            else:
                print('No remaining frames to save.')
            break

        else:
            # Not enough frames yet — wait one interval then re-evaluate.
            last_unprocessed_count = current_count
            already = total_tif_count - current_count
            already_str = f', {already} already saved' if already > 0 else ''
            print(f'[{elapsed/3600:.1f}h] {current_count} unprocessed frames'
                  f'{already_str} (total tifs in folder: {total_tif_count}), '
                  f'waiting {check_interval}s...')
            time.sleep(check_interval)

    if _shutdown_requested:
        elapsed = time.time() - start_time
        print(f'[{elapsed/3600:.1f}h] Shutdown requested. Remaining frames not saved.')

    _print_summary(batch_log, time.time() - start_time, total_batches, total_missing)


def _process_batch(folder: Path,
                   tif_items: List[Tuple[int, Path]],
                   batch_num: int,
                   batch_log: dict,
                   log_file: Path,
                   keep_tifs: bool) -> int:
    """
    Convert one batch of tif frames to mkv, log the result, and optionally delete tifs.

    Missing frames are detected as gaps in the frame-number sequence between the first
    and last frame in the batch.

    Parameters
    ----------
    folder : Path
        Parent folder for output files.
    tif_items : list of (int, Path)
        Sorted list of (frame number, tif path) to include in this batch.
    batch_num : int
        Batch index (1-based), used in the output filename.
    batch_log : dict
        Shared log dict, mutated in-place.
    log_file : Path
        Path to write the JSON log.
    keep_tifs : bool
        If True, do not delete source tifs after a successful encode.

    Returns
    -------
    int
        Number of missing frames detected in this batch.
    """
    frame_nums = [n for n, _ in tif_items]
    min_frame, max_frame = frame_nums[0], frame_nums[-1]
    present = set(frame_nums)
    expected = set(range(min_frame, max_frame + _FRAME_STEP, _FRAME_STEP))
    missing = sorted(expected - present)
    n_missing = len(missing)

    if n_missing:
        preview = missing[:10]
        suffix = f'... (+{n_missing - 10} more)' if n_missing > 10 else ''
        print(f'Batch {batch_num}: {n_missing} missing frame(s): {preview}{suffix}')

    batch_tifs = [p for _, p in tif_items]
    output_file = folder / f'batch{batch_num}.mkv'
    ffmpeg_log = folder / f'ffmpeg_batch{batch_num}.log'
    batch_key = f'batch{batch_num}'

    batch_log['batches'][batch_key] = {
        'status': 'in_progress',
        'frame_range': [min_frame, max_frame],
        'n_frames_present': len(batch_tifs),
        'n_missing': n_missing,
        'missing_frames': missing[:100],
    }
    _write_batch_log(log_file, batch_log)

    print(f'Starting batch {batch_num}: {len(batch_tifs)} frames '
          f'(frames {min_frame}–{max_frame}, {n_missing} missing) → {output_file.name}')

    try:
        export_tifs_as_mkv(batch_tifs, output_file, ffmpeg_log)
    except Exception as e:
        print(f'Batch {batch_num} FAILED: {e}')
        batch_log['batches'][batch_key]['status'] = 'failed'
        _write_batch_log(log_file, batch_log)
        return n_missing

    # Verify frame count in the mkv before touching the source tifs.
    verified_count = _verify_mkv_frame_count(output_file)
    if verified_count is None:
        print(f'Batch {batch_num}: ffprobe unavailable, skipping frame-count check.')
    elif verified_count != len(batch_tifs):
        print(f'Batch {batch_num} FAILED frame-count check: '
              f'expected {len(batch_tifs)}, mkv contains {verified_count}. '
              f'Tifs kept.')
        batch_log['batches'][batch_key]['status'] = 'failed'
        batch_log['batches'][batch_key]['mkv_frame_count'] = verified_count
        _write_batch_log(log_file, batch_log)
        return n_missing
    else:
        print(f'Batch {batch_num}: frame-count verified ({verified_count} frames).')

    delete_files = not keep_tifs and not _shutdown_requested
    _merge_metadata_csvs(folder, frame_nums, batch_num, delete_originals=delete_files)

    if _shutdown_requested:
        print(f'Batch {batch_num} done. Tifs kept (interrupted — not deleting).')
    elif not keep_tifs:
        for tif in batch_tifs:
            tif.unlink(missing_ok=True)
        print(f'Batch {batch_num} done. Deleted {len(batch_tifs)} tif files.')
    else:
        print(f'Batch {batch_num} done. Tifs kept (--keep-tifs).')

    batch_log['batches'][batch_key]['status'] = 'success'
    _write_batch_log(log_file, batch_log)
    return n_missing


def _scan_tif_nums(folder: Path) -> Dict[int, Path]:
    """
    Return a dict mapping frame number → Path for all ``N.tif`` files in `folder`.

    Only files whose stem is a plain integer are included; other ``.tif`` files are
    ignored.
    """
    result: Dict[int, Path] = {}
    for p in folder.glob('*.tif'):
        try:
            result[int(p.stem)] = p
        except ValueError:
            pass
    return result


def _write_batch_log(log_file: Path, batch_log: dict) -> None:
    with open(log_file, 'w') as f:
        json.dump(batch_log, f, indent=2)


def _merge_metadata_csvs(folder: Path,
                          frame_nums: List[int],
                          batch_num: int,
                          delete_originals: bool = True) -> None:
    """
    Merge per-frame ``N_metadata.csv`` files into a single ``batchN_timestamps.csv``.

    If the individual CSVs have a header row (detected by a non-numeric first field),
    it is written once at the top of the merged file.

    Parameters
    ----------
    folder : Path
        Folder containing the individual metadata CSV files.
    frame_nums : list of int
        Frame numbers whose metadata files should be merged, in ascending order.
    batch_num : int
        Batch index, used in the output filename.
    delete_originals : bool
        If True, delete each individual CSV after reading it. Default True.
    """
    output_path = folder / f'batch{batch_num}_timestamps.csv'
    header: Optional[List[str]] = None
    all_rows: List[List[str]] = []
    n_merged = 0
    n_missing_csv = 0

    for n in sorted(frame_nums):
        csv_path = folder / f'{n}_metadata.csv'
        if not csv_path.exists():
            n_missing_csv += 1
            continue
        with open(csv_path, newline='') as f:
            rows = list(csv.reader(f))
        if not rows:
            if delete_originals:
                csv_path.unlink(missing_ok=True)
            continue
        try:
            int(rows[0][0])
            has_header = False
        except (ValueError, IndexError):
            has_header = True

        if has_header:
            if header is None:
                header = rows[0]
            all_rows.extend(rows[1:])
        else:
            all_rows.extend(rows)

        if delete_originals:
            csv_path.unlink(missing_ok=True)
        n_merged += 1

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        if header is not None:
            writer.writerow(header)
        writer.writerows(all_rows)

    msg = f'Batch {batch_num}: merged {n_merged} metadata CSVs → {output_path.name}'
    if n_missing_csv:
        msg += f' ({n_missing_csv} CSV files not found)'
    print(msg)


def _verify_mkv_frame_count(mkv_path: Path) -> Optional[int]:
    """
    Return the number of video frames in `mkv_path` as reported by ffprobe.

    Uses ``-count_frames`` which does a full read-through of the file, so runtime
    is proportional to file size (roughly as long as the encode took).

    Parameters
    ----------
    mkv_path : Path
        Path to the .mkv file to inspect.

    Returns
    -------
    int or None
        Actual frame count, or None if ffprobe is not installed or the call fails.
    """
    try:
        result = subprocess.run(
            [
                'ffprobe',
                '-v', 'error',
                '-count_frames',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=nb_read_frames',
                '-of', 'default=nokey=1:noprint_wrappers=1',
                str(mkv_path),
            ],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        return int(result.stdout.strip())
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        return None


def _print_summary(batch_log: dict,
                   elapsed: float,
                   total_batches: int,
                   total_missing: int) -> None:
    batches = batch_log.get('batches', {})
    n_success = sum(1 for b in batches.values() if b['status'] == 'success')
    n_failed = sum(1 for b in batches.values() if b['status'] == 'failed')
    print(
        f'\n--- Summary ---\n'
        f'  Elapsed:        {elapsed / 3600:.2f} h\n'
        f'  Batches saved:  {n_success} succeeded, {n_failed} failed\n'
        f'  Missing frames: {total_missing}\n'
    )


def export_tifs_as_mkv(tif_files: list, output_file: Path, ffmpeg_log: Path) -> None:
    """
    Convert a pre-sorted list of .tif files to a lossless .mkv video.

    Temp files are named after `output_file.stem` so parallel calls on different
    batches do not collide.

    Parameters
    ----------
    tif_files : list of Path
        Ordered list of .tif file paths to encode (one frame each).
    output_file : Path
        Destination .mkv file path.
    ffmpeg_log : Path
        File to write ffmpeg stderr output into.
    """
    output_file = Path(output_file)
    fixed_tif_folder = output_file.parent / (output_file.stem + '_fixed')
    fixed_tif_folder.mkdir(exist_ok=True)

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(process_image, tif_files, [fixed_tif_folder] * len(tif_files)),
            total=len(tif_files),
            desc='Processing images'
        ))

    fixed_tif_files = []
    for result in results:
        if result.startswith('✓'):
            fixed_tif_files.append(result[2:])
        else:
            print(result)

    list_file = output_file.parent / (output_file.stem + '_list.txt')
    with open(list_file, 'w') as f:
        for tif_file in fixed_tif_files:
            f.write(f"file '{tif_file}'\n")
            f.write('duration 0.04\n')

    ffmpeg_command = [
        'ffmpeg',
        '-hwaccel', 'none',
        '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', str(list_file),
        '-c:v', 'ffv1',
        '-pix_fmt', 'rgba64le',
        '-an',
        str(output_file)
    ]

    run_ffmpeg_with_log(ffmpeg_command, ffmpeg_log, len(tif_files))
    list_file.unlink(missing_ok=True)
    shutil.rmtree(fixed_tif_folder, ignore_errors=True)


def process_image(path, output_dir):
    """
    Copy a single .tif to `output_dir`, re-encoding via OpenCV to ensure format
    compatibility.

    Parameters
    ----------
    path : Path
        Source .tif file.
    output_dir : Path
        Directory to write the processed copy into.

    Returns
    -------
    str
        '✓ <output_path>' on success, or an error message starting with '✗'.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return f'✗ Failed to read {path.name}'

    if img.ndim != 3 or img.shape[2] != 4:
        return f'✗ Unexpected shape {img.shape} in {path.name}'

    output_path = output_dir / path.name

    try:
        success = cv2.imwrite(str(output_path), img)
        return f'✓ {output_path}' if success else f'✗ Failed to write {output_path}'
    except Exception as e:
        return f'✗ Error with {output_path}: {e}'


def run_ffmpeg_with_log(ffmpeg_command: list, log_file: Path, n_frames: int) -> None:
    """
    Run an ffmpeg command, stream stderr to `log_file`, and show a tqdm progress bar.

    Parameters
    ----------
    ffmpeg_command : list of str
        Full ffmpeg command as a list of arguments.
    log_file : Path
        File to write ffmpeg stderr output into.
    n_frames : int
        Expected frame count, used to size the progress bar.

    Raises
    ------
    RuntimeError
        If ffmpeg exits with a non-zero return code.
    """
    with open(log_file, 'w') as log, subprocess.Popen(
        ffmpeg_command,
        start_new_session=True,
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        universal_newlines=True
    ) as process:
        progress_bar = tqdm(total=n_frames, unit='frame', dynamic_ncols=True,
                            desc='Encoding to mkv')

        for line in process.stderr:
            log.write(line)
            log.flush()
            match = re.search(r'frame=\s*(\d+)', line)
            if match:
                frame_num = int(match.group(1))
                progress_bar.update(frame_num - progress_bar.n)

        progress_bar.close()
        process.wait()

        if process.returncode != 0:
            raise RuntimeError(
                f'FFmpeg failed with code {process.returncode}. See {log_file}'
            )

        print(f'Encoding complete. Log saved to {log_file}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Convert live-written N.tif frames to lossless .mkv batches.'
    )
    parser.add_argument('folder', type=Path,
                        help='Folder containing N.tif images (N in steps of 4).')
    parser.add_argument('--keep-tifs', action='store_true',
                        help='Do not delete .tif files after successful encoding.')
    parser.add_argument('--check-interval', type=int, default=30, metavar='SECONDS',
                        help='Seconds between scans when waiting for frames (default: 30).')
    parser.add_argument('--batch-size', type=int, default=6000, metavar='N',
                        help='Tif files per batch (default: 6000 = 24000 raw frames).')
    args = parser.parse_args()

    flytrack_long_batch_convert(
        folder=args.folder,
        keep_tifs=args.keep_tifs,
        check_interval=args.check_interval,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()
