#!/usr/bin/env python3
"""
Watch a SCAPE experiment folder on the scapepc and MOVE each finished
recording to the server, so the scapepc disk doesn't fill up during long
(multi-day) acquisitions.

Source layout (as written by the SCAPE / Andor software), e.g.

    /mnt/scapepc/MA_scape_data/20261021_xxx/
        sample1_run1/                    <- .dat spool files, acquisitionmetadata.ini,
                                            Spooled files.sifx, ModifiedData/
        sample1_run1_info.mat
        sample1_run1_stim_data.bin
        sample1_run2/
        ...

Destination layout (same as the rest of /mnt/upramdya_data/MA/SCAPE), e.g.

    /mnt/upramdya_data/MA/SCAPE/20261021_xxx/
        sample1_run01_20261021_1641/
            sample1_run01_20261021_1641/     <- everything that was inside the source run folder
            sample1_run01_20261021_1641_info.mat
            sample1_run01_20261021_1641_stim_data.bin
        sample1_run02_short_20261021_1741/
        ...

A recording counts as finished when its `_info.mat` and `Spooled files.sifx`
both exist (these are written last, at the very end of the acquisition)
and none of its files have changed for `--settle` seconds. This works for any
recording length.

Each file is copied to a temporary `.partial` name, its size (and optionally
md5 checksum) is checked against the source, it is renamed to the final name,
and only then is the source file deleted. If the script is interrupted it can
simply be restarted: files already present at the destination with the right
size are not copied again.

Runs are renumbered on the server. The SCAPE software numbers runs from the
folders it still sees on the scapepc, so once runs are moved away it reuses
names (e.g. many `sample1_run1`). Each run is therefore saved as the next free
zero-padded number for its prefix in the server experiment folder, in the
order the recordings finish, keeping any suffix, followed by the recording
start time `_YYYYMMDD_HHMM`: `sample1_run1_short` ->
`sample1_run01_short_20261021_1630`, the next `sample1_run1` ->
`sample1_run02_20261021_1641`, etc. The start time is `scanStartTimeApprox`
from `_info.mat` (needs scipy), or else the creation time of
`acquisitionmetadata.ini`. The scapepc -> server name mapping is written to
the log.

A `.transfer_source` marker in the server run folder (written at the start of
a transfer, removed once the scapepc run folder is gone) identifies which
recording it holds, so an interrupted transfer resumes into the same folder,
and files that Solis writes into a run folder after it was moved (e.g.
`ModifiedData/*locations.dat` when a recording is opened in Solis) are added
to the right server folder.

Run it in a `tmux`/`screen` session so it survives closing the terminal:

    python transfer_scape_recordings.py /mnt/scapepc/MA_scape_data/20261021_V7xPR_pupa_long

(give the full path to get tab completion, just the folder name, or nothing
to choose the folder from a numbered list of experiments on the scapepc).

Stop it with Ctrl+C (an interrupted transfer resumes on the next start).
"""

import argparse
import hashlib
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

SRC_ROOT = Path('/mnt/scapepc/MA_scape_data')
DST_ROOT = Path('/mnt/upramdya_data/MA/SCAPE')
# _info.mat last: its presence marks a run as finished on the scapepc
SIDECAR_SUFFIXES = ['_stim_data.bin', '_info.mat']
MARKER = '.transfer_source'
# sample1_run1_short -> ('sample1_run', '1', '_short')
RUN_RE = re.compile(r'^(?P<prefix>.*run)(?P<num>\d+)(?P<suffix>.*)$')
END_MARKER = 'Spooled files.sifx'
LOG_NAME = 'transfer_log.txt'

_log_file: Optional[Path] = None
_status_shown = False  # True while the one-line watching status is on screen


def log(msg: str) -> None:
    """
    Print `msg` with a timestamp and append it to the log file, if set.
    """
    global _status_shown
    line = f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}'
    if _status_shown:
        print()
        _status_shown = False
    print(line, flush=True)
    if _log_file is not None:
        try:
            with open(_log_file, 'a') as f:
                f.write(line + '\n')
        except OSError:
            pass


def natural_key(name: str) -> list:
    """
    Sort key so that 'sample1_run10' comes after 'sample1_run9'.
    """
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', name)]


def find_experiment(src_root: Path,
                    experiment: Optional[str]) -> Path:
    """
    Resolve the experiment folder to watch.

    Parameters
    ----------
    src_root : Path
        Folder on the scapepc containing the dated experiment folders.
    experiment : str or None
        Experiment folder name (or full path). If None, the dated folders in
        `src_root` (names starting with yyyymmdd) are listed, newest first,
        and the user picks one by number (Enter picks the newest).

    Returns
    -------
    Path
        The experiment folder on the scapepc.
    """
    if experiment is not None:
        path = Path(experiment)
        if not path.is_absolute():
            path = src_root / experiment
        if not path.is_dir():
            sys.exit(f'Experiment folder not found: {path}')
        return path
    candidates = sorted([p for p in src_root.iterdir()
                         if p.is_dir() and re.match(r'^\d{8}', p.name)],
                        key=lambda p: p.name, reverse=True)
    if not candidates:
        sys.exit(f'No dated experiment folders found in {src_root}')
    print(f'Experiment folders in {src_root}:')
    for i, p in enumerate(candidates[:20], 1):
        print(f'  {i:2d}  {p.name}')
    while True:
        choice = input('Select folder number [1]: ').strip() or '1'
        if choice.isdigit() and 1 <= int(choice) <= min(len(candidates), 20):
            return candidates[int(choice) - 1]
        print('  invalid choice')


def run_files(src_exp: Path,
              run: str) -> List[Path]:
    """
    List every file belonging to a run: everything inside the run folder plus
    the sidecar `_info.mat` / `_stim_data.bin` files next to it.
    """
    files = [p for p in (src_exp / run).rglob('*') if p.is_file()]
    files += [src_exp / (run + s) for s in SIDECAR_SUFFIXES
              if (src_exp / (run + s)).is_file()]
    return files


def snapshot(files: List[Path]) -> tuple:
    """
    Summarise the state of `files` as (n_files, total_size, newest_mtime).
    """
    total, newest = 0, 0.0
    for p in files:
        try:
            st = p.stat()
        except FileNotFoundError:
            continue
        total += st.st_size
        newest = max(newest, st.st_mtime)
    return len(files), total, newest


def md5(path: Path,
        chunk: int = 8 * 1024 * 1024) -> str:
    """
    Return the md5 hex digest of `path`.
    """
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def move_file(src: Path,
              dst: Path,
              checksum: bool,
              dry_run: bool) -> int:
    """
    Copy `src` to `dst` via a temporary `.partial` file, verify it, then
    delete `src`.

    Parameters
    ----------
    src, dst : Path
        Source and destination file paths.
    checksum : bool
        If True, also compare md5 checksums (slower, reads both files again).
    dry_run : bool
        If True, only report what would be done.

    Returns
    -------
    int
        Number of bytes copied (0 if the file was already at the destination).
    """
    size = src.stat().st_size
    if dry_run:
        print(f'    would move {src} -> {dst}')
        return size
    copied = 0
    if not (dst.is_file() and dst.stat().st_size == size):
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + '.partial')
        shutil.copyfile(src, tmp)
        try:  # keep original modification times
            shutil.copystat(src, tmp)
        except OSError:
            pass
        if tmp.stat().st_size != size:
            tmp.unlink()
            raise IOError(f'Size mismatch after copying {src}')
        os.replace(tmp, dst)
        copied = size
    if checksum and md5(src) != md5(dst):
        dst.unlink()
        raise IOError(f'Checksum mismatch for {src}')
    src.unlink()
    return copied


def remove_empty_dirs(folder: Path) -> None:
    """
    Remove `folder` and all its subfolders, provided they contain no files.
    """
    if not folder.is_dir():
        return
    for d in sorted([p for p in folder.rglob('*') if p.is_dir()] + [folder],
                    key=lambda p: len(p.parts), reverse=True):
        try:
            d.rmdir()
        except OSError:
            pass


def recording_id(src_exp: Path,
                 run: str) -> str:
    """
    Identify a recording by the size and modification time of its `_info.mat`.
    """
    st = (src_exp / (run + '_info.mat')).stat()
    return f'{run} {st.st_size} {int(st.st_mtime)}'


def marker_dirs(dst_exp: Path) -> dict:
    """
    Map each server run folder holding a `.transfer_source` marker to the
    recording id stored in it.
    """
    out = {}
    for d in dst_exp.iterdir():
        marker = d / MARKER
        if d.is_dir() and marker.is_file():
            out[d.name] = marker.read_text().strip()
    return out


def start_time(src_exp: Path,
               run: str) -> datetime:
    """
    Return the start time of a recording: `scanStartTimeApprox` from its
    `_info.mat` if scipy can read it, otherwise the modification time of
    `acquisitionmetadata.ini` (written when the acquisition starts).
    """
    try:
        from scipy.io import loadmat
        info = loadmat(src_exp / (run + '_info.mat'), squeeze_me=True,
                       struct_as_record=False,
                       variable_names=['info'])['info']
        return datetime.strptime(str(info.scanStartTimeApprox),
                                 '%d-%b-%Y %H:%M:%S')
    except Exception:
        ini = src_exp / run / 'acquisitionmetadata.ini'
        if ini.is_file():
            return datetime.fromtimestamp(ini.stat().st_mtime)
        return datetime.fromtimestamp(
            (src_exp / (run + '_info.mat')).stat().st_mtime)


def destination_name(src_exp: Path,
                     dst_exp: Path,
                     run: str) -> str:
    """
    Choose the server name for `run`: the folder already holding an
    interrupted transfer of this same recording if there is one, otherwise the
    next free number for its prefix, zero-padded to 2 digits, keeping any
    suffix and adding the start time (e.g. `sample1_run1_short` ->
    `sample1_run03_short_20261021_1630` if `sample1_run01...` and
    `sample1_run02...` exist).
    """
    rec_id = recording_id(src_exp, run)
    for name, marker_id in marker_dirs(dst_exp).items():
        if marker_id == rec_id:
            return name
    stamp = f'{start_time(src_exp, run):%Y%m%d_%H%M}'
    m = RUN_RE.match(run)
    if m is None:
        # Not a '...run<N>...' name: keep it, add the start time
        name, i = f'{run}_{stamp}', 2
        while (dst_exp / name).exists():
            name, i = f'{run}_{stamp}_{i}', i + 1
        return name
    prefix, suffix = m['prefix'], m['suffix'] + '_' + stamp
    pattern = re.compile(rf'^{re.escape(prefix)}(\d{{2,}})(?!\d)')
    numbers = [int(mm[1]) for d in dst_exp.iterdir()
               if d.is_dir() and (mm := pattern.match(d.name))]
    n = max(numbers, default=0) + 1
    while (dst_exp / f'{prefix}{n:02d}{suffix}').exists():
        n += 1
    return f'{prefix}{n:02d}{suffix}'


def move_leftovers(src_exp: Path,
                   dst_exp: Path,
                   run: str,
                   name: str,
                   dry_run: bool) -> None:
    """
    Move files that appeared in the scapepc folder of an already transferred
    run (e.g. Solis' `ModifiedData/<run>locations.dat`) into its server
    folder `name`, then clear the marker once the scapepc folder is gone.
    """
    src_run = src_exp / run
    dst_data = dst_exp / name / name
    files = [p for p in src_run.rglob('*') if p.is_file()]
    log(f'Moving {len(files)} late file(s) of {run} -> {name}: '
        + ', '.join(str(p.relative_to(src_run)) for p in files))
    for p in files:
        move_file(p, dst_data / p.relative_to(src_run), False, dry_run)
    if not dry_run:
        remove_empty_dirs(src_run)
        if not src_run.exists():
            (dst_exp / name / MARKER).unlink(missing_ok=True)


def transfer_run(src_exp: Path,
                 dst_exp: Path,
                 run: str,
                 workers: int,
                 checksum: bool,
                 dry_run: bool) -> None:
    """
    Move one finished run from the scapepc to the server, reorganising it as
    `dst_exp/name/name/<raw files>` plus `dst_exp/name/name_info.mat` and
    `dst_exp/name/name_stim_data.bin`, where `name` is the renumbered run name
    (see `destination_name()`).
    """
    src_run = src_exp / run
    name = destination_name(src_exp, dst_exp, run)
    dst_run = dst_exp / name
    dst_data = dst_run / name
    if not dry_run:
        dst_run.mkdir(parents=True, exist_ok=True)
        (dst_run / MARKER).write_text(recording_id(src_exp, run) + '\n')

    jobs = []
    for p in src_run.rglob('*'):
        if p.is_file():
            jobs.append((p, dst_data / p.relative_to(src_run)))
        elif p.is_dir() and not dry_run:
            (dst_data / p.relative_to(src_run)).mkdir(parents=True, exist_ok=True)
    # Sidecars last, so a half-transferred run is easy to recognise on the
    # server (it has no _info.mat yet)
    sidecars = [(src_exp / (run + s), dst_run / (name + s))
                for s in SIDECAR_SUFFIXES if (src_exp / (run + s)).is_file()]

    n_total = len(jobs) + len(sidecars)
    total_bytes = sum(s.stat().st_size for s, _ in jobs + sidecars)
    log(f'Moving {run} -> {name}: {n_total} files, {total_bytes / 1e9:.1f} GB '
        f'-> {dst_run}')
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        copied = sum(pool.map(lambda j: move_file(*j, checksum, dry_run), jobs))
    for s, d in sidecars:
        copied += move_file(s, d, checksum, dry_run)

    if not dry_run:
        # Solis normally creates an empty ModifiedData folder; keep the layout
        # consistent even if it hadn't been created yet
        (dst_data / 'ModifiedData').mkdir(parents=True, exist_ok=True)
        remove_empty_dirs(src_run)
        if src_run.exists():
            log(f'  WARNING: {src_run} not empty after transfer, the '
                f'remaining files will be added to {name} later')
        else:
            (dst_run / MARKER).unlink(missing_ok=True)

    dt = time.time() - t0
    log(f'Done {run} -> {name}: {copied / 1e9:.1f} GB in {dt / 60:.1f} min '
        f'({copied / 1e6 / max(dt, 1e-6):.0f} MB/s)')


def watch(src_exp: Path,
          dst_exp: Path,
          settle: float,
          poll: float,
          workers: int,
          checksum: bool,
          once: bool,
          dry_run: bool) -> None:
    """
    Poll `src_exp` and move every finished run to `dst_exp`, forever (or
    until no finished runs are left, if `once`).

    Parameters
    ----------
    src_exp, dst_exp : Path
        Experiment folder on the scapepc and on the server.
    settle : float
        Seconds without any file change before a finished run is moved.
    poll : float
        Seconds between checks of `src_exp`.
    workers : int
        Number of files copied in parallel.
    checksum : bool
        Verify copies with md5 checksums in addition to file sizes.
    once : bool
        Transfer whatever is finished now, then exit.
    dry_run : bool
        Only report what would be moved.
    """
    last_state = {}   # run -> snapshot from previous poll
    waiting = set()   # runs already reported as 'recording in progress'
    while True:
        runs = sorted([p.name for p in src_exp.iterdir() if p.is_dir()],
                      key=natural_key)
        # Server folders whose scapepc run folder still has leftover files
        moved = {}  # scapepc run name -> server name
        for name, marker_id in marker_dirs(dst_exp).items():
            src_name = marker_id.split()[0]
            if (src_exp / src_name).is_dir():
                moved[src_name] = name
            elif not dry_run:  # scapepc folder gone: transfer is complete
                (dst_exp / name / MARKER).unlink(missing_ok=True)

        for run in runs:
            src_run = src_exp / run
            has_end = ((src_exp / (run + '_info.mat')).is_file()
                       and (src_run / END_MARKER).is_file())
            files = run_files(src_exp, run)

            if not files:
                # Leftover empty folders (e.g. ModifiedData recreated by
                # Solis after a run was already moved)
                if (time.time() - src_run.stat().st_mtime > settle
                        and not dry_run):
                    remove_empty_dirs(src_run)
                continue
            # Files written after the run was moved: no sidecars and no raw
            # data (so not a new recording that reuses the name)
            if (run in moved
                    and not (src_exp / (run + '_info.mat')).exists()
                    and not (src_run / 'acquisitionmetadata.ini').exists()
                    and not any(src_run.glob('*spool.dat'))
                    and time.time() - snapshot(files)[2] > settle):
                try:
                    move_leftovers(src_exp, dst_exp, run, moved[run], dry_run)
                except Exception as e:
                    log(f'ERROR moving late files of {run}: {e!r} (will retry)')
                continue
            if not has_end:
                if run not in waiting:
                    log(f'{run}: recording in progress, waiting')
                    waiting.add(run)
                continue

            state = snapshot(files)
            stable = (last_state.get(run) == state
                      and time.time() - state[2] > settle)
            last_state[run] = state
            if not stable:
                continue

            try:
                transfer_run(src_exp, dst_exp, run, workers, checksum, dry_run)
            except Exception as e:  # keep watching, retry on next poll
                log(f'ERROR moving {run}: {e!r} (will retry)')
            last_state.pop(run, None)
            waiting.discard(run)

        if once and not any(r in last_state for r in runs):
            return
        try:
            free = shutil.disk_usage(src_exp).free / 1e9
            status = f'{free:.0f} GB free on scapepc'
        except OSError:
            status = ''
        global _status_shown
        print(f'\r[{datetime.now():%H:%M:%S}] watching {src_exp.name}  '
              f'{status}   ', end='', flush=True)
        _status_shown = True
        time.sleep(poll)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Move finished SCAPE recordings from the scapepc to the '
                    'server as they are acquired.')
    parser.add_argument('experiment', nargs='?', default=None,
                        help='Experiment folder name in --src-root (e.g. '
                             '20261021_V7xPR_pupa_long) or full path (use '
                             'the full path to get tab completion). If '
                             'omitted, choose from a numbered list.')
    parser.add_argument('--src-root', type=Path, default=SRC_ROOT,
                        help=f'Default: {SRC_ROOT}')
    parser.add_argument('--dst-root', type=Path, default=DST_ROOT,
                        help=f'Default: {DST_ROOT}')
    parser.add_argument('--settle', type=float, default=300,
                        help='Seconds without file changes before a finished '
                             'run is moved (default: 300)')
    parser.add_argument('--poll', type=float, default=60,
                        help='Seconds between checks (default: 60)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Files copied in parallel (default: 4)')
    parser.add_argument('--checksum', action='store_true',
                        help='Also verify md5 checksums (slower)')
    parser.add_argument('--once', action='store_true',
                        help='Move runs that are already finished, then exit')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only print what would be moved')
    args = parser.parse_args()

    global _log_file
    src_exp = find_experiment(args.src_root, args.experiment)
    dst_exp = args.dst_root / src_exp.name
    if not args.dry_run:
        dst_exp.mkdir(parents=True, exist_ok=True)
        _log_file = dst_exp / LOG_NAME
    log(f'Watching {src_exp}\n    moving finished runs to {dst_exp}\n'
        f'    settle={args.settle:.0f}s poll={args.poll:.0f}s '
        f'workers={args.workers} checksum={args.checksum}')
    try:
        watch(src_exp, dst_exp, args.settle, args.poll, args.workers,
              args.checksum, args.once, args.dry_run)
    except KeyboardInterrupt:
        log('Stopped by user')


if __name__ == '__main__':
    main()
