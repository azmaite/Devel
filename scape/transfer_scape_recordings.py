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
recording length. If `Spooled files.sifx` is never written (this happens,
e.g. on 2026-10-29) but the `_info.mat` and the raw `.dat` files are there,
the recording is taken as finished once nothing has changed for
`--no-sifx-after` seconds, with a warning in the log. A folder with an
`_info.mat` but no raw data (an acquisition that failed to start) is left
alone.

Runs are renamed ON THE SCAPEPC as soon as they finish, then moved. The SCAPE
software names a new run from the runs it still sees on the scapepc
(apparently number of runs present + 1), so when there is a backlog of runs
waiting to be moved it can pick the name of one that is still there. That
acquisition then fails and its `_info.mat` / `_stim_data.bin` overwrite those
of the older run (this lost 2 recordings and mislabelled 2 others on
2026-10-29). Renaming finished runs right away means no name the software
picks can match an existing run.

The new name is the next free zero-padded number for the run's prefix (among
the server folders and the runs already renamed on the scapepc), in the order
the recordings finish, keeping any suffix, followed by the recording start
time `_YYYYMMDD_HHMM`: `sample1_run1_short` ->
`sample1_run01_short_20261021_1630`, the next `sample1_run1` ->
`sample1_run02_20261021_1641`, etc. The start time is `scanStartTimeApprox`
from `_info.mat` (needs scipy), cross-checked against the creation time of
`acquisitionmetadata.ini`; if they differ by more than 5 min (the `_info.mat`
was overwritten by a later acquisition) the latter is used and a warning is
logged. The scapepc -> server name mapping is written to the log.

To rename a run, a `.transfer_source` marker holding the original and the new
name is written into its folder, then the `_info.mat` / `_stim_data.bin` and
the folder are renamed. Folders holding a marker are the renamed runs waiting
to be moved; the marker is deleted last, once everything else has been moved.
If the script is interrupted it can simply be restarted: an interrupted
rename is completed, and an interrupted transfer resumes.

Each file is copied to a temporary `.partial` name, its size (and optionally
md5 checksum) is checked against the source, it is renamed to the final name,
and only then is the source file deleted. Files already present at the
destination with the right size are not copied again.

Files that Solis writes into a run folder under its original name after the
run was renamed (e.g. `ModifiedData/*locations.dat`, written when the next
acquisition starts or when a recording is opened in Solis) are added to the
right server folder, as long as the renamed run is still on the scapepc.

Run it in a `tmux`/`screen` session so it survives closing the terminal:

    python transfer_scape_recordings.py /mnt/scapepc/MA_scape_data/20261021_V7xPR_pupa_long

(give the full path to get tab completion, just the folder name, or nothing
to choose the folder from a numbered list of experiments on the scapepc).

It stops by itself once nothing has happened for `--timeout` hours (default
3): no run renamed or moved, and no file of a run on the scapepc changed (so
a recording in progress keeps it running). `--timeout 0` watches forever.
Stop it earlier with Ctrl+C (an interrupted transfer resumes on the next
start).
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
from typing import Dict, List, Optional

SRC_ROOT = Path('/mnt/scapepc/MA_scape_data')
DST_ROOT = Path('/mnt/upramdya_data/MA/SCAPE')
# _info.mat last: its presence marks a run as finished on the scapepc
SIDECAR_SUFFIXES = ['_stim_data.bin', '_info.mat']
MARKER = '.transfer_source'
# sample1_run1_short -> ('sample1_run', '1', '_short')
RUN_RE = re.compile(r'^(?P<prefix>.*run)(?P<num>\d+)(?P<suffix>.*)$')
END_MARKER = 'Spooled files.sifx'
LOG_NAME = 'transfer_log.txt'
# Max difference between scanStartTimeApprox and acquisitionmetadata.ini time
MAX_START_DIFF = 300
# _info.mat this much newer than the raw data: probably overwritten later
MAX_INFO_DELAY = 1800

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
    List every file belonging to a run: everything inside the run folder
    (except the `.transfer_source` marker) plus the sidecar `_info.mat` /
    `_stim_data.bin` files next to it.
    """
    files = [p for p in (src_exp / run).rglob('*')
             if p.is_file() and p.name != MARKER]
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


def read_marker(folder: Path) -> Optional[tuple]:
    """
    Return (original name, new name) from the `.transfer_source` marker in
    `folder`, or None if it has no marker.
    """
    try:
        lines = (folder / MARKER).read_text().split()
    except (FileNotFoundError, NotADirectoryError):
        return None
    if len(lines) != 2:
        return None
    return lines[0], lines[1]


def start_time(src_exp: Path,
               run: str) -> datetime:
    """
    Return the start time of a recording: `scanStartTimeApprox` from its
    `_info.mat` if scipy can read it, otherwise the modification time of
    `acquisitionmetadata.ini` (written when the acquisition starts). If both
    exist but differ by more than `MAX_START_DIFF` seconds, the `_info.mat`
    probably belongs to a later acquisition that reused the run name, so the
    `acquisitionmetadata.ini` time is used and a warning is logged.
    """
    ini = src_exp / run / 'acquisitionmetadata.ini'
    ini_time = (datetime.fromtimestamp(ini.stat().st_mtime)
                if ini.is_file() else None)
    try:
        from scipy.io import loadmat
        info = loadmat(src_exp / (run + '_info.mat'), squeeze_me=True,
                       struct_as_record=False,
                       variable_names=['info'])['info']
        info_time = datetime.strptime(str(info.scanStartTimeApprox),
                                      '%d-%b-%Y %H:%M:%S')
    except Exception:
        info_time = None
    if info_time is not None and ini_time is not None:
        if abs((info_time - ini_time).total_seconds()) > MAX_START_DIFF:
            log(f'  WARNING: {run}: _info.mat start time {info_time} does '
                f'not match the raw data ({ini_time:%Y-%m-%d %H:%M:%S}), the '
                f'_info.mat may come from a later acquisition; naming it by '
                f'the raw data time')
            return ini_time
        return info_time
    if info_time is not None:
        return info_time
    if ini_time is not None:
        return ini_time
    return datetime.fromtimestamp(
        (src_exp / (run + '_info.mat')).stat().st_mtime)


def destination_name(src_exp: Path,
                     dst_exp: Path,
                     run: str) -> str:
    """
    Choose the new name for `run`: the next free number for its prefix among
    the server folders and the runs already renamed on the scapepc,
    zero-padded to 2 digits, keeping any suffix and adding the start time
    (e.g. `sample1_run1_short` -> `sample1_run03_short_20261021_1630` if
    `sample1_run01...` and `sample1_run02...` exist).
    """
    stamp = f'{start_time(src_exp, run):%Y%m%d_%H%M}'
    taken = {d.name for d in dst_exp.iterdir() if d.is_dir()}
    taken |= {m[1] for d in src_exp.iterdir()
              if d.is_dir() and (m := read_marker(d))}
    m = RUN_RE.match(run)
    if m is None:
        # Not a '...run<N>...' name: keep it, add the start time
        name, i = f'{run}_{stamp}', 2
        while name in taken:
            name, i = f'{run}_{stamp}_{i}', i + 1
        return name
    prefix, suffix = m['prefix'], m['suffix'] + '_' + stamp
    pattern = re.compile(rf'^{re.escape(prefix)}(\d{{2,}})(?!\d)')
    numbers = [int(mm[1]) for t in taken if (mm := pattern.match(t))]
    n = max(numbers, default=0) + 1
    while f'{prefix}{n:02d}{suffix}' in taken:
        n += 1
    return f'{prefix}{n:02d}{suffix}'


def finish_rename(src_exp: Path,
                  run: str,
                  name: str) -> None:
    """
    Rename the sidecar files and then the folder of `run` to `name` (the
    marker must already be in the folder). Safe to call again after an
    interruption.
    """
    for s in SIDECAR_SUFFIXES:
        if (src_exp / (run + s)).is_file():
            (src_exp / (run + s)).rename(src_exp / (name + s))
    (src_exp / run).rename(src_exp / name)


def rename_run(src_exp: Path,
               dst_exp: Path,
               run: str,
               dry_run: bool) -> str:
    """
    Give finished run `run` its final name on the scapepc (see
    `destination_name()`), so the SCAPE software can't reuse its name while it
    waits to be moved, and return the new name.
    """
    name = destination_name(src_exp, dst_exp, run)
    info_mtime = (src_exp / (run + '_info.mat')).stat().st_mtime
    raw_mtime = snapshot([p for p in (src_exp / run).rglob('*spool.dat')])[2]
    if info_mtime - raw_mtime > MAX_INFO_DELAY:
        log(f'  WARNING: {run}_info.mat was written '
            f'{(info_mtime - raw_mtime) / 60:.0f} min after the last raw data '
            f'file, it may come from a later acquisition that reused the name')
    log(f'Renaming {run} -> {name} on the scapepc')
    if not dry_run:
        (src_exp / run / MARKER).write_text(f'{run}\n{name}\n')
        finish_rename(src_exp, run, name)
    return name


def move_leftovers(src_exp: Path,
                   dst_exp: Path,
                   run: str,
                   name: str,
                   dry_run: bool) -> None:
    """
    Move files that appeared in a scapepc folder with the original name `run`
    of an already renamed run (e.g. Solis' `ModifiedData/<run>locations.dat`)
    into its server folder `name`.
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


def transfer_run(src_exp: Path,
                 dst_exp: Path,
                 src_name: str,
                 name: str,
                 workers: int,
                 checksum: bool,
                 dry_run: bool) -> None:
    """
    Move one finished run, renamed to `name` on the scapepc (`src_name` is
    only different in a dry run, where nothing is renamed), to the server,
    as `dst_exp/name/name/<raw files>` plus `dst_exp/name/name_info.mat` and
    `dst_exp/name/name_stim_data.bin`.
    """
    src_run = src_exp / src_name
    dst_run = dst_exp / name
    dst_data = dst_run / name

    jobs = []
    for p in src_run.rglob('*'):
        if p.is_file() and p.name != MARKER:
            jobs.append((p, dst_data / p.relative_to(src_run)))
        elif p.is_dir() and not dry_run:
            (dst_data / p.relative_to(src_run)).mkdir(parents=True, exist_ok=True)
    # Sidecars last, so a half-transferred run is easy to recognise on the
    # server (it has no _info.mat yet)
    sidecars = [(src_exp / (src_name + s), dst_run / (name + s))
                for s in SIDECAR_SUFFIXES
                if (src_exp / (src_name + s)).is_file()]

    n_total = len(jobs) + len(sidecars)
    total_bytes = sum(s.stat().st_size for s, _ in jobs + sidecars)
    log(f'Moving {name}: {n_total} files, {total_bytes / 1e9:.1f} GB '
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
        (src_run / MARKER).unlink(missing_ok=True)
        remove_empty_dirs(src_run)
        if src_run.exists():
            log(f'  WARNING: {src_run} not empty after transfer')

    dt = time.time() - t0
    log(f'Done {name}: {copied / 1e9:.1f} GB in {dt / 60:.1f} min '
        f'({copied / 1e6 / max(dt, 1e-6):.0f} MB/s)')


def watch(src_exp: Path,
          dst_exp: Path,
          settle: float,
          no_sifx_after: float,
          timeout: float,
          poll: float,
          workers: int,
          checksum: bool,
          once: bool,
          dry_run: bool) -> None:
    """
    Poll `src_exp`, rename every finished run and move it to `dst_exp`,
    until nothing has happened for `timeout` seconds (or until no finished
    runs are left, if `once`).

    Parameters
    ----------
    src_exp, dst_exp : Path
        Experiment folder on the scapepc and on the server.
    settle : float
        Seconds without any file change before a finished run is renamed.
    no_sifx_after : float
        Seconds without any file change after which a run with an
        `_info.mat` and raw data but no `Spooled files.sifx` counts as
        finished.
    timeout : float
        Seconds without any activity (a run renamed or moved, or a run's
        files on the scapepc changing) after which to stop. 0 to never stop.
    poll : float
        Seconds between checks of `src_exp`.
    workers : int
        Number of files copied in parallel.
    checksum : bool
        Verify copies with md5 checksums in addition to file sizes.
    once : bool
        Transfer whatever is finished now, then exit.
    dry_run : bool
        Only report what would be renamed and moved.
    """
    last_state = {}   # run -> snapshot from previous poll
    reported = set()  # (run, reason) already logged, to log each only once
    dry_renamed = {}  # dry run only: original name -> would-be new name
    dry_moved = set()  # dry run only: original names already 'moved'
    last_activity = time.time()  # last rename/move or scapepc file change

    def report(run: str,
               reason: str,
               msg: str) -> None:
        if (run, reason) not in reported:
            log(msg)
            reported.add((run, reason))

    while True:
        runs = sorted([p.name for p in src_exp.iterdir() if p.is_dir()],
                      key=natural_key)
        renamed = {}    # new name -> original name, for renamed runs
        for run in runs:
            if (marker := read_marker(src_exp / run)) is None:
                continue
            orig, name = marker
            if run != name:  # rename was interrupted: finish it
                log(f'Finishing interrupted rename {orig} -> {name}')
                try:
                    finish_rename(src_exp, run, name)
                except OSError as e:
                    log(f'ERROR renaming {run}: {e!r} (will retry)')
                    continue
            renamed[name] = orig
        originals = {orig: name for name, orig in renamed.items()}

        # 1. Rename every finished run, so the SCAPE software can't reuse
        # its name
        for run in runs:
            src_run = src_exp / run
            if run in renamed or read_marker(src_run) is not None:
                continue
            files = run_files(src_exp, run)
            if not files:
                # Leftover empty folders (e.g. ModifiedData recreated by
                # Solis after a run was already moved)
                try:
                    old = time.time() - src_run.stat().st_mtime > settle
                except FileNotFoundError:
                    continue
                if old and not dry_run:
                    remove_empty_dirs(src_run)
                continue
            has_info = (src_exp / (run + '_info.mat')).is_file()
            has_raw = any(src_run.glob('*spool.dat'))
            state = snapshot(files)
            last_activity = max(last_activity, state[2])
            # Files written after the run was renamed: no sidecars and no
            # raw data (so not a new recording that reuses the name)
            if (run in originals and not has_info and not has_raw
                    and not (src_run / 'acquisitionmetadata.ini').exists()
                    and time.time() - state[2] > settle):
                try:
                    move_leftovers(src_exp, dst_exp, run, originals[run],
                                   dry_run)
                except Exception as e:
                    log(f'ERROR moving late files of {run}: {e!r} (will retry)')
                continue
            if has_info and not has_raw:
                report(run, 'no raw', f'  WARNING: {run} has an _info.mat '
                       f'but no raw data (failed acquisition?), leaving it')
                continue
            if not has_info:
                report(run, 'waiting', f'{run}: recording in progress, waiting')
                continue
            if not (src_run / END_MARKER).is_file():
                if time.time() - state[2] < no_sifx_after:
                    report(run, 'waiting',
                           f'{run}: recording in progress, waiting')
                    continue
                report(run, 'no sifx', f'  WARNING: {run} has no '
                       f'{END_MARKER} but nothing changed for '
                       f'{no_sifx_after / 60:.0f} min, moving it anyway')
            stable = (last_state.get(run) == state
                      and time.time() - state[2] > settle)
            last_state[run] = state
            if not stable or run in dry_renamed:
                continue
            try:
                name = rename_run(src_exp, dst_exp, run, dry_run)
            except Exception as e:  # keep watching, retry on next poll
                log(f'ERROR renaming {run}: {e!r} (will retry)')
                continue
            last_state.pop(run, None)
            # the SCAPE software reuses names: report the next one again
            reported.difference_update({k for k in reported if k[0] == run})
            if dry_run:
                dry_renamed[run] = name
            else:
                renamed[name] = run

        # 2. Move the renamed run with the lowest number, then check for
        # newly finished runs again before moving the next one
        todo = sorted(renamed, key=natural_key)
        todo_dry = [(run, name) for run, name in dry_renamed.items()
                    if run not in dry_moved]
        if todo or todo_dry:
            src_name, name = ((todo[0], todo[0]) if todo else todo_dry[0])
            try:
                transfer_run(src_exp, dst_exp, src_name, name, workers,
                             checksum, dry_run)
            except Exception as e:  # keep watching, retry on next poll
                log(f'ERROR moving {name}: {e!r} (will retry)')
                time.sleep(poll)
            if dry_run:
                dry_moved.add(src_name)
            last_activity = time.time()
            continue

        if once and not last_state:
            return
        idle = time.time() - last_activity
        if timeout and idle > timeout:
            log(f'Nothing happened for {timeout / 3600:g} h, stopping')
            return
        try:
            free = shutil.disk_usage(src_exp).free / 1e9
            status = f'{free:.0f} GB free on scapepc'
            if timeout:
                status += (f', idle {idle / 3600:.1f} h of '
                           f'{timeout / 3600:g} h')
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
    parser.add_argument('--no-sifx-after', type=float, default=1800,
                        help='Seconds without file changes after which a run '
                             'with an _info.mat and raw data but no '
                             f'"{END_MARKER}" is moved anyway (default: 1800)')
    parser.add_argument('--timeout', type=float, default=3,
                        help='Hours without any activity (a run renamed or '
                             'moved, or a run\'s files on the scapepc '
                             'changing) after which to stop; 0 to never stop '
                             '(default: 3)')
    parser.add_argument('--poll', type=float, default=60,
                        help='Seconds between checks (default: 60)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Files copied in parallel (default: 4)')
    parser.add_argument('--checksum', action='store_true',
                        help='Also verify md5 checksums (slower)')
    parser.add_argument('--once', action='store_true',
                        help='Move runs that are already finished, then exit')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only print what would be renamed and moved')
    args = parser.parse_args()

    global _log_file
    src_exp = find_experiment(args.src_root, args.experiment)
    dst_exp = args.dst_root / src_exp.name
    if not args.dry_run:
        dst_exp.mkdir(parents=True, exist_ok=True)
        _log_file = dst_exp / LOG_NAME
    log(f'Watching {src_exp}\n    moving finished runs to {dst_exp}\n'
        f'    settle={args.settle:.0f}s no_sifx_after={args.no_sifx_after:.0f}s '
        f'timeout={args.timeout:g}h '
        f'poll={args.poll:.0f}s workers={args.workers} '
        f'checksum={args.checksum}')
    try:
        watch(src_exp, dst_exp, args.settle, args.no_sifx_after,
              args.timeout * 3600, args.poll,
              args.workers, args.checksum, args.once, args.dry_run)
    except KeyboardInterrupt:
        log('Stopped by user')


if __name__ == '__main__':
    main()
