"""
TEMPORARY script: convert the runs of existing SCAPE experiments from .dat spool files to
single `.ozx` files, then delete the .dat files. Delete this script once all experiments are
converted (new recordings are converted by `transfer_scape_recordings.py`).

For each experiment folder given, every `sample#_run#` folder that still has its raw data
subfolder (`sample#_run#/sample#_run#/`, removed once its .dat files are deleted; the .dat
files themselves are never listed beforehand, as there are thousands per run) (not the `_HR` ones, which are converted to nrrd by `preprocess_scape_exp.py`, nor the
`sample#_combined` ones) gets its folder structure fixed (`scape_utils.check_folder_structure`)
and is converted with `scapeio.ozx.convert.convert` to `sample#_run#/sample#_run#.ozx`
(`SHARD_VOLUMES` volumes per shard), which `scape_utils.RawRecording` then reads instead. The
converter checks the file (every frame decoded and compared after encoding; sampled frames, the
DAQ data and the sidecar files read back with an independent reader; every byte read back from
the server and checked against its CRC32) before giving it its final name, and only then
deletes the run's .dat, `.ini`, `.sifx`, `ModifiedData`, `_info.mat` and `_stim_data.bin` files
(all stored in the `.ozx`), unless `--keep-dat`. A run that already has a `.ozx` but still has
its raw data subfolder is converted again (overwriting it), so its .dat files are only deleted
after a check.

A 220 GB run takes ~50 min (reading the .dat files from the server is the limit). Runs that
fail are logged and skipped. Everything is logged
to `<exp_folder>/ozx_conversion_log.txt`.

Afterwards, recompute the projections of every converted experiment (a `.ozx` volume has 2
planes fewer than a .dat one, see `scape_utils.RawRecording`):
`python preprocess_scape_exp.py <exp_folder> -o`.

Usage (from the `scape` folder): `uv run python convert_exps_to_ozx_TEMP.py <exp_folder> [...]
[--keep-dat]`
"""

import argparse
import time
import traceback
from datetime import datetime
from pathlib import Path

from scapeio.ozx.convert import convert

import scape_utils

SHARD_VOLUMES = 10   # volumes per shard (~1.6 GB shards for 754-plane volumes)
LOG_NAME = 'ozx_conversion_log.txt'


def _log(exp_folder: Path,
         msg: str) -> None:
    """
    Print `msg` with a timestamp and append it to the experiment's log file.
    """
    line = f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}'
    print(line, flush=True)
    with open(exp_folder / LOG_NAME, 'a') as f:
        f.write(line + '\n')


def _has_raw_data(scape_path: Path) -> bool:
    """
    Whether a run still has its raw data: its raw data subfolder, or (before its folder
    structure is fixed) .dat files directly in the run folder. Never lists the raw data
    subfolder, which holds thousands of .dat files.
    """
    return ((scape_path / scape_path.name).is_dir()
            or any(p.suffix == '.dat' for p in scape_path.iterdir()))


def runs_to_convert(exp_folder: Path) -> list[Path]:
    """
    The `sample#_run#` folders of an experiment that still have raw data (not `_HR`
    or `sample#_combined` ones), sorted by name.

    Parameters
    ----------
    exp_folder : Path
        The experiment folder.

    Returns
    -------
    list[Path]
        The run folders to convert.
    """
    return sorted(p for p in exp_folder.iterdir()
                  if p.is_dir() and p.name.startswith('sample') and '_run' in p.name
                  and not p.name.endswith('_HR') and _has_raw_data(p))


def convert_one_run(scape_path: Path,
                    delete_dat: bool) -> bool:
    """
    Convert one run to `scape_path/scape_path.name.ozx` (see the module docstring), and
    delete its .dat files once the `.ozx` is checked (if `delete_dat`).

    Parameters
    ----------
    scape_path : Path
        The `sample#_run#` folder.
    delete_dat : bool
        Delete the run's .dat and sidecar files once the `.ozx` is checked.

    Returns
    -------
    bool
        Whether the conversion succeeded.
    """
    exp_folder = scape_path.parent
    try:
        scape_utils.check_folder_structure(scape_path)
        raw_path = scape_path / scape_path.name
        _log(exp_folder, f'Converting {scape_path.name}...')
        t0 = time.time()
        last_tenth = [0]

        def progress(done: int,
                     total: int) -> None:
            tenth = 10 * done // max(total, 1)
            if tenth > last_tenth[0]:
                last_tenth[0] = tenth
                _log(exp_folder, f'  {scape_path.name}: {tenth * 10}% of the frames converted '
                                 f'({(time.time() - t0) / 60:.0f} min)')

        report = convert(raw_path, scape_path, colors=1, shard_volumes=SHARD_VOLUMES,
                         overwrite=True, delete_source=delete_dat,
                         progress=progress)
        if delete_dat:
            try:  # the converter deletes the files; remove the emptied folder
                raw_path.rmdir()
            except OSError:
                pass
        for warning in report.get('warnings', []):
            _log(exp_folder, f'  {scape_path.name}: converter warning: {warning}')
        dt = time.time() - t0
        _log(exp_folder, f'Done {scape_path.name}: {report["output_bytes"] / 1e9:.1f} GB '
                         f'({report["ratio_frames"]:.2f}x smaller) '
                         f'in {dt / 60:.1f} min'
                         + ('; .dat files deleted' if delete_dat else '; .dat files kept'))
        return True
    except Exception as e:
        (scape_path / f'{scape_path.name}.ozx.partial').unlink(missing_ok=True)
        _log(exp_folder, f'ERROR converting {scape_path.name}: {e!r}; its .dat files are '
                         f'untouched\n{traceback.format_exc()}')
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description='TEMPORARY: convert existing SCAPE '
                                                 'experiments from .dat files to .ozx.')
    parser.add_argument('exp_folders', nargs='+', type=Path,
                        help='Experiment folders (full paths, or names in '
                             f'{scape_utils.MAIN_PATH}).')
    parser.add_argument('--keep-dat', action='store_true',
                        help='Keep the .dat files after the conversion.')
    args = parser.parse_args()

    exp_folders = [p if p.is_absolute() else Path(scape_utils.MAIN_PATH) / p
                   for p in args.exp_folders]
    todo = []
    for exp_folder in exp_folders:
        if not exp_folder.is_dir():
            raise SystemExit(f'Not a folder: {exp_folder}')
        runs = runs_to_convert(exp_folder)
        print(f'{exp_folder.name}: {len(runs)} runs with raw data')
        for run in runs:
            print(f'    {run.name}')
        todo += runs
    if not todo:
        print('Nothing to convert.')
        return
    action = 'keeping' if args.keep_dat else 'deleting (after checking each .ozx)'
    print(f'Converting these {len(todo)} runs and {action} their .dat files...')

    results = [convert_one_run(run, not args.keep_dat) for run in todo]
    failed = [run for run, ok in zip(todo, results) if not ok]
    print(f'\nConverted {len(todo) - len(failed)} of {len(todo)} runs.')
    for run in failed:
        print(f'    FAILED: {run.parent.name}/{run.name} (see {run.parent / LOG_NAME})')
    if failed:  # nonzero exit, so a chained `&& preprocess_scape_exp.py` does not start
        raise SystemExit('Run this script again to retry the failed runs.')
    print('\nNow recompute the projections of each converted experiment:')
    for exp_folder in exp_folders:
        print(f'    uv run python preprocess_scape_exp.py {exp_folder} -o')


if __name__ == '__main__':
    main()
