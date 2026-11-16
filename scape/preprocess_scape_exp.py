"""
Script to preprocess a scape experiment
Takes as input the folder for a scape experiment.
Expects the folder to contain subfolders called sample#_run#,
some with _HR in the end (high-resolution single-volume recordings).
First, the folder structure of every _HR folder is fixed like for normal runs (see
scape_utils.check_folder_structure), and every _HR folder that still has raw .dat files is
converted (not deskewed) to a single nrrd file (sample#_run#_HR/sample#_run#_HR/
sample#_run#_HR.nrrd, via scapeio.convert), which is checked before the .dat files are
deleted. Each HR nrrd is also deskewed and saved as a calibrated OME-TIFF next to it
(sample#_run#_HR_deskewed.ome.tif), if not already there.
Then lists all non-_HR folders, sorted by sample# then run#, each with an index, and asks
which indices to keep (supports ranges, e.g. '2-5,7,9-12').
Then, sample by sample, goes run by run and checks:
- if the folder structure is correct (raw data inside a sub-subfolder)
- if the max/min/std projections have been calculated already
- if the median projection has been calculated already (exact for runs with fewer than 75
  volumes, `scape_utils.MEDIAN_MAX_VOLUMES`; longer runs get an approximate median from 50
  evenly spaced volumes, `scape_utils.MEDIAN_SAMPLES`, as it needs all its volumes at once)
If any of these have not been done, it does it (a missing median alone is computed
without recomputing the other projections, loading only the volumes it needs).
Once all runs for a sample are done, if the sample has more than one (kept) run, it
creates a new folder called sample#_combined and combines all the projections for that
sample, before moving on to the next sample. Samples with a single run get no
sample#_combined folder (use the sample#_run# folder directly).

Long recordings: if the experiment folder name ends with the puparium formation time
(`..._MM-DD_HHhMM`, e.g. `20261023_V7xPR_pupa_long_scape_prepup_10-21_11h15`, optionally
followed by a suffix such as `_FAIL`), the runs get no individual activity plot. Instead, at
the end (also in live mode), every run except the `_short` ones is summarized in
`<exp_folder>/00_preanalysis/` (see save_preanalysis and `common.preanalysis`, as in
`flytrack/preprocess_flytrack.py`), each run labelled with its hours after puparium formation
(hAPF):
- `activity_shared.png` / `activity_own.png`: the saved `activity` traces (no raw data loaded)
  in 3-column grids in time order (shared x and y limits with the subplots touching, so each
  row reads as one continuous trace, or each subplot with its own y limits)
- `max_projection_grid.png` / `median_grid.png`: every run's max / median projection, flattened
  by averaging over depth (axis 0 of the deskewed (z, x, y) projection) and shown with its
  true x/y aspect ratio, all with one shared contrast
The recording time is `scanStartTimeApprox` from the run's `_info.mat`, or else the
`_YYYYMMDD_HHMM` at the end of the run folder name; if neither is found the hAPF is omitted.
The sample#_combined activity plot stays in its folder. The frequency analysis of the activity
is in `analysis/scape/`.

Live mode (-l/--live): to run alongside `transfer_scape_recordings.py` during a long
acquisition. There is no selection prompt; every run is preprocessed as soon as it has been
fully transferred to the server (its `_info.mat`, which the transfer script moves last, is
there), checking for new runs every minute. When no new run has appeared for 3 hours
(--live-timeout), or on Ctrl+C, the runs are combined per sample and the script ends.

Can be run from the terminal (`python preprocess_scape_exp.py [exp_folder] [-b N] [-o] [-l]`)
or imported and called as a function. If `exp_folder` is not given, a folder-selection dialog
opens, starting in `scape_utils.MAIN_PATH`, with the newest (`YYYYMMDD_...`) folders on top.
The number of volumes loaded per batch when computing projections is chosen automatically
from each run's volume size, unless given with -b. With -o/--overwrite, projections are
recomputed even if they already exist.

save_HR_deskewed_tif: deskew a converted HR nrrd and save it as a calibrated OME-TIFF.
convert_HR_folders: convert all sample#_run#_HR subfolders of an experiment folder to a
    single (not deskewed) nrrd each, delete their raw .dat files, and save a deskewed
    OME-TIFF of each (see save_HR_deskewed_tif).
load_activity_traces: load the activity traces of a long-recording experiment's runs.
save_preanalysis: save the activity, max projection and median grids of a long recording in
    `<exp_folder>/00_preanalysis/` (can be run on its own, e.g. for experiments processed
    before it existed).
preprocess_scape_exp: main function, converts the HR folders (see convert_HR_folders), then
    lists and preprocesses all other sample#_run# subfolders of an experiment folder, and
    combines projections per sample (for samples with more than one run); with live=True,
    preprocesses the runs as they are transferred and combines them at the end.
"""

import argparse
import datetime
import re
import time
from pathlib import Path

import h5py
import numpy as np
import scapeio
import tifffile

import scape_utils
from common import preanalysis
from common.dialogs import ask_directory
from common.preanalysis import PREANALYSIS_FOLDER, get_hours_apf, get_time_0

# image grids in 00_preanalysis: dataset in projection_images.h5 -> file name
GRID_NAMES = {'max_proj': 'max_projection_grid.png', 'median_proj': 'median_grid.png'}
LIVE_POLL = 60              # live mode: seconds between checks for new runs
LIVE_TIMEOUT_HOURS = 3      # live mode: stop after this long without a new run


def _sample_run_sort_key(scape_path: Path) -> tuple[int, int]:
    """
    Sort key for `sample#_run#` folder names, sorting numerically by sample#
    then run# (so e.g. `run9` sorts before `run10`).

    Parameters
    ----------
    scape_path : Path
        Path to a `sample#_run#` folder.

    Returns
    -------
    tuple[int, int]
        (sample number, run number).
    """
    match = re.match(r'sample(\d+)_run(\d+)', scape_path.name)
    return (int(match.group(1)), int(match.group(2)))


def _parse_index_ranges(indices_input: str) -> set[int]:
    """
    Parse a comma-separated list of 1-based indices and ranges (e.g.
    `'2-5,7,9-12'`) into a set of integers.

    Parameters
    ----------
    indices_input : str
        Comma-separated indices and/or ranges (`'a-b'`, inclusive).

    Returns
    -------
    set[int]
        The individual indices described by `indices_input`.
    """
    indices = set()
    for part in indices_input.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start, end = part.split('-')
            indices.update(range(int(start), int(end) + 1))
        else:
            indices.add(int(part))
    return indices


def save_HR_deskewed_tif(hr_path: str | Path) -> Path:
    """
    Deskew a high-resolution (single-volume) recording that was already
    converted to nrrd (see `convert_HR_folders`) and save it as a calibrated
    OME-TIFF next to the nrrd, as
    `sample#_run#_HR/sample#_run#_HR/sample#_run#_HR_deskewed.ome.tif`.

    Uses the same axis order and metadata as the projection OME-TIFFs (see
    `scape_utils.get_projection_images`): the deskewed volume (depth, galvo
    scan axis, camera image width) is transposed to ZYX, with the deskewed
    voxel size from `scape_utils.load_voxel_size` embedded, so it opens
    correctly scaled in napari. The saved file is checked by reading it back.
    Skipped if the tif already exists.

    Parameters
    ----------
    hr_path : str | Path
        Path to a `sample#_run#_HR` folder (with its folder structure already
        fixed by `scape_utils.check_folder_structure`).

    Returns
    -------
    Path
        Path to the deskewed OME-TIFF.
    """
    hr_path = Path(hr_path)
    raw_path = hr_path / hr_path.name
    tiff_path = raw_path / f'{hr_path.name}_deskewed.ome.tif'
    if tiff_path.exists():
        print(f'Deskewed tif already found for {hr_path.name}, skipping.')
        return tiff_path

    print(f'Deskewing {hr_path.name} and saving it as a tif...')
    raw_volume = scapeio.load_scan(str(raw_path), colors=1, try_loading_copies=['nrrd'])
    deskewed = scapeio.operations.deskew_scan(str(raw_path), images=raw_volume)
    del raw_volume
    if np.issubdtype(deskewed.dtype, np.floating):
        deskewed = np.clip(np.rint(deskewed), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    deskewed = deskewed.transpose(0, 2, 1)

    voxel_size = scape_utils.load_voxel_size(hr_path)
    tiff_metadata = {'axes': 'ZYX',
                     'PhysicalSizeX': voxel_size['x'], 'PhysicalSizeXUnit': 'um',
                     'PhysicalSizeY': voxel_size['y'], 'PhysicalSizeYUnit': 'um',
                     'PhysicalSizeZ': voxel_size['z'], 'PhysicalSizeZUnit': 'um'}
    tifffile.imwrite(tiff_path, deskewed, ome=True, metadata=tiff_metadata)

    if not np.array_equal(tifffile.imread(tiff_path), deskewed):
        raise RuntimeError(f'{tiff_path} did not read back identical to the deskewed volume.')
    print(f'Saved {tiff_path} (shape {deskewed.shape}, {deskewed.dtype}).')
    return tiff_path


def convert_HR_folders(exp_folder: str | Path) -> None:
    """
    Convert every high-resolution (single-volume) recording in an experiment
    folder, i.e. every `sample#_run#_HR` subfolder, to a single nrrd file with
    `scapeio.convert` (not deskewed), then delete its raw .dat files.

    First fixes the folder structure of each HR folder like for normal runs
    (see `scape_utils.check_folder_structure`): `_info.mat` and `_stim_data.bin`
    are moved into the HR folder and the raw data into an HR/HR subfolder (also
    for HR folders converted before, which get their existing .nrrd moved).
    The nrrd is saved by scapeio next to the raw data, as
    `sample#_run#_HR/sample#_run#_HR/sample#_run#_HR.nrrd` (with the raw voxel
    size in its header), so `scapeio.load_scan(hr_path / hr_path.name, colors=1)`
    loads it afterwards instead of the .dat files (and can then be deskewed with
    `scapeio.operations.deskew_scan`). Before deleting the .dat files, the nrrd is
    checked by loading it back and making sure it's a non-empty 3D volume, and by
    `scapeio.delete_dat_files` checking it's big enough to hold the full scan.
    HR folders with no .dat files left (already converted) skip the conversion.
    Finally, every HR folder with an nrrd gets a deskewed OME-TIFF (see
    `save_HR_deskewed_tif`), also those converted in earlier runs.

    Parameters
    ----------
    exp_folder : str | Path
        Path to the experiment folder containing the `sample#_run#_HR` subfolders.

    Returns
    -------
    None
    """
    hr_paths = sorted((p for p in Path(exp_folder).iterdir()
                       if p.is_dir() and p.name.startswith('sample') and p.name.endswith('_HR')),
                      key=_sample_run_sort_key)
    print(f'Found {len(hr_paths)} HR folders in {exp_folder}')

    for hr_path in hr_paths:
        scape_utils.check_folder_structure(hr_path)
        raw_path = str(hr_path / hr_path.name) # raw data folder, string needed for scapeio
        if not scapeio.io.list_dat_files(raw_path):
            print(f'No .dat files found in {raw_path}, skipping conversion (already converted?)')
        else:
            nrrd_path = scapeio.io.get_data_copy_name(raw_path, copy_format='nrrd', colors=1,
                                                      check_exists=True)
            if nrrd_path is None:
                print(f'Converting {raw_path} to nrrd...')
                nrrd_path = scapeio.convert(raw_path, output_format='nrrd', colors=1)
            else:
                print(f'nrrd already found for {raw_path}, checking it...')

            # check the nrrd loads back fine before deleting the raw data
            volume = scapeio.load_scan(raw_path, colors=1, try_loading_copies=['nrrd'])
            if volume.ndim != 3 or volume.size == 0 or not volume.any():
                raise RuntimeError(f'{nrrd_path} loaded back as an empty or non-3D volume '
                                   f'(shape {volume.shape}). Raw .dat files were NOT deleted.')
            print(f'Checked {nrrd_path} (shape {volume.shape}, {volume.dtype}).')
            del volume

            scapeio.delete_dat_files(raw_path, verify_copy_exists='nrrd', fake=False)

        if scapeio.io.get_data_copy_name(raw_path, copy_format='nrrd', colors=1,
                                         check_exists=True) is None:
            print(f'No nrrd found for {raw_path}, skipping deskewed tif.')
            continue
        save_HR_deskewed_tif(hr_path)


def _get_recording_time(scape_path: Path) -> datetime.datetime | None:
    """
    Get the start time of a run: `scanStartTimeApprox` from its `_info.mat`
    (inside the run folder, or next to it as acquired), or else the
    `_YYYYMMDD_HHMM` at the end of the run folder name.

    Parameters
    ----------
    scape_path : Path
        The `sample#_run#` folder.

    Returns
    -------
    datetime.datetime or None
        Recording time, or None if it can't be found.
    """
    for info_path in (scape_path / f'{scape_path.name}_info.mat',
                      scape_path.parent / f'{scape_path.name}_info.mat'):
        if not info_path.is_file():
            continue
        try:
            from scipy.io import loadmat
            info = loadmat(info_path, squeeze_me=True, struct_as_record=False,
                           variable_names=['info'])['info']
            return datetime.datetime.strptime(str(info.scanStartTimeApprox),
                                              '%d-%b-%Y %H:%M:%S')
        except Exception:
            break
    m = re.search(r'_(\d{8})_(\d{4})$', scape_path.name)
    if m:
        return datetime.datetime.strptime(m[1] + m[2], '%Y%m%d%H%M')
    return None


def load_activity_traces(exp_folder: str | Path) -> list[dict]:
    """
    Load the activity traces of a long-recording experiment: the saved `activity` of every
    run with projections, except the `_short` ones, in sample# then run# order, labelled with
    its hours APF (plus the sample name if the experiment has several samples).

    Parameters
    ----------
    exp_folder : str | Path
        Path to the experiment folder.

    Returns
    -------
    list[dict]
        One trace per run, with keys 'name' (run folder name), 'title', 'hours' (int or None),
        'activity' and 'fs' (the `scan_rate` attribute) (see `common.preanalysis`).
    """
    exp_folder = Path(exp_folder)
    time_0 = get_time_0(exp_folder)
    run_paths = _long_run_folders(exp_folder)
    several_samples = len({p.name.split('_run')[0] for p in run_paths}) > 1

    traces = []
    for scape_path in run_paths:
        with h5py.File(scape_path / scape_utils.ANALYSIS_PATH_NAME
                       / scape_utils.PROJECTION_FILE_NAME, 'r') as f:
            activity = f['activity'][:]  # type: ignore[index]
            scan_rate = float(f.attrs['scan_rate'])  # type: ignore[arg-type]
        hours = get_hours_apf(time_0, _get_recording_time(scape_path))
        title = scape_path.name if hours is None else f'{hours} hAPF'
        if hours is not None and several_samples:
            title = scape_path.name.split('_run')[0] + ' - ' + title
        traces.append({'name': scape_path.name, 'title': title, 'hours': hours,
                       'activity': activity, 'fs': scan_rate})
    return traces


def _long_run_folders(exp_folder: Path) -> list[Path]:
    """
    The runs of a long recording that are summarized: every run with projections, except the
    `_short` ones, sorted by sample# then run#.
    """
    return [p for p in _list_run_folders(exp_folder)
            if _has_projections(p) and '_short' not in p.name]


def _save_grids(exp_folder: Path) -> None:
    """
    Save grids of the runs' max and median projections (see `GRID_NAMES`) in
    `<exp_folder>/00_preanalysis/`, each flattened by averaging over depth (axis 0 of the
    deskewed (z, x, y) projection) and shown with its true aspect ratio (x voxel size over y
    voxel size), labelled with its recording time and hours APF. Runs without a median
    projection are left out of the median grid.
    """
    time_0 = get_time_0(exp_folder)
    run_paths = _long_run_folders(exp_folder)
    for key, name in GRID_NAMES.items():
        images, titles, aspects = [], [], []
        for scape_path in run_paths:
            with h5py.File(scape_path / scape_utils.ANALYSIS_PATH_NAME
                           / scape_utils.PROJECTION_FILE_NAME, 'r') as f:
                if key not in f:
                    continue
                images.append(np.mean(f[key][:], axis=0))  # type: ignore[index]
                voxel_x, voxel_y, _ = f['voxel_sizes_xyz'][:]  # type: ignore[index]
            aspects.append(voxel_x / voxel_y)
            time_t = _get_recording_time(scape_path)
            hours = get_hours_apf(time_0, time_t)
            title = time_t.strftime('%m%d_%H%M') if time_t is not None else scape_path.name
            titles.append(title if hours is None else f'{title} - {hours}h APF')
        if len(images) < len(run_paths):
            print(f'{len(run_paths) - len(images)} runs have no {key}, left out of {name}.')
        if len(set(aspects)) > 1:
            print(f'Warning: the runs have different voxel sizes; {name} uses the first '
                  f'run\'s aspect ratio.')
        if images:
            preanalysis.save_image_grid(images, titles, exp_folder / PREANALYSIS_FOLDER / name,
                                        aspect=aspects[0], suptitle=exp_folder.name)


def save_preanalysis(exp_folder: str | Path) -> None:
    """
    Save the pre-analysis plots of a long-recording experiment in
    `<exp_folder>/00_preanalysis/`: the activity grids of its runs (see
    `load_activity_traces()` and `common.preanalysis.save_activity_summary()`) and the grids of
    their max and median projections (see `_save_grids()`). Does nothing if the experiment
    folder name does not end with the puparium formation time (`..._MM-DD_HHhMM`).

    Parameters
    ----------
    exp_folder : str | Path
        Path to the experiment folder.

    Returns
    -------
    None
    """
    exp_folder = Path(exp_folder)
    if get_time_0(exp_folder) is None:
        return
    preanalysis.save_activity_summary(load_activity_traces(exp_folder),
                                      exp_folder / PREANALYSIS_FOLDER, exp_folder.name,
                                      ylabel='Average fluorescence', linewidth=0.5)
    _save_grids(exp_folder)


def _list_run_folders(exp_folder: Path) -> list[Path]:
    """
    List the `sample#_run#` folders of an experiment (not the `_HR` ones),
    sorted by sample# then run#.
    """
    return sorted((p for p in exp_folder.iterdir()
                   if p.is_dir() and p.name.startswith('sample') and '_run' in p.name
                   and not p.name.endswith('_HR')),
                  key=_sample_run_sort_key)


def _is_transferred(scape_path: Path) -> bool:
    """
    Whether a run has been fully transferred to the server: its `_info.mat`,
    which `transfer_scape_recordings.py` moves last, is inside the run folder
    (or next to it, as acquired).
    """
    info_name = f'{scape_path.name}_info.mat'
    return (scape_path / info_name).is_file() or (scape_path.parent / info_name).is_file()


def _has_projections(scape_path: Path) -> bool:
    """
    Whether the projection images of a run have been computed.
    """
    return (scape_path / scape_utils.ANALYSIS_PATH_NAME
            / scape_utils.PROJECTION_FILE_NAME).exists()


def _has_median(scape_path: Path) -> bool:
    """
    Whether the median projection of a run has been computed (see
    `scape_utils.get_median_projection`).
    """
    with h5py.File(scape_path / scape_utils.ANALYSIS_PATH_NAME
                   / scape_utils.PROJECTION_FILE_NAME, 'r') as f:
        return 'median_proj' in f


def _process_run(exp_folder: Path,
                 scape_path: Path,
                 time_0: datetime.datetime | None,
                 batch_size: int | None,
                 overwrite: bool) -> None:
    """
    Fix the folder structure of one run and compute its projection images,
    unless they exist already (and not `overwrite`), including the median projection
    (computed on its own if only it is missing; exact or approximate, see
    `scape_utils.get_median_projection`). Long recordings (`time_0` known) get no activity
    plot (they are summarized in `00_preanalysis`, see `save_preanalysis()`).
    """
    scape_utils.check_folder_structure(scape_path)
    exists = _has_projections(scape_path)
    if exists and not overwrite:
        if not _has_median(scape_path):
            print(f'Projections already found for {scape_path.name}, but not the median '
                  f'projection. Computing it...')
            scape_utils.get_median_projection(scape_path, batch_size=batch_size)
        else:
            print(f'Projections already found for {scape_path.name}.')
        return
    print(f'Projections {"found, overwriting" if exists else "not found"} '
          f'for {scape_path.name}. Computing them (with the median)...')
    scape_utils.get_projection_images(scape_path, batch_size=batch_size, median=True,
                                      plot_activity=time_0 is None)


def _combine_samples(exp_folder: Path,
                     scape_paths: list[Path]) -> None:
    """
    Combine the projections of the runs in `scape_paths` that have them into
    one `sample#_combined` folder per sample with more than one such run.
    """
    sample_run_paths: dict[str, list[Path]] = {}
    for scape_path in scape_paths:
        if _has_projections(scape_path):
            sample_name = scape_path.name.split('_run')[0]
            sample_run_paths.setdefault(sample_name, []).append(scape_path)
    for sample_name, run_paths in sample_run_paths.items():
        if len(run_paths) > 1:
            scape_utils.combine_sample_projections(exp_folder, sample_name, run_paths)
        else:
            print(f'Only one run for {sample_name}, not creating a {sample_name}_combined folder.')


def _preprocess_live(exp_folder: Path,
                     time_0: datetime.datetime | None,
                     batch_size: int | None,
                     overwrite: bool,
                     timeout_hours: float,
                     poll: float = LIVE_POLL) -> None:
    """
    Preprocess every run of the experiment as soon as it has been fully
    transferred (see `_is_transferred()`), checking for new runs every `poll`
    seconds, until no new run has appeared for `timeout_hours` or the user
    presses Ctrl+C. Then combines the projections of each sample.
    """
    done: set[str] = set()   # runs processed (or found processed) in this session
    last_new = time.time()   # when the last run was processed (or the start)
    status_shown = False     # True while the one-line waiting status is on screen
    print(f'Live mode: preprocessing runs as they are transferred, until no new run '
          f'appears for {timeout_hours:g} h (or Ctrl+C), then combining the samples.')
    try:
        while True:
            todo = [p for p in _list_run_folders(exp_folder)
                    if p.name not in done and _is_transferred(p)]
            if todo and status_shown:
                print()
                status_shown = False
            for scape_path in todo:
                _process_run(exp_folder, scape_path, time_0, batch_size, overwrite)
                done.add(scape_path.name)
                last_new = time.time()
            idle = time.time() - last_new
            if idle > timeout_hours * 3600:
                print(f'\nNo new runs for {timeout_hours:g} h, stopping.')
                break
            print(f'\r[{datetime.datetime.now():%H:%M:%S}] waiting for new runs '
                  f'({idle / 3600:.1f} h of {timeout_hours:g} h without one)   ',
                  end='', flush=True)
            status_shown = True
            time.sleep(poll)
    except KeyboardInterrupt:
        print('\nStopped by user (Ctrl+C).')
    _combine_samples(exp_folder, _list_run_folders(exp_folder))
    save_preanalysis(exp_folder)


def preprocess_scape_exp(exp_folder: str | Path | None = None,
                         batch_size: int | None = None,
                         overwrite: bool = False,
                         live: bool = False,
                         live_timeout: float = LIVE_TIMEOUT_HOURS) -> None:
    """
    Preprocess all SCAPE recordings in an experiment folder.

    First converts all `sample#_run#_HR` subfolders to (not deskewed) nrrd,
    deletes their raw .dat files and saves a deskewed OME-TIFF of each (see
    `convert_HR_folders`). Then lists all `sample#_run#` subfolders (ignoring any ending in `_HR`), sorted
    by sample# then run#, each with an index, and asks which indices to keep
    (supports ranges, e.g. `'2-5,7,9-12'`). Then, sample by sample, for each
    remaining run checks (and if needed fixes/computes) the folder structure
    and projection images via `scape_utils` (including the median projection; exact for
    runs with fewer than `scape_utils.MEDIAN_MAX_VOLUMES` volumes, else approximate), and
    once all of a sample's runs
    are done, combines their projections into a new `sample#_combined` folder
    before moving on to the next sample. Samples with a single (kept) run get
    no `sample#_combined` folder.
    If the experiment folder name ends with the puparium formation time
    (`..._MM-DD_HHhMM`), the runs get no individual activity plot; instead, at
    the end, all runs are summarized in `<exp_folder>/00_preanalysis/` (activity,
    max projection and median grids, see `save_preanalysis`).

    In live mode (`live=True`), there is no selection prompt: every run is
    preprocessed as soon as it has been fully transferred to the server by
    `transfer_scape_recordings.py` (its `_info.mat` is there), new runs are
    checked for every minute, and once no new run has appeared for
    `live_timeout` hours, or on Ctrl+C, the runs' projections are combined per
    sample.

    Parameters
    ----------
    exp_folder : str | Path | None
        Path to the experiment folder containing the `sample#_run#` subfolders.
        If None, a folder-selection dialog opens starting in `scape_utils.MAIN_PATH`,
        listing the newest (`YYYYMMDD_...`) folders first.
    batch_size : int | None
        Number of volumes loaded at a time when computing the projection images
        (passed to `scape_utils.get_projection_images`). If None (default), chosen
        automatically for each run from its volume size (see
        `scape_utils.get_auto_batch_size`).
    overwrite : bool
        If True, recompute the projection images of every kept run even if they
        already exist (the sample#_combined projections are always recomputed).
        Default False.
    live : bool
        Keep running and preprocess new runs as they are transferred (see
        above). Default False.
    live_timeout : float
        In live mode, hours without a new run after which the script combines
        the samples and stops. Default 3.

    Returns
    -------
    None
    """
    if exp_folder is None:
        selected = ask_directory(scape_utils.MAIN_PATH, 'Select experiment folder')
        if not selected:
            print('No folder selected. Exiting.')
            return
        exp_folder = selected
    exp_folder = Path(exp_folder)
    time_0 = get_time_0(exp_folder)
    if time_0 is not None:
        print(f'Long recording (puparium formation at {time_0:%m-%d %H:%M}): the activity '
              f'is summarized in {exp_folder / PREANALYSIS_FOLDER}, with hours APF.')

    if live:
        convert_HR_folders(exp_folder)
        _preprocess_live(exp_folder, time_0, batch_size, overwrite, live_timeout)
        return

    scape_paths = _list_run_folders(exp_folder)

    print(f'Found {len(scape_paths)} run folders in {exp_folder}:')
    previous_sample_name = None
    for index, scape_path in enumerate(scape_paths, start=1):
        sample_name = scape_path.name.split('_run')[0]
        if previous_sample_name is not None and sample_name != previous_sample_name:
            print()
        print(f'  [{index}] {scape_path.name}')
        previous_sample_name = sample_name

    keep_input = input("Enter indices to keep (e.g. '2-5,7,9-12'), or press Enter to keep all: ")
    if keep_input.strip():
        keep_indices = _parse_index_ranges(keep_input)
        scape_paths = [p for index, p in enumerate(scape_paths, start=1) if index in keep_indices]

    convert_HR_folders(exp_folder)

    sample_run_paths: dict[str, list[Path]] = {}
    for scape_path in scape_paths:
        sample_name = scape_path.name.split('_run')[0]
        sample_run_paths.setdefault(sample_name, []).append(scape_path)

    for sample_name, run_paths in sample_run_paths.items():
        for scape_path in run_paths:
            _process_run(exp_folder, scape_path, time_0, batch_size, overwrite)
        _combine_samples(exp_folder, run_paths)

    # long recordings: analysis of every run's activity (not only the kept ones)
    save_preanalysis(exp_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess a SCAPE experiment folder.')
    parser.add_argument('exp_folder', nargs='?', default=None,
                        help='Path to the experiment folder. If omitted, a folder-selection '
                             f'dialog opens starting in {scape_utils.MAIN_PATH}.')
    parser.add_argument('-b', '--batch_size', '--batch-size', type=int, default=None,
                        help='Number of volumes loaded at a time when computing the '
                             'projection images (default: chosen automatically for each '
                             'run from its volume size).')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='Recompute the projection images of each kept run even if '
                             'they already exist.')
    parser.add_argument('-l', '--live', action='store_true',
                        help='Keep running: preprocess each run as soon as '
                             'transfer_scape_recordings.py has finished moving it to the '
                             'server (no selection prompt), and combine the samples once no '
                             'new run has appeared for --live-timeout hours, or on Ctrl+C.')
    parser.add_argument('--live-timeout', type=float, default=LIVE_TIMEOUT_HOURS,
                        help='Hours without a new run after which live mode stops '
                             f'(default: {LIVE_TIMEOUT_HOURS:g}).')
    args = parser.parse_args()
    preprocess_scape_exp(args.exp_folder, batch_size=args.batch_size, overwrite=args.overwrite,
                         live=args.live, live_timeout=args.live_timeout)
