"""
Script to preprocess a scape experiment
Takes as input the folder for a scape experiment.
Expects the folder to contain subfolders called sample#_run#,
some with _HR in the end (high-resolution single-volume recordings).
First, every _HR folder that still has raw .dat files is converted (not deskewed) to a
single nrrd file (sample#_run#_HR.nrrd, inside the _HR folder, via scapeio.convert),
which is checked before the .dat files are deleted.
Then lists all non-_HR folders, sorted by sample# then run#, each with an index, and asks
which indices to keep (supports ranges, e.g. '2-5,7,9-12').
Then, sample by sample, goes run by run and checks:
- if the folder structure is correct (raw data inside a sub-subfolder)
- if the max/min/std projections have been calculated already
If either of these have not been done, it does it.
Once all runs for a sample are done, it creates a new folder called sample#_combined
and combines all the projections for that sample, before moving on to the next sample.

Can be run from the terminal (`python preprocess_scape_exp.py [exp_folder]`) or imported
and called as a function. If `exp_folder` is not given, a folder-selection dialog opens,
starting in `scape_utils.MAIN_PATH`.

convert_HR_folders: convert all sample#_run#_HR subfolders of an experiment folder to a
    single (not deskewed) nrrd each, then delete their raw .dat files.
preprocess_scape_exp: main function, converts the HR folders (see convert_HR_folders), then
    lists and preprocesses all other sample#_run# subfolders of an experiment folder, and
    combines projections per sample.
"""

import argparse
import os
import re
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import scapeio

import scape_utils


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


def _hr_scan_path(hr_path: Path) -> Path:
    """
    Get the folder scapeio should treat as the scan (the folder containing the
    raw data, with a matching `_info.mat` file next to it) for a
    `sample#_run#_HR` folder, which can be in either of two layouts:
    - as acquired: raw data directly in `hr_path`, with `_info.mat` and
      `_stim_data.bin` next to it in the experiment folder -> `hr_path`
    - moved inside (as done by `scape_utils.check_folder_structure` for normal
      runs): raw data in `hr_path / hr_path.name`, with `_info.mat` and
      `_stim_data.bin` in `hr_path` -> `hr_path / hr_path.name`

    Parameters
    ----------
    hr_path : Path
        Path to a `sample#_run#_HR` folder.

    Returns
    -------
    Path
    """
    nested_path = hr_path / hr_path.name
    if nested_path.is_dir() and (hr_path / f'{hr_path.name}_info.mat').exists():
        return nested_path
    return hr_path


def convert_HR_folders(exp_folder: str | Path) -> None:
    """
    Convert every high-resolution (single-volume) recording in an experiment
    folder, i.e. every `sample#_run#_HR` subfolder, to a single nrrd file with
    `scapeio.convert` (not deskewed), then delete its raw .dat files.

    The nrrd is saved by scapeio inside the HR folder as `sample#_run#_HR.nrrd`
    (with the raw voxel size in its header), so `scapeio.load_scan(hr_path, colors=1)`
    loads it afterwards instead of the .dat files (and can then be deskewed with
    `scapeio.operations.deskew_scan`). Before deleting the .dat files, the nrrd is
    checked by loading it back and making sure it's a non-empty 3D volume, and by
    `scapeio.delete_dat_files` checking it's big enough to hold the full scan.
    HR folders with no .dat files left (already converted) are skipped. Works
    whether or not the raw data were already moved inside the HR folder (see
    `_hr_scan_path`); if so, the nrrd is saved next to the raw data, in
    `sample#_run#_HR/sample#_run#_HR/`.

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
        hr_path = str(_hr_scan_path(hr_path)) # string needed for scapeio
        if not scapeio.io.list_dat_files(hr_path):
            print(f'No .dat files found in {hr_path}, skipping (already converted?)')
            continue

        nrrd_path = scapeio.io.get_data_copy_name(hr_path, copy_format='nrrd', colors=1,
                                                  check_exists=True)
        if nrrd_path is None:
            print(f'Converting {hr_path} to nrrd...')
            nrrd_path = scapeio.convert(hr_path, output_format='nrrd', colors=1)
        else:
            print(f'nrrd already found for {hr_path}, checking it...')

        # check the nrrd loads back fine before deleting the raw data
        volume = scapeio.load_scan(hr_path, colors=1, try_loading_copies=['nrrd'])
        if volume.ndim != 3 or volume.size == 0 or not volume.any():
            raise RuntimeError(f'{nrrd_path} loaded back as an empty or non-3D volume '
                               f'(shape {volume.shape}). Raw .dat files were NOT deleted.')
        print(f'Checked {nrrd_path} (shape {volume.shape}, {volume.dtype}).')
        del volume

        scapeio.delete_dat_files(hr_path, verify_copy_exists='nrrd', fake=False)


def preprocess_scape_exp(exp_folder: str | Path | None = None) -> None:
    """
    Preprocess all SCAPE recordings in an experiment folder.

    First converts all `sample#_run#_HR` subfolders to (not deskewed) nrrd and
    deletes their raw .dat files (see `convert_HR_folders`). Then lists all `sample#_run#` subfolders (ignoring any ending in `_HR`), sorted
    by sample# then run#, each with an index, and asks which indices to keep
    (supports ranges, e.g. `'2-5,7,9-12'`). Then, sample by sample, for each
    remaining run checks (and if needed fixes/computes) the folder structure
    and projection images via `scape_utils`, and once all of a sample's runs
    are done, combines their projections into a new `sample#_combined` folder
    before moving on to the next sample.

    Parameters
    ----------
    exp_folder : str | Path | None
        Path to the experiment folder containing the `sample#_run#` subfolders.
        If None, a folder-selection dialog opens starting in `scape_utils.MAIN_PATH`.

    Returns
    -------
    None
    """
    if exp_folder is None:
        # some native (e.g. GTK/KDE portal) directory choosers ignore `initialdir` and
        # default to the process's current working directory instead, so change into
        # MAIN_PATH before opening the dialog as well
        original_cwd = os.getcwd()
        if os.path.isdir(scape_utils.MAIN_PATH):
            os.chdir(scape_utils.MAIN_PATH)
        else:
            print(f'Warning: MAIN_PATH ({scape_utils.MAIN_PATH}) not found; dialog may not '
                  'start there.')
        root = tk.Tk()
        root.withdraw()
        selected = filedialog.askdirectory(initialdir=scape_utils.MAIN_PATH, title='Select experiment folder')
        root.destroy()
        os.chdir(original_cwd)
        if not selected:
            print('No folder selected. Exiting.')
            return
        exp_folder = selected
    exp_folder = Path(exp_folder)

    scape_paths = sorted((p for p in exp_folder.iterdir()
                          if p.is_dir() and p.name.startswith('sample') and '_run' in p.name
                          and not p.name.endswith('_HR')),
                         key=_sample_run_sort_key)

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
            scape_utils.check_folder_structure(scape_path)
            h5_path = scape_path / scape_utils.ANALYSIS_PATH_NAME / scape_utils.PROJECTION_FILE_NAME
            if h5_path.exists():
                print(f'Projections already found for {scape_path.name}.')
            else:
                print(f'Projections not found for {scape_path.name}. Computing them...')
                scape_utils.get_projection_images(scape_path)

        scape_utils.combine_sample_projections(exp_folder, sample_name, run_paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess a SCAPE experiment folder.')
    parser.add_argument('exp_folder', nargs='?', default=None,
                        help='Path to the experiment folder. If omitted, a folder-selection '
                             f'dialog opens starting in {scape_utils.MAIN_PATH}.')
    args = parser.parse_args()
    preprocess_scape_exp(args.exp_folder)
