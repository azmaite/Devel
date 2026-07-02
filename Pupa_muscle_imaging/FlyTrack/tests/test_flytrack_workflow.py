#!/usr/bin/env python3
"""
End-to-end integration test of the FlyTrack preprocessing/motion-correction flow.

The three tests share a single module-scoped copy of the synthetic fixture
(built by `make_synthetic_test_data.py`) and run, in order, the real workflow a
user would follow:

1. `test_1_preprocess` runs `preprocess_flytrack` on the raw input folder and
   checks that every recording gets its per-recording projection outputs and
   preview images, and that the analysis folder gets `mkv_paths.json` (with size,
   frame count, fps, and group shift per recording) plus the grouped projections.
2. `test_2_motion_correct` runs `motion_correct_flytrack` on that analysis folder
   and checks that each recording is now drift-corrected (a `shifts` dataset, a
   `drift.png`, a motion-corrected summary video, and `motion_corrected: True`).
3. `test_3_preprocess_again` re-runs `preprocess_flytrack` on the analysis folder
   and checks that the already-corrected per-recording outputs are reused (the
   `shifts` survive, i.e. they are not reprocessed) and the grouped projections
   are rebuilt from the corrected images.

The tests are ordered and stateful: they must run top-to-bottom (pytest's default)
because each builds on the previous one's outputs. The workflow's interactive
subfolder/recording pickers are driven by patching `input` to answer ``all``.

The fixture lives on the lab server; if it is not mounted the tests skip.
"""

import os
import sys
import glob
import json
import shutil

import h5py
import numpy as np
import pytest

# make the FlyTrack package importable (this file lives in FlyTrack/tests/)
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_FLYTRACK_DIR = os.path.dirname(_TESTS_DIR)
if _FLYTRACK_DIR not in sys.path:
    sys.path.insert(0, _FLYTRACK_DIR)

from preprocess_flytrack import preprocess_flytrack
from motion_correct_flytrack import motion_correct_flytrack
from flytrack_core import (PROJ_H5_NAME, METADATA_NAME, ACT_SAVE_NAME,
                           DRIFT_SAVE_NAME, MEDIAN_SAVE_NAME, STD_SAVE_NAME,
                           MAX_PROJ_SAVE_NAME, SUMMARY_NAME, SUMMARY_MC_NAME,
                           MKV_PATHS_NAME, GROUPED_H5_NAME, PROJ_DATASETS)

# synthetic fixture built by make_synthetic_test_data.py (on the lab server)
FIXTURE = ('/mnt/labserver/data/MA/Development_project/'
           'Pupa_muscle_long_recordings/_synthetic-test-data/'
           '20260701_test-flytrack-data_prepup_06-29_10h55')

# recording subfolders inside the fixture
SUBFOLDERS = ('001', '002')


@pytest.fixture(scope='module')
def workdir(tmp_path_factory):
    """
    Copy the synthetic fixture to a temp folder shared by the ordered tests.

    Module-scoped so the three workflow steps operate on (and accumulate outputs
    in) the same tree. Skips the whole module if the fixture is not mounted.

    Returns
    -------
    str
        Path to the writable copy of the input folder.
    """

    if not os.path.isdir(FIXTURE):
        pytest.skip(f'Synthetic fixture not available: {FIXTURE}')
    dest = str(tmp_path_factory.mktemp('flytrack') / 'data')
    shutil.copytree(FIXTURE, dest)
    return dest


@pytest.fixture(autouse=True)
def _answer_all(monkeypatch):
    """
    Answer every interactive picker prompt with ``all`` for all tests.
    """

    monkeypatch.setattr('builtins.input', lambda *args, **kwargs: 'all')


def _analysis_folder(workdir: str) -> str:
    """
    Return the single ``*_analysis`` folder created inside `workdir`.
    """

    matches = glob.glob(os.path.join(workdir, '*_analysis'))
    assert len(matches) == 1, f'Expected one analysis folder, got {matches}'
    return matches[0]


def _recording_dirs(workdir: str) -> list[str]:
    """
    Return the recording subfolder paths (``001``, ``002``) inside `workdir`.
    """

    return [os.path.join(workdir, sub) for sub in SUBFOLDERS]


def _num_metadata_rows(rec_dir: str) -> int:
    """
    Return the number of real-frame rows in a recording's `_tif_metadata.h5`.
    """

    meta = glob.glob(os.path.join(rec_dir, '*_tif_metadata.h5'))[0]
    with h5py.File(meta, 'r') as f:
        return int(f['timestamps'].shape[0])


def _check_recording_outputs(rec_dir: str, motion_corrected: bool) -> None:
    """
    Assert a recording folder holds the expected per-recording outputs.

    Checks the projection-images h5 (all required datasets, with the right frame
    count and consistent shapes), the metadata text file, and the preview images.
    When `motion_corrected` is set, the drift outputs (`shifts` dataset,
    `drift.png`, the corrected summary video, and the `motion_corrected: True`
    metadata line) must also be present.

    Parameters
    ----------
    rec_dir : str
        A recording subfolder holding one .mkv.
    motion_corrected : bool
        Whether motion-correction outputs are expected.
    """

    h5_path = os.path.join(rec_dir, PROJ_H5_NAME)
    assert os.path.exists(h5_path)

    n_rows = _num_metadata_rows(rec_dir)
    with h5py.File(h5_path, 'r') as f:
        for key in PROJ_DATASETS:
            assert key in f, f'Missing dataset {key} in {h5_path}'
        frame_count = int(f['frame_count'][()])
        assert frame_count == n_rows, (frame_count, n_rows)
        assert f['act'].shape[0] == frame_count
        median_shape = f['median'].shape
        assert f['max_proj'].shape == median_shape
        assert f['std'].shape == median_shape
        assert ('shifts' in f) == motion_corrected
        if motion_corrected:
            assert f['shifts'].shape == (frame_count, 2)

    # preview images always written
    for name in (ACT_SAVE_NAME, MEDIAN_SAVE_NAME, MAX_PROJ_SAVE_NAME,
                 STD_SAVE_NAME):
        assert os.path.getsize(os.path.join(rec_dir, name)) > 0, name

    # summary video: uncorrected always; corrected only after motion correction
    assert os.path.getsize(os.path.join(rec_dir, SUMMARY_NAME)) > 0
    mc_summary = os.path.join(rec_dir, SUMMARY_MC_NAME)
    drift_png = os.path.join(rec_dir, DRIFT_SAVE_NAME)
    if motion_corrected:
        assert os.path.getsize(mc_summary) > 0
        assert os.path.getsize(drift_png) > 0

    # metadata text records the correction state
    with open(os.path.join(rec_dir, METADATA_NAME)) as f:
        metadata = f.read()
    assert f'motion_corrected: {motion_corrected}' in metadata
    assert f'num_frames: {n_rows}' in metadata


def _check_grouped_outputs(analysis_folder: str, rec_dir: str) -> None:
    """
    Assert the analysis folder holds valid grouped projections and previews.

    Parameters
    ----------
    analysis_folder : str
        The analysis folder holding the grouped outputs.
    rec_dir : str
        A recording folder, used to cross-check the grouped image shape.
    """

    grouped = os.path.join(analysis_folder, GROUPED_H5_NAME)
    assert os.path.exists(grouped)

    with h5py.File(os.path.join(rec_dir, PROJ_H5_NAME), 'r') as f:
        rec_shape = f['median'].shape
    with h5py.File(grouped, 'r') as f:
        for key in ('median', 'max_proj', 'std'):
            assert key in f, key
            assert f[key].shape == rec_shape

    for name in (MEDIAN_SAVE_NAME, MAX_PROJ_SAVE_NAME, STD_SAVE_NAME):
        assert os.path.getsize(os.path.join(analysis_folder, name)) > 0, name


def test_1_preprocess(workdir):
    """
    Preprocess the raw input folder and check per-recording and grouped outputs.
    """

    preprocess_flytrack(workdir)

    analysis_folder = _analysis_folder(workdir)
    rec_dirs = _recording_dirs(workdir)

    # per-recording outputs exist and are not yet motion-corrected
    for rec_dir in rec_dirs:
        _check_recording_outputs(rec_dir, motion_corrected=False)

    # mkv_paths.json has one entry per recording with size/frames/fps/shift
    with open(os.path.join(analysis_folder, MKV_PATHS_NAME)) as f:
        entries = json.load(f)
    assert len(entries) == len(rec_dirs)
    for entry in entries:
        assert os.path.exists(entry['mkv_path'])
        assert entry['size'] == [1024, 576]  # [width, height]
        assert entry['num_frames'] == _num_metadata_rows(
            os.path.dirname(entry['mkv_path']))
        assert entry['fps'] > 0
        assert len(entry['group_shift']) == 2
    # the first recording aligns to itself (zero shift)
    assert entries[0]['group_shift'] == [0, 0]

    _check_grouped_outputs(analysis_folder, rec_dirs[0])


def test_2_motion_correct(workdir):
    """
    Motion-correct the analysis folder and check the drift outputs appear.
    """

    analysis_folder = _analysis_folder(workdir)
    motion_correct_flytrack(analysis_folder)

    for rec_dir in _recording_dirs(workdir):
        _check_recording_outputs(rec_dir, motion_corrected=True)


def test_3_preprocess_again(workdir):
    """
    Re-preprocess the corrected analysis folder: outputs reused, grouped rebuilt.
    """

    analysis_folder = _analysis_folder(workdir)
    grouped = os.path.join(analysis_folder, GROUPED_H5_NAME)

    # grouped file predates this step (written during test_1); it must be rebuilt
    mtime_before = os.stat(grouped).st_mtime_ns

    preprocess_flytrack(analysis_folder)

    rec_dirs = _recording_dirs(workdir)

    # per-recording corrected outputs are reused, not reprocessed: the `shifts`
    # written by motion correction survive (a reprocess would drop them)
    for rec_dir in rec_dirs:
        _check_recording_outputs(rec_dir, motion_corrected=True)

    # the grouped projections are rebuilt (from the corrected images)
    assert os.stat(grouped).st_mtime_ns > mtime_before
    _check_grouped_outputs(analysis_folder, rec_dirs[0])
