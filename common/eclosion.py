"""
The recording of a long recording in which the fly ecloses, shared by the FlyTrack and SCAPE
analyses (`analysis/flytrack/`, `analysis/scape/`). After eclosion the activity is not pupal
anymore, so the analyses only use the recordings before it.

The eclosion recording is picked once by the user (see `pick_eclosion()`) and saved in
`<experiment folder>/00_preanalysis/ECLOSION_FILE`, as the path of the recording folder relative
to the experiment folder (e.g. `20261101_023209` or `sample1_run26_20261025_1525`), or `none`
if the fly did not eclose during the recording. The file can also be written or edited by
hand.

pick_eclosion: list the recordings, ask in which one the fly eclosed, and save it.
load_eclosion: read the saved eclosion recording.
get_eclosion_index: index of the eclosion recording in a list of traces, asking for it if it
    was not picked yet.
"""

import os
from pathlib import Path

from common.preanalysis import PREANALYSIS_FOLDER

ECLOSION_FILE = 'eclosion.txt'  # in PREANALYSIS_FOLDER
NO_ECLOSION = 'none'            # saved when the fly did not eclose during the recording


def _eclosion_path(exp_folder: str | os.PathLike) -> Path:
    """
    Path of the eclosion file of an experiment.
    """
    return Path(exp_folder) / PREANALYSIS_FOLDER / ECLOSION_FILE


def pick_eclosion(exp_folder: str | os.PathLike,
                  traces: list[dict]) -> str | None:
    """
    List the recordings of a long recording and ask in which one the fly eclosed, then save it
    in the eclosion file (see module docstring).

    Parameters
    ----------
    exp_folder : str or os.PathLike
        The experiment (main) folder.
    traces : list of dict
        Its recordings' traces (see `common.preanalysis`), in time order, each with a 'name'
        (recording folder relative to `exp_folder`) and a 'title' (e.g. '72 hAPF').

    Returns
    -------
    str or None
        The eclosion recording's name, `NO_ECLOSION`, or None if cancelled (nothing saved).
    """
    print(f'\nRecordings of {exp_folder}:')
    for i, tr in enumerate(traces):
        print(f'  [{i + 1}] {tr["name"]}  ({tr["title"]})')
    while True:
        answer = input("\nIn which recording did the fly eclose? Enter its number, 'h' + its "
                       "hAPF (e.g. 'h96'), 'n' if it did not eclose, or 'q' to cancel: "
                       ).strip().lower()
        if answer in ('q', 'quit', ''):
            return None
        if answer in ('n', 'no', NO_ECLOSION):
            name = NO_ECLOSION
            break
        index = None
        if answer.startswith('h') and answer[1:].isdigit():
            hours = int(answer[1:])
            index = next((i for i, tr in enumerate(traces)
                          if tr.get('hours') is not None and tr['hours'] >= hours), None)
        elif answer.isdigit() and 1 <= int(answer) <= len(traces):
            index = int(answer) - 1
        if index is not None:
            name = traces[index]['name']
            break
        print(f"Not understood: '{answer}'.")

    path = _eclosion_path(exp_folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(name + '\n')
    print(f'Saved the eclosion recording ({name}) in {path}')
    return name


def load_eclosion(exp_folder: str | os.PathLike) -> str | None:
    """
    Read the eclosion recording of a long recording from its eclosion file.

    Parameters
    ----------
    exp_folder : str or os.PathLike
        The experiment (main) folder.

    Returns
    -------
    str or None
        The eclosion recording's folder relative to `exp_folder` (an absolute path in the file
        is made relative), `NO_ECLOSION`, or None if there is no eclosion file.
    """
    path = _eclosion_path(exp_folder)
    if not path.is_file():
        return None
    name = path.read_text().strip()
    if os.path.isabs(name):
        name = os.path.relpath(name, exp_folder)
    return name


def get_eclosion_index(exp_folder: str | os.PathLike,
                       traces: list[dict],
                       pick: bool = False) -> int | None:
    """
    Index of the eclosion recording in `traces`, from the eclosion file, or asked for (and
    saved, see `pick_eclosion()`) if there is none, it doesn't match any recording, or `pick`.

    Parameters
    ----------
    exp_folder : str or os.PathLike
        The experiment (main) folder.
    traces : list of dict
        Its recordings' traces, in time order, each with a 'name' (see `pick_eclosion()`).
    pick : bool
        Ask even if the eclosion recording was already picked.

    Returns
    -------
    int or None
        Index of the eclosion recording (the recordings before it are `traces[:index]`), or
        `len(traces)` if the fly did not eclose, or None if cancelled.
    """
    names = [tr['name'] for tr in traces]
    name = None if pick else load_eclosion(exp_folder)
    if name is not None and name != NO_ECLOSION and name not in names:
        print(f'Eclosion recording {name} (in {_eclosion_path(exp_folder)}) is not among the '
              f'preprocessed recordings.')
        name = None
    if name is None:
        name = pick_eclosion(exp_folder, traces)
        if name is None:
            return None
    else:
        print(f'Eclosion recording: {name} (from {_eclosion_path(exp_folder)})')
    return len(traces) if name == NO_ECLOSION else names.index(name)
