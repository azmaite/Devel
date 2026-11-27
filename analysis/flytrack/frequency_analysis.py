#!/usr/bin/env python3
"""
Frequency analysis of a FlyTrack long recording, before eclosion.

Takes the main folder of a long recording already preprocessed with
`flytrack/preprocess_flytrack.py` (one subfolder per recording, the folder name
ending with the puparium formation time, ``..._MM-DD_HHhMM``). The recordings'
saved activity traces are loaded (see `preprocess_flytrack.load_activity_traces()`,
no video decoding), and only the recordings before the one in which the fly
eclosed are analysed (see `common.eclosion`: the eclosion recording is read from
``00_preanalysis/eclosion.txt``, or asked for and saved there if not picked
yet). Saved in ``<main>/00_frequency_analysis/`` (see `common.power_spectrum`):

    - ``power_spectrum_log.png`` / ``power_spectrum_linear.png``: the power
      spectra of the dF/F of every recording, overlaid and colored by hours
      after puparium formation (hAPF), with a log or linear frequency axis
    - ``power_spectrum_log_normalized.png``: the same, each divided by the
      recording's total power
    - ``band_power.png``: the power in each frequency band against hAPF
    - ``power_spectrum_heatmap_relative.png``: hAPF against frequency, the
      normalized power relative to its mean over the recordings

Each plot is also copied to ``_analysis/frequency_analysis/`` next to the main folder (in
``Pupa_muscle_long_recordings/``), named with the experiment date
(``power_spectrum_log_YYYYMMDD.png``), to compare experiments, and the
experiment is added to ``_analysis/experiments.csv`` (with its eclosion and a
comment).

Run with the uv environment (``.venv``). Example usages from the terminal:
        python frequency_analysis.py
        python frequency_analysis.py path/to/main
        python frequency_analysis.py path/to/main --pick-eclosion
From python:
        from frequency_analysis import frequency_analysis
        frequency_analysis('path/to/main')
"""

import argparse
import os
import sys
from pathlib import Path

# the FlyTrack preprocessing is a folder of scripts, not a package
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'flytrack'))
import preprocess_flytrack  # noqa: E402

from common.dialogs import ask_directory  # noqa: E402
from common.paths import FLYTRACK_PATH  # noqa: E402
from common.power_spectrum import FREQUENCY_FOLDER, save_frequency_analysis  # noqa: E402
from common.preanalysis import PREANALYSIS_FOLDER  # noqa: E402


def frequency_analysis(main: str = '',
                       pick_eclosion: bool = False) -> str | None:
    """
    Save the frequency analysis of a FlyTrack long recording, before eclosion
    (see module docstring).

    Parameters
    ----------
    main : str, optional
        Main folder of the long recording (or its ``00_preanalysis`` or
        ``00_frequency_analysis`` folder). If empty, a dialog opens to select it.
    pick_eclosion : bool, optional
        Ask for the eclosion recording even if it was already picked.

    Returns
    -------
    str or None
        The output folder, or None if nothing was saved.
    """
    if not main:
        main = ask_directory(FLYTRACK_PATH, 'Select a preprocessed FlyTrack long recording')
        if not main:
            print('No folder selected. Exiting.')
            return None
    main = os.path.abspath(main)
    if os.path.basename(main) in (PREANALYSIS_FOLDER, FREQUENCY_FOLDER):
        main = os.path.dirname(main)
    traces = preprocess_flytrack.load_activity_traces(main)
    return save_frequency_analysis(main, traces, pick_eclosion)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Frequency analysis of a FlyTrack long recording, before eclosion.')
    parser.add_argument('main', nargs='?', default='',
                        help='Main folder of the long recording (opens a dialog if omitted)')
    parser.add_argument('-e', '--pick-eclosion', dest='pick_eclosion', action='store_true',
                        help='Ask for the eclosion recording even if it was already picked')
    args = parser.parse_args()
    frequency_analysis(args.main, args.pick_eclosion)
