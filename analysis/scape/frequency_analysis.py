"""
Frequency analysis of a SCAPE long recording, before eclosion.

Takes an experiment folder already preprocessed with `scape/preprocess_scape_exp.py` (one
`sample#_run#` subfolder per run, the folder name ending with the puparium formation time,
`..._MM-DD_HHhMM`). The runs' saved activity traces are loaded (see
`preprocess_scape_exp.load_activity_traces()`, no raw data loaded; `_short` runs are left out),
and only the runs before the one in which the fly eclosed are analysed (see `common.eclosion`:
the eclosion run is read from `00_preanalysis/eclosion.txt`, or asked for and saved there if not
picked yet). Saved in `<exp_folder>/00_frequency_analysis/` (see `common.power_spectrum`):

- `power_spectrum_log.png` / `power_spectrum_linear.png`: the power spectra of the dF/F of every
  run, overlaid and colored by hours after puparium formation (hAPF), with a log or linear
  frequency axis
- `power_spectrum_log_normalized.png`: the same, each divided by the run's total power
- `band_power.png`: the power in each frequency band against hAPF
- `power_spectrum_heatmap_relative.png`: hAPF against frequency, the normalized power relative
  to its mean over the runs

Each plot is also copied to `_analysis/frequency_analysis/` next to the experiment folder (in the SCAPE folder),
named with the experiment date (`power_spectrum_log_YYYYMMDD.png`), to compare experiments,
and the experiment is added to `_analysis/experiments.csv` (with its eclosion and a comment).

Run with the conda environment `scapeio`. Example usages from the terminal:
    python frequency_analysis.py
    python frequency_analysis.py path/to/exp_folder
    python frequency_analysis.py path/to/exp_folder --pick-eclosion
From python:
    from frequency_analysis import frequency_analysis
    frequency_analysis('path/to/exp_folder')
"""

import argparse
import sys
from pathlib import Path

# the SCAPE preprocessing is a folder of scripts, not a package
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scape'))
import preprocess_scape_exp  # noqa: E402

from common.dialogs import ask_directory  # noqa: E402
from common.paths import SCAPE_PATH  # noqa: E402
from common.power_spectrum import FREQUENCY_FOLDER, save_frequency_analysis  # noqa: E402
from common.preanalysis import PREANALYSIS_FOLDER  # noqa: E402


def frequency_analysis(exp_folder: str | Path | None = None,
                       pick_eclosion: bool = False) -> str | None:
    """
    Save the frequency analysis of a SCAPE long recording, before eclosion (see module
    docstring).

    Parameters
    ----------
    exp_folder : str | Path | None
        The experiment folder (or its `00_preanalysis` or `00_frequency_analysis` folder). If
        None, a dialog opens to select it.
    pick_eclosion : bool
        Ask for the eclosion run even if it was already picked.

    Returns
    -------
    str | None
        The output folder, or None if nothing was saved.
    """
    if exp_folder is None:
        exp_folder = ask_directory(SCAPE_PATH, 'Select a preprocessed SCAPE long recording')
        if not exp_folder:
            print('No folder selected. Exiting.')
            return None
    exp_folder = Path(exp_folder).resolve()
    if exp_folder.name in (PREANALYSIS_FOLDER, FREQUENCY_FOLDER):
        exp_folder = exp_folder.parent
    traces = preprocess_scape_exp.load_activity_traces(exp_folder)
    return save_frequency_analysis(exp_folder, traces, pick_eclosion)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Frequency analysis of a SCAPE long recording, before eclosion.')
    parser.add_argument('exp_folder', nargs='?', default=None,
                        help='Experiment folder (opens a dialog if omitted)')
    parser.add_argument('-e', '--pick-eclosion', dest='pick_eclosion', action='store_true',
                        help='Ask for the eclosion run even if it was already picked')
    args = parser.parse_args()
    frequency_analysis(args.exp_folder, args.pick_eclosion)
