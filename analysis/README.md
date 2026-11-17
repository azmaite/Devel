# analysis

Analyses of long recordings (one pupa recorded many times over several days) that were already
preprocessed by `flytrack/preprocess_flytrack.py` or `scape/preprocess_scape_exp.py`. One
subfolder per setup; the shared code is in `common/` (`power_spectrum.py`, `eclosion.py`,
`preanalysis.py`). Each script saves its results in its own `00_*` folder in the experiment
folder, next to the preprocessing's `00_preanalysis`.

The analyses only use the recordings before eclosion: the first time a script runs on an
experiment, it lists the recordings with their hours after puparium formation (hAPF) and asks in
which one the fly eclosed (`n` if it did not). The answer is saved in
`00_preanalysis/eclosion.txt` (the recording folder, relative to the experiment folder, or
`none`) and reused afterwards; `-e`/`--pick-eclosion` asks again.

- `flytrack/frequency_analysis.py` (uv environment, `.venv`) and `scape/frequency_analysis.py`
  (conda environment `scapeio`): frequency analysis of the activity traces (dF/F) before
  eclosion, in `00_frequency_analysis/`:
  - `power_spectrum_log.png` / `power_spectrum_linear.png`: the power spectra of every recording
    overlaid and colored by hAPF, with a log and a linear frequency axis
  - `power_spectrum_log_normalized.png`: the same, each divided by the recording's total power
    (how the power is spread over frequency, whatever the amount of activity)
  - `band_power.png`: the power in 5 frequency bands (0.02-0.1, 0.1-0.5, 0.5-2, 2-8 and
    8-20 Hz, cut at the Nyquist frequency) against hAPF
  - `power_spectrum_heatmap_relative.png`: hAPF against frequency (20 bands per decade), the
    color being each recording's normalized power relative to the mean over the recordings at
    that frequency (red: more of its power there than the average recording, blue: less)

  Each plot is also copied to a common `_analysis/` folder next to the experiment folders (in
  the SCAPE folder, or in `Pupa_muscle_long_recordings/` for FlyTrack), with the experiment
  date added to its name (`power_spectrum_log_YYYYMMDD.png`), so the same plot of every
  experiment is listed together. `_analysis/experiments.csv` lists the analysed experiments
  (sorted by date) with their eclosion recording and hAPF and a `comment` column to fill in by
  hand; an experiment is added (or its eclosion updated, keeping the comment) each time it is
  analysed.

  E.g. `python frequency_analysis.py path/to/experiment_folder`.
