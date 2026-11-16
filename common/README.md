# common

Helpers shared by several pipelines, installed as the `common` package (see the root README):

- `paths.py`: data locations (`DATA_PATH`, `SCAPE_PATH`, `FLYTRACK_PATH`, `MASTER_REFERENCE_PATH`).
  Change them here, not in the scripts.
- `masks.py`: keep the pixels of images and videos inside a mask (flattened), put them back, merge
  and intersect masks.
- `nmf.py`: non-negative matrix factorisation of flattened frames into temporal and spatial
  components, with the frame normalisation and selection used before it.
- `plotting.py`: plot temporal and spatial components.
- `dialogs.py`: folder-selection dialogs listing folders newest first (`ask_directory`,
  `sort_dialog_descending`).
- `preanalysis.py`: the `00_preanalysis` plots of a long recording, made by both preprocessing
  scripts: grids of the activity traces and of the max/median projections, and the hours after
  puparium formation (from the folder name, `get_time_0`, `get_hours_apf`).
- `eclosion.py`: pick the recording in which the fly eclosed (saved in
  `00_preanalysis/eclosion.txt`), used by the analyses in `analysis/`.
- `power_spectrum.py`: frequency analysis of the dF/F activity traces before eclosion, in
  `00_frequency_analysis`: power spectra (Welch, also normalized by the total power) overlaid
  and colored by hours APF, the power per frequency band against hours APF, and a heatmap of
  the normalized spectra relative to their mean (run by `analysis/flytrack/` and
  `analysis/scape/`).

FlyTrack-specific loaders (mkv lists, timestamps, ROIs) are in `flytrack/flytrack_io.py`.
