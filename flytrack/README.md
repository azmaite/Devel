# flytrack

Pipeline 1: preprocess 2D muscle videos recorded on FlyTrack (.mkv videos where each RGBA frame
packs 4 real frames). Run with the uv environment (`.venv`).

1. `preprocess_flytrack.py`: for each recording, decode the videos and save
   `projection_images.h5` (median, min/max projections, std, per-frame activity), preview images,
   an activity plot and a half-resolution summary video; recordings of one fly are aligned to
   each other and summarised together. Optional motion correction (`-m`). See its docstring for
   the expected folder structures. E.g. `python preprocess_flytrack.py --folder path/to/folder`.
2. `select_manual_roi.py`: draw one free-hand ROI per leg segment on a recording's std image, saved
   in `segment_rois.h5` (also works on SCAPE std images).
3. `get_segmentation_data.py`: ROI-masked, activity-subsampled flattened video data per segment,
   for segmentation (e.g. NMF, see `traces_2d/`).

Helpers: `flytrack_core.py` (constants and the per-recording processing used by
`preprocess_flytrack.py`, plus `tif2mkv()` for raw tif exports), `flytrack_io.py` (loaders for
recordings and their analysis files), `load_dat_files.py` and `load_tif_batches.py` (two raw
formats being tested as replacements for .mkv). `run_flytrack_analysis.ipynb` runs the steps
interactively.
