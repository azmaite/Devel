# scape

Pipeline 2: preprocess 3D muscle videos recorded on SCAPE. Run with the conda environment
`scapeio`.

- `transfer_scape_recordings.py`: runs on the SCAPE PC during acquisition, and moves each finished
  recording to the server (so the SCAPE PC disk does not fill up). Each run is renamed on the SCAPE
  PC as soon as it finishes (`sample1_run##_YYYYMMDD_HHMM`, numbered in order), so the acquisition
  software can't reuse its name while it waits to be moved. It stops by itself after 3 h without
  new recordings or transfers (`--timeout`). It keeps its own paths and imports
  nothing from this repository.
- `preprocess_scape_exp.py`: for an experiment folder, fix the folder structure of each
  `sample#_run#` recording, compute its projection images (deskewed min/max/std projections and
  activity, `analysis/projection_images.h5` and OME-TIFFs; plus the median projection,
  `median_proj`, exact for runs with fewer than 75 volumes, else from 50 evenly spaced volumes,
  as it needs all its volumes at once),
  combine the runs of each sample, and
  convert and deskew the high-resolution `_HR` volumes. For long recordings (folder name ending in the
  puparium formation time, `..._MM-DD_HHhMM`, optionally followed by e.g. `_FAIL`) the runs get
  no individual activity plot; instead all (non-`_short`) runs are summarised in
  `exp_folder/00_preanalysis/` (grids of the activity traces and of the max and median
  projections, flattened over depth, see `common/preanalysis.py`; the frequency analysis is in
  `analysis/scape/`). With
  `-l`/`--live` it runs alongside
  `transfer_scape_recordings.py`, preprocessing each run as soon as it is transferred, and
  combines the runs once no new run has appeared for 3 h (or on Ctrl+C). E.g.
  `python preprocess_scape_exp.py exp_folder [-b N] [-o]`.
- `register_min_proj.py`: rigidly register the min projection of one run to another run of the
  same sample.
- `edit_labels_napari.py`: open a label tif in napari as an editable Labels layer next to the
  recording's projection.
- `segment_muscles_activity.py`: naive split of leg segments (front femur and tibia, middle
  tibia; both sides) into units active together, from long 10-min runs: (1) `extract` saves the
  voxels of each segment (master leg segments mapped onto the run via the short run's
  `master_registration_median`) for all volumes (`analysis/segment_voxel_traces.h5`, ~1.5 h per
  run); (2) `segment` pools the runs of a fly, selects the most different active volumes, and
  groups the voxels by seed correlation expansion (`00_muscle_activity_segmentation/` in the
  experiment folder). Used to check the master muscle split and its CT match, and to look for
  motor units.
- `scape_utils.py`: shared SCAPE functions (folder structure, voxel size, projections, batch size,
  combining runs).

The projection images are the input of `master_reference_3d/`.
