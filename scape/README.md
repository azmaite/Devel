# scape

Pipeline 2: preprocess 3D muscle videos recorded on SCAPE. Run with the repository's uv
environment (`uv run python ...`; Python 3.12 with scapeio's `ozx` extra, needed for `.ozx`
files).

Warning: in napari's default 3D view (depth axis `'towards'`, napari >= 0.6), the deskewed
OME-TIFFs look left-right mirrored, although the data are not. Set napari's depth axis to
`'away'`; see "Warning: napari 3D views can look mirrored" in the root `README.md`.

Raw data: a run is either thousands of `.dat` spool files (`sample#_run#/sample#_run#/`, with
`_info.mat` and `_stim_data.bin` next to them) or a single lossless `.ozx` archive
(`sample#_run#/sample#_run#.ozx`, ~2.5x smaller, made by `scapeio convert` from the `ozx`
branch of scapeio, Python >= 3.12; see `docs/ozx.md` there). `scape_utils.RawRecording` reads
either, preferring the `.ozx`, which is ~5x faster to read from the server. For runs with
754-plane volumes, convert with `--shard-volumes 10` (the default of 400 volumes per shard
makes ~70 GB shards). A `.ozx` volume has no flyback frames, so it has 2 planes fewer than the
same volume from `.dat` files: recompute all projections of an experiment after converting it,
so the runs of a sample can still be combined.

- `transfer_scape_recordings.py`: runs during acquisition (on this machine, with the SCAPE PC
  mounted at `/mnt/scapepc`), and converts each finished recording to a single `.ozx` on the
  server, checks it, then deletes the run on the SCAPE PC (so its disk does not fill up); ~50 min
  per 220 GB run, 10 volumes per shard. A run that fails to convert twice is moved as `.dat`
  files instead; `--no-convert` always moves the files. Each run is renamed on the SCAPE
  PC as soon as it finishes (`sample1_run##_YYYYMMDD_HHMM`, numbered in order), so the acquisition
  software can't reuse its name while it waits to be converted. It stops by itself after 3 h
  without new recordings or transfers (`--timeout`). It keeps its own paths and imports nothing
  from this repository, but needs scapeio's `ozx` extra: run it with `uv run python ...`.
- `preprocess_scape_exp.py`: for an experiment folder, fix the folder structure of each
  `sample#_run#` recording, compute its projection images (deskewed min/max/std projections and
  activity, `analysis/projection_images.h5` and OME-TIFFs; plus the median projection,
  `median_proj`, exact for runs with fewer than 75 volumes, else from 50 evenly spaced volumes,
  as it needs all its volumes at once; a stack over 8 GB is done one x-section per pass, to limit
  RAM; all computed on the raw volumes, and only the final projections deskewed),
  combine the runs of each sample, and
  convert and deskew the high-resolution `_HR` volumes. For long recordings (folder name ending in the
  puparium formation time, `..._MM-DD_HHhMM`, optionally followed by e.g. `_FAIL`) each run still
  gets its own activity plot, and all (non-`_short`) runs are also summarised in
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
