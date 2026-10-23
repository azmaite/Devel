# scape

Pipeline 2: preprocess 3D muscle videos recorded on SCAPE. Run with the conda environment
`scapeio`.

- `transfer_scape_recordings.py`: runs on the SCAPE PC during acquisition, and moves each finished
  recording to the server (so the SCAPE PC disk does not fill up). It keeps its own paths and
  imports nothing from this repository.
- `preprocess_scape_exp.py`: for an experiment folder, fix the folder structure of each
  `sample#_run#` recording, compute its projection images (deskewed min/max/std projections and
  activity, `analysis/projection_images.h5` and OME-TIFFs), combine the runs of each sample, and
  convert and deskew the high-resolution `_HR` volumes. E.g.
  `python preprocess_scape_exp.py exp_folder [-b N] [-o]`.
- `register_min_proj.py`: rigidly register the min projection of one run to another run of the
  same sample.
- `edit_labels_napari.py`: open a label tif in napari as an editable Labels layer next to the
  recording's projection.
- `scape_utils.py`: shared SCAPE functions (folder structure, voxel size, projections, batch size,
  combining runs).

The projection images are the input of `master_reference_3d/`.
