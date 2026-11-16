# master_reference_3d

Pipelines 4-6 (run with the conda environment `scapeio`):
- 4: build the master reference, a master 3D image and muscle mask labelled by leg segment, from several SCAPE recordings (`master_mean_generation/`), and identify the muscles in it (`muscle_segmentation/`, in progress),
- 5: register a 3D video to the master, and map the master's labels back onto it (`register_recording.py`),
- 6 (planned): extract the time trace of each label from the 3D video, with the labels mapped onto it.

Self-contained from the projection images on: it starts from `{sample#_run#}/analysis/projection_images.h5`, computed in the `scape` folder (`scape/preprocess_scape_exp.py`), and imports nothing from the other pipeline folders (only the shared `common` package). Videos to check the master are made in the top-level `visualization/` folder.

### Folders

- `registration/`: the shared modules: splitting a recording into hemibodies, the automatic muscle mask, and the registration of a hemibody to a template, leg segment by leg segment. See its README.
- `master_mean_generation/`: builds the master once, from the median projections of the recordings in `recordings.txt`, in 3 registration rounds (`build_master_mean.py`). See its README, and `pipeline.md` for the method.
- `muscle_segmentation/`: work in progress: splitting the master muscle mask into individual muscles, and registering the CT-scan muscle labels to the master to identify them. See its README.

### Registering a new recording

`python register_recording.py sample#_run#_folder` registers a recording to the master (`master_segments.tif`, saved by `master_mean_generation/build_master_mean.py`) and brings a label map of the master back onto the recording, in the recording's own frame (`{sample#_run#}/analysis/registration_to_{master folder}/master_labels.tif`, same shape and axis order as the projections in `projection_images.h5`), so the recording can be analysed without being deformed. The default label map is the master's leg segments; a map of identified muscles on the master grid can be given with `--label_map` (and `--label_names`). The master image (`master_mean.tif`) is brought back onto the recording too (`master_mean_on_recording.tif`, same grid as `master_labels.tif`). The recording's median projection is used, as for the master. The modules are imported from `registration/`.
