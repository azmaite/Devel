## Shape registration of SCAPE hemibodies

Registers SCAPE recordings of pupal leg muscles to each other by the shape of their muscle masks, one leg segment at a time, and averages them into a master reference. The method is described in `pipeline.md`.

Self-contained from the projection images on: it starts from `{sample#_run#}/analysis/projection_images.h5`, computed in the `scape` folder (`scape/MANUAL/preprocess_scape_exp.py`), and imports nothing from other folders of the repository.

Intensity-based registration (SPAM, see `../SPAM_registration`) does not converge well on the long tube-like muscles: their intensity barely changes when they slide along their own axis. Matching the shape of the muscle masks (their ends and outlines) avoids this.

### Running it

`python build_master_reference.py` runs all the steps in order, skipping those already done: hemibodies, muscle clusters, then (once the reference's clusters are assigned to leg segments by hand) the registration rounds. The recordings (paths of the `sample#_run#` folders whose minimum projections are used), the reference hemibody and the number of rounds are set at the top of `build_master_reference.py`. If the reference's segment table is missing, it stops and says what to write. After the last round, the final master reference is saved directly in `master_reference`: `master_mean.tif` (master image) and `master_segments.tif` (master muscle mask labelled by leg segment, names in `master_segments.json`), on the same grid; these are what new recordings are registered to. Existing cluster labels are never recomputed, as they may have been edited by hand.

### Registering a new recording

`python register_recording.py sample#_run#_folder` registers a recording to the final master reference (`master_reference/master_segments.tif`, saved by `build_master_reference.py` after the last round) and brings a label map of the master back onto the recording, in the recording's own frame (`{sample#_run#}/analysis/master_registration/master_labels.tif`, same shape and axis order as the projections in `projection_images.h5`), so the recording can be analysed without being deformed. The default label map is the master's leg segments; a map of identified muscles on the master grid can be given with `--label_map` (and `--label_names`). The recording's hemibodies registered into the master frame are saved too, to improve the master later (add the recording to `RECORDINGS` in `build_master_reference.py` and rerun the rounds).

### Scripts, in the order they are used

- `split_hemibodies.py`: loads the minimum projection of a recording, resamples it to isotropic 1.5 um voxels, fits the left-right symmetry plane and saves the two hemibodies (R mirrored) as OME-TIFFs, with a json of the plane and a png to check it. E.g. `python split_hemibodies.py run_folder [run_folder ...] -o /mnt/labserver/data/MA/SCAPE/master_reference/original_hemibodies`.
- `label_mask_components.py`: computes the automatic muscle mask of hemibody tifs (`muscle_mask.py`) and labels its connected components (muscle clusters), saved as napari Labels tifs next to a scale-1 copy of the image, to assign each cluster to a leg segment by hand (`{recording}_segment_ids.json`, only needed for the reference). E.g. `python label_mask_components.py hemibody.ome.tif [...] -o /mnt/labserver/data/MA/SCAPE/master_reference/segment_masks`.
- `registration_round.py`: one registration round: registers all hemibodies to a template, one leg segment at a time (the head gets no fit of its own), blends the segment transforms into one smooth deformation, and averages the normalised registered images into a master image and majority-vote muscle mask. Round 1 uses the labelled reference hemibody as the template (`master_reference/registered_hemibodies`); later rounds use the master muscle mask of the previous round (`master_reference/registered_hemibodies_round2`, ...). Run by `build_master_reference.py`.
- `register_recording.py`: registers a new recording to the master and brings a master label map back onto it (see above).
- `muscle_mask.py`: the automatic muscle mask (log-contrast to the local background, hysteresis threshold, removal of small or non-elongated components), copied from the mask part of `scape/MANUAL/auto_segment_muscles.py`.
- `register_segments.py`: the registration itself (signed distance maps of the masks, fit restricted to the inside of the template segment, target field of view handled as unknown; rigid then affine), used by `registration_round.py`.
