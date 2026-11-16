# master_mean_generation

Builds the master reference once (run with the conda environment `scapeio`): a master 3D image (`master_mean.tif`) and master muscle mask labelled by leg segment (`master_segments.tif` / `.json`), from the median projections of the SCAPE recordings listed in `recordings.txt`, in 3 registration rounds. The method is described in `pipeline.md`.

The registration matches SCAPE recordings by the shape of their muscle masks, one leg segment at a time. Intensity-based registration (SPAM, tried before and since removed) does not converge well on the long tube-like muscles: their intensity barely changes when they slide along their own axis. Matching the shape of the muscle masks (their ends and outlines) avoids this.

### Running it

1. Edit `recordings.txt` if needed: one `sample#_run#` folder per line (relative to `SCAPE_PATH`), optionally followed by `L` or `R` to use only that hemibody.
2. `python build_master_mean.py` (in `tmux` over ssh: it takes many hours). Settings (output folder `OUT_PATH`, reference hemibody, number of rounds) are at the top of the script. Each step is skipped if its output exists, so it can be rerun after a crash.

Outputs, in `OUT_PATH` (`master_reference/master_median`): `hemibodies/`, `segment_masks/`, `registered_hemibodies_round1..3/`, and the master itself (`master_mean.tif`, `master_segments.tif`, `master_segments.json`, a copy of `recordings.txt`). New recordings are registered to it with `../register_recording.py`.

The leg segments of the reference hemibody (the round-1 template) are not labelled again: they are transferred from the hand labelling of its minimum-projection clusters (`master_reference/segment_masks/{recording}_segment_ids.json`, from an earlier min-projection master). The reference is split with the same mirror plane as then, so its median hemibody is on the same grid.

### Scripts

- `build_master_mean.py`: runs all the steps (see above).
- `registration_round.py`: one registration round: registers hemibodies to a template (`../registration/deformation.py`) and averages the normalised registered images into a master image and majority-vote muscle mask; `template_from_round` gives the consensus template of the next round.
- `recordings.txt`: the recordings used.
- `pipeline.md`: the method.

The splitting into hemibodies, the muscle masks and the registration itself are in `../registration/`, shared with `../register_recording.py`.
