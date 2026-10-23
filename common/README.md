# common

Helpers shared by several pipelines, installed as the `common` package (see the root README):

- `paths.py`: data locations (`DATA_PATH`, `SCAPE_PATH`, `FLYTRACK_PATH`, `MASTER_REFERENCE_PATH`).
  Change them here, not in the scripts.
- `masks.py`: keep the pixels of images and videos inside a mask (flattened), put them back, merge
  and intersect masks.
- `nmf.py`: non-negative matrix factorisation of flattened frames into temporal and spatial
  components, with the frame normalisation and selection used before it.
- `plotting.py`: plot temporal and spatial components.

FlyTrack-specific loaders (mkv lists, timestamps, ROIs) are in `flytrack/flytrack_io.py`.
