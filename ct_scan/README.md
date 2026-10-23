# ct_scan

Pipeline 3: 3D labels drawn in WebKnossos (e.g. the leg muscles of a CT scan of a pupa). Run with
the uv environment (`.venv`, which has `wkw`).

- `webknossos.py`: load a WebKnossos volume-annotation export as a volume of segment ids (alone or
  placed in the full frame of the dataset it was drawn on), with the segment names and the groups
  (folders) they are in.
- `webknossos_labels_to_tif.py`: convert labels drawn in WebKnossos on a SCAPE projection to a
  uint32 label tif that napari opens as a Labels layer (axis order of the OME-TIFF projections,
  z, y, x), with a json of the label names. E.g.
  `python webknossos_labels_to_tif.py run_folder labels_folder out_name`.

Planned: register the CT-scan labels to the master reference of `master_reference_3d/`.
