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

- `ct_muscle_names.py`: fixes for the known typos of the raw label names (`ID_TO_NAME_FIXES`), and the merging of the parts of a muscle (`_p1`, `_p2`, ...) into one muscle, except the accessory heart muscles (`KEEP_PARTS_SEPARATE`), whose p1 and p2 stay separate.
- `export_ct_muscles.py`: export the labels of a WebKnossos export at the voxel size of the master reference (1.5 um), in array order (z, y, x) of the WebKnossos axes, with the label names, groups and voxel size (`ct_labels_1p5um.tif` / `.json`), for `master_reference_3d/register_ct_labels.py`. E.g. `python export_ct_muscles.py labels_folder /mnt/labserver/data/MA/SCAPE/master_reference/ct_alignment`.

The CT scan of the leg muscles (front and middle legs of one side, wild-type pupa) is in `/mnt/labserver/data/MA/CT_scans_PIXE/2026-06-16_Webknossos_muscle_leg_tracing_pupa_annotations/RF-RM Leg muscles in pupa/labels`: isotropic voxels of 0.6995 um (the `<scale>` of the `.nml`, in nm; `export_ct_muscles.ct_voxel_size_um`), one label per traced structure, grouped by leg and segment (`RF_femur`, `RM_thorax`, ...; nested subgroups like `FeTi_extensor`), with `*_non-muscles` groups for nerves etc. Loading the full export takes about 3 min.

The CT labels are registered to the master reference by `master_reference_3d/register_ct_labels.py`.
