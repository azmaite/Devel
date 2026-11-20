# Devel

Repository for the motor development project: long-term fluorescence recordings of muscles in
muscle-GCaMP *Drosophila* pupae. Started on 2025-12-03 by Maite Azcorra Sedano (Neuroengineering
Laboratory, EPFL). Under active development.

Two types of raw data:
- 2D videos from Victor Stimplin's **FlyTrack**, saved as .mkv videos where each saved RGBA frame
  packs 4 real frames (one per channel).
- 3D videos from Jasper Phelps's **SCAPE**, saved as .dat volumes, loaded with `scapeio`; they
  need deskewing.

## Pipelines and folders

| Pipeline | Folder | Status |
|---|---|---|
| 1. Preprocess 2D FlyTrack videos | `flytrack/` | working |
| 2. Preprocess 3D SCAPE videos | `scape/` | working |
| 3. Preprocess 3D labels of a CT scan (WebKnossos) | `ct_scan/` | label loading working |
| 4. Build a master 3D image (master muscle map, later registered to the CT labels) | `master_reference_3d/master_mean_generation/` (muscle identities: `master_reference_3d/muscle_segmentation/`) | working (muscle identities in progress) |
| 5. Align 3D videos to the master | `master_reference_3d/` (`register_recording.py`) | working |
| 6. Extract time traces from a 3D video with the aligned master muscle map | `master_reference_3d/` | planned |
| 7. Align a 3D image to the matching 2D FlyTrack image | `traces_2d/` | planned |
| 8. Extract time traces from a 2D video with the aligned, flattened 3D image and muscle map | `traces_2d/` | exploratory (NMF) |

`common/` holds the helpers shared by several pipelines (data paths, masking, NMF, plotting, the
long-recording pre-analysis and frequency analysis).
`analysis/` holds the analyses of preprocessed FlyTrack and SCAPE long recordings (one subfolder
each; e.g. the frequency analysis of the activity before eclosion).
`visualization/` holds the scripts that make videos to check the results (e.g. of the master
reference).
Each folder has a README with its scripts, inputs and outputs.

Data flow: FlyTrack videos -> `flytrack/` (projections, summary videos, ROIs); SCAPE recordings ->
`scape/` (projection images, `analysis/projection_images.h5`) -> `master_reference_3d/`
(hemibodies, muscle masks, master reference; new recordings registered to it, with the master's
labels mapped back onto each recording) -> time traces. CT-scan labels from WebKnossos ->
`ct_scan/` -> (later) registered to the master.

## Setup

The pipeline folders are folders of scripts, run from their own folder; only `common` is a
package. Install the repository once in each environment you use, so `from common import ...`
works everywhere:
- uv environment (`.venv`, used for FlyTrack and WebKnossos): `uv sync` (installs the
  dependencies of `pyproject.toml` and this repository in editable mode).
- conda environment `scapeio` (SCAPE and the master reference): `pip install -e . --no-deps`.

Data locations are set in `common/paths.py`.

## Warning: napari 3D views can look mirrored

Since napari 0.6, the default 3D camera has the depth axis pointing **towards** the viewer
(`'towards'`; before 0.6 it was `'away'`). With that default, a 3D render of our deskewed SCAPE
volumes (ZYX OME-TIFFs, z = 0 at the objective/glass) is the **mirror image** of the real sample,
so left and right are swapped. 2D views, slices and z-projections are not affected.

The data are not mirrored: checked on 2026-11-20 by imaging the asymmetric tip of a fluorescent
tape through the glass on SCAPE, FlyTrack and an epifluorescence scope
(`20261119_test-microscope-mirroring`). The SCAPE z-projection (rows = Y, columns = X, seen from
z = 0) matches the FlyTrack image and the photo by a rotation alone, with no flip. Nothing in
`scapeio` loading/deskewing, `scape/preprocess_scape_exp.py` or the napari OME-TIFF reader
flips an axis.

So:
- Set napari's depth axis to `'away'` in every environment with napari (done in `scapeio`):
  File -> Preferences -> Application -> Depth Axis Orientation -> Away, or
  `napari.settings.get_settings().application.depth_axis_orientation = 'away'` (saved
  per environment). In a script, `viewer.camera.orientation = ('away', 'down', 'right')` does the
  same for one viewer. `scape_view` (scapeio) already sets `'away'` itself.
- When matching SCAPE to FlyTrack (pipeline 7), use the arrays (a z-projection of the ZYX volume)
  rather than what a 3D view shows: both are imaged through the glass, so they match up to
  rotation, scale and translation, never a flip.
- `scape_view -s` (deskewed) probably still shows a mirrored view (it keeps `'away'` but its
  deskewed axes are in a different order); judge left and right from its raw view or from the
  OME-TIFFs.
