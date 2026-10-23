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
| 4. Build a master 3D image (master muscle map, later registered to the CT labels) | `master_reference_3d/` | working |
| 5. Align 3D videos to the master | `master_reference_3d/` (`register_recording.py`) | working |
| 6. Extract time traces from a 3D video with the aligned master muscle map | `master_reference_3d/` | planned |
| 7. Align a 3D image to the matching 2D FlyTrack image | `traces_2d/` | planned |
| 8. Extract time traces from a 2D video with the aligned, flattened 3D image and muscle map | `traces_2d/` | exploratory (NMF) |

`common/` holds the helpers shared by several pipelines (data paths, masking, NMF, plotting).
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
