"""
Data locations, in one place. The lab server is mounted at `/mnt/labserver` on the analysis
machine (the same share is also mounted at `/mnt/upramdya_data`, which the SCAPE PC uses; see
`scape/transfer_scape_recordings.py`, which runs there and keeps its own paths).

DATA_PATH: the lab server's data folder for this project.
SCAPE_PATH: SCAPE experiments (one folder per experiment, with `sample#_run#` folders).
FLYTRACK_PATH: FlyTrack recordings.
MASTER_REFERENCE_PATH: the master reference built by `master_reference_3d/` (hemibodies, muscle
    clusters, registration rounds, and the final `master_mean.tif` / `master_segments.tif`).
"""

from pathlib import Path

DATA_PATH = Path('/mnt/labserver/data/MA')
SCAPE_PATH = DATA_PATH / 'SCAPE'
FLYTRACK_PATH = DATA_PATH / 'Development_project'
MASTER_REFERENCE_PATH = SCAPE_PATH / 'master_reference'
