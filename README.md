# Devel
Repository for motor development project
Started on 2025-12-03 by Maite Azcorra Sedano (Neuroengineering Laboratory, EPFL)
Under active development.


Two types of raw datasets, both recordings of muscle-GCaMP Drosophila pupas
- 2d videos collected on Victor Stimplin's FlyTrack, saved as .mkv videos where each saved frame is RGBA but each channel corresponds to 1 real frame
- 3d videos collected on Jasper Phelp's SCAPE, saved as single .dat volumes and loadable using the scape-io module. Needs deskewing.