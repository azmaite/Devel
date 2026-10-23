# traces_2d

Pipelines 7 and 8 (planned): align a 3D image to the matching 2D FlyTrack image, then extract the
time traces of each muscle from 2D videos, using the aligned and flattened 3D image and master
muscle map (each 2D pixel mixes several overlapping muscles, so the traces need unmixing).

- `auto_segmentation_by_segment.ipynb`: exploratory NMF of muscle activity per leg segment (on
  SCAPE data so far), with the grouping of components into muscles; uses `common.nmf` and
  `common.plotting`.
