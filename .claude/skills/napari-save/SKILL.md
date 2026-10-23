---
name: napari-save
description: How to save 3D images and label volumes as tifs that napari opens correctly (right layer type, voxel size, axis order). Use whenever you write a .tif/.ome.tif that will be viewed or edited in napari - projections, segmentation or label masks, WebKnossos/auto-segmentation/registration outputs, binary masks - or when a file opens in napari as an Image instead of Labels, looks rotated/mirrored/shifted/stretched against another layer, or has the wrong scale. Make sure to use this even if the user only says "save it so I can look at it in napari".
---

# Saving napari-compatible tifs

napari decides the layer type from the **dtype**, the scale from **OME metadata only**, and the axes from the **array order** (it has no axes metadata for plain tifs). Most "it looks wrong in napari" problems are one of those three, so decide them explicitly when saving. All of this was verified with napari 0.7 in the `scapeio` conda env.

## Labels (segmentations, masks)

- Save as a **plain tif** (`tifffile.imwrite(path, labels)`) with dtype **uint32** (int32/uint64 also work). napari opens only 32/64-bit integer tifs as a Labels layer; uint8/uint16 open as an Image, and so does anything saved as OME-TIFF through the OME reader plugin (it always returns an image).
- Background is 0, ids are 1..N. Binary masks too: save as uint32 if it should open as Labels directly (uint8 is fine if opening as an Image is OK).
- ImageJ-format tifs cannot hold uint32, so don't use `imagej=True` for labels.
- `compression='zlib'` is fine for labels (they compress very well, and napari still opens them as Labels).
- A plain tif **cannot carry the voxel size to napari** (resolution/spacing tags are ignored). Store it in the tif tags and in a sidecar json for reference, and set the scale in napari: `viewer.add_labels(labels, scale=scale)` or, for a dropped file, `labels_layer.scale = image_layer.scale` in the console.
- Put names/ids in a sidecar `{name}_ids.json` (`id_to_name`, `axes`, `voxel_size_um_...`), since tifs have nowhere to keep them.

## Images (projections, volumes)

- Save as **OME-TIFF** (`.ome.tif`) with `float32` or `uint16`, so the physical voxel size travels with the file:

```python
tifffile.imwrite(path, volume, ome=True,
                 metadata={'axes': 'ZYX',
                           'PhysicalSizeX': vx, 'PhysicalSizeXUnit': 'um',
                           'PhysicalSizeY': vy, 'PhysicalSizeYUnit': 'um',
                           'PhysicalSizeZ': vz, 'PhysicalSizeZUnit': 'um'})
```

- The scale is applied by the `napari-ome-tiff-reader` plugin (this is what opens dropped `.ome.tif` files here); napari's built-in tif reader ignores it. If a file shows at scale 1, that is the reader, not the file.
- A plain tif works as an image too (float or uint16 give an Image), just unscaled.

## Axis order: the one rule

**Labels must have exactly the same array shape and axis order as the image they overlay**, because napari displays arrays as they are and the scale tuple follows the same order. A label volume that looks rotated, mirrored, or at the wrong place against the image has almost always the wrong axis order (or a missing scale), not a wrong segmentation. Check shapes before anything else.

Conventions in this repo (SCAPE):

| Data | Array order | Scale (um) |
|---|---|---|
| `projection_images.h5` (`min_proj`, `max_proj`, `std_image`), `scape_utils.load_proj` | (Z, X, Y), e.g. (316, 960, 850) | `(vz, vx, vy)` = `(voxel_sizes_xyz[2], [0], [1])` |
| `min_proj.ome.tif` etc. (written with `data.transpose(0, 2, 1)`) | (Z, Y, X), e.g. (316, 850, 960) | `(vz, vy, vx)`; X is the 960-long axis |
| WebKnossos volumes (`wkw`) | (x, y, z) | - |
| Old manual label tifs (`RLF-RLM_leg_segments.tif`), `edit_labels_napari.py` | (Z, X, Y) like the h5 | `(vz, vx, vy)` |

- `voxel_sizes_xyz` in the h5 is stored `(x, y, z)`; `scape_utils.load_voxel_size(folder)` returns a dict with keys `x`, `y`, `z`.
- WebKnossos (x, y, z) -> OME order: `np.transpose(labels, (2, 1, 0))`; -> h5 order: `np.transpose(labels, (2, 0, 1))`. (See `ct_scan/webknossos_labels_to_tif.py`.)
- Swapping between the two image conventions is `.transpose(0, 2, 1)`.
- Say in the output docstring/json which order was used (`'axes': 'zyx'`), and when loading a label tif to overlay on `load_proj`, transpose it if its order differs (pattern in `scape/edit_labels_napari.py`).

## Check before declaring it done

1. Shapes: `labels.shape == image.shape` for the image it overlays (`load_proj` order or OME order, see above).
2. Alignment sanity check without opening a viewer: the mean image intensity under the labels should be clearly higher than the image mean (for fluorescent structures), and almost none of the labels should fall on the zero padding. A wrong axis order or flip fails this by a wide margin.
3. Layer type and scale as napari sees them: `python .claude/skills/napari-save/scripts/check_napari_tif.py file1.tif file2.ome.tif` (headless, prints layer type and scale for the default reader and the OME plugin). Run it with the `scapeio` env's python.

Other things worth knowing: keep arrays C-contiguous and the dtype explicit (a stray float64 or int64 doubles the file size), name files `{what}.tif` / `{what}.ome.tif` next to the image they belong to, and mention in the final message that labels need the scale set (or copy it from the image) if you saved a plain tif.
