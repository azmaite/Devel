"""
Print how napari opens each given tif: layer type, dtype, shape and scale (headless, no GUI).

Opens every file with napari's default reader and, for .ome.tif files, also with the
napari-ome-tiff-reader plugin (the plugin applies the OME voxel size as the layer scale; the
default reader does not). Labels are saved as plain uint32 tifs and open as Labels layers
(scale 1: set it by hand); images are saved as OME-TIFFs and open as Images with the scale.

Usage (with the env that has napari, e.g. scapeio):
    python check_napari_tif.py file1.tif file2.ome.tif
"""

import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import napari  # noqa: E402


def describe(viewer: napari.Viewer, path: str, plugin: str | None = None) -> None:
    """
    Open `path` in `viewer` (optionally with a given reader plugin) and print the layer.

    Parameters
    ----------
    viewer : napari.Viewer
        Headless viewer.
    path : str
        Path to the tif.
    plugin : str or None
        Reader plugin to use, or None for napari's default reader.

    Returns
    -------
    None
    """
    reader = plugin or 'default reader'
    try:
        layer = viewer.open(path, plugin=plugin)[0] if plugin else viewer.open(path)[0]
        scale = [round(float(s), 3) for s in layer.scale]
        print(f'  {reader:26s} -> {type(layer).__name__:6s} {layer.data.dtype} '
              f'shape={tuple(layer.data.shape)} scale={scale}')
    except Exception as error:
        print(f'  {reader:26s} -> failed: {error!r}'[:200])
    viewer.layers.clear()


if __name__ == '__main__':
    viewer = napari.Viewer(show=False)
    for path in sys.argv[1:]:
        print(path)
        describe(viewer, path)
        if path.endswith('.ome.tif'):
            describe(viewer, path, plugin='napari-ome-tiff-reader')
