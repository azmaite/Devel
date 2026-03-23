""" Script to load a Z stack saved as a tif file into napari with
the correct Z-step and pixel size
(default it is 1:1:1 xyz, which ends up flattened often).
Run in napari-env environment, which has napari and tifffile installed as per
the original instructions."""

import tifffile
import napari

import tkinter as tk
from tkinter import filedialog


def load_tif_napari():

    # select the tif file you want to load
    main_path = '/mnt/labserver/data/MA/Confocal'
    root = tk.Tk()
    root.withdraw()
    tif_path = filedialog.askopenfilename(initialdir=main_path, title='Select .tif file', filetypes=[('TIF files', '*.tif')])
    print(f"Selected file: {tif_path}")

    # Load the raw data
    data = tifffile.imread(tif_path)

    # Get the metadata from the tif file to determine the scale
    with tifffile.TiffFile(tif_path) as tif:
        page = tif.pages[0]

        # get xyz pixel size from metadata
        metadata = tif.imagej_metadata  # works for Fiji/ImageJ TIFFs
        z_step = metadata.get("spacing", None)

        x_res = page.tags.get("XResolution")
        y_res = page.tags.get("YResolution")
        x_step = x_res.value[1] / x_res.value[0]
        y_step = y_res.value[1] / y_res.value[0]

    # Manually define the scale (Z, Y, X)
    scale = (z_step, y_step, x_step)

    # open napari
    viewer = napari.Viewer()

    # load image into napari with the correct scale
    viewer.add_image(data, scale=scale, name="image")

    # run napari
    napari.run()




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Load a tif into napari.')

    load_tif_napari()