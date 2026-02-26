import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from matplotlib.widgets import LassoSelector
import av

def manual_roi_selection(image):
    """ manually select one or more ROIs.
    The image can be:
    - an image (rgb or grayscale)
    - a path to an image file (.jpg or .png)
    - a path to a .h5 file containing a dataset called 'frames'
    - a path to a .mkv file where each frame contains 4 reals frames as channels
    After roi selection, the rois will be saved to a .pkl file
    """

    # if image is an actual image, ask for folder to save rois (filedialog)
    if isinstance(image, np.ndarray):
        from tkinter import Tk
        from tkinter import filedialog

        root = Tk()
        root.withdraw()
        main_path = '/mnt/labserver/data/MA/Development_project'
        recording_path = filedialog.askdirectory(initialdir=main_path, title="Select folder to save ROIs")

    # otherwise, check that the file exists and is a file (.jpg, .png, .h5, or .mkv)
    else:
        assert os.path.exists(image), f"File {image} does not exist"
        assert os.path.isfile(image), f"Path {image} must be a file"
        assert image.endswith('.jpg') or image.endswith('.png') or image.endswith('.h5') or image.endswith('.mkv'), f"File {image} must be a .jpg, .png, .h5, or .mkv file"

        # will save ROIs in the same folder as the image
        recording_path = os.path.dirname(image)

        # if image .jpg, .png, load it
        if image.endswith('.jpg') or image.endswith('.png'):
            image = plt.imread(image)

        # if image is .h5, load the average of the first 100 frames
        elif image.endswith('.h5'):
             with h5py.File(image, 'r') as f:
                image = np.mean(f['frames'][:][0:100], axis=0)

        # if image is .mkv, load the average of the first 4 frames
        elif image.endswith('.mkv'):
            container = av.open(image)
            image = next(container.decode(video=0))
            image = image.to_ndarray(format='rgba64le')
            image = image[..., 0]
            container.close()



    # manual ROI selection
    rois = []

    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Draw ROI (freehand). Close window when done.")
    ax.axis("off")

    def on_select(verts):
        verts = np.array(verts)
        rois.append(verts)
        ax.plot(verts[:, 0], verts[:, 1], '-r')
        fig.canvas.draw_idle()

    lasso = LassoSelector(ax, on_select)

    plt.show()
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()

    # rois is a list of (N, 2) arrays of polygon vertices


    # save rois to a file
    roi_file = os.path.join(recording_path, 'manual_rois.pkl')
    with open(roi_file, 'wb') as f:
        pickle.dump(rois, f)

    print(f"Saved {len(rois)} ROIs to {roi_file}")