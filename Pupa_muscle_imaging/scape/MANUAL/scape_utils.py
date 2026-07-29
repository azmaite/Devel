""" 
GENERAL PURPOSE FUNCTIONS FOR SCAPE DATA ANALYSIS
Writen manually with limited help from CLAUDE (max single functions)

It assumes the following folder structure (if not, it will fix it):
- main_path : /mnt/upamdya_data/MA/SCAPE
-- exp_folder : date_info folder than contains recordings from one day
--- sample#_run# : folder containing a single SCAPE recording (scape_path)
---- sample#_run# : subfolder folder containing raw data (.dat) files (scape_raw_path)
---- sample#_run#_info.mat : metadata file for the SCAPE recording
---- analysis : folder containing analysis results (analysis_path)
--- analysis_sample# : folder containing analysis results for several recordings for the same sample (analysis_combined_path)

load_napari_rois: load manually segmented ROIs from napari saved in NRRD format
_check_folder_structure: check if the folder structure is correct for SCAPE data path (whether the .dat files are inside a subfolder with the same name and whether there is a matching _info.mat file). If not, it will fix it by creating the subfolder and moving the files. Raises errors if the structure is not correct and cannot be fixed.
load_voxel_size: get voxel size of a deskewed scape 3D video (XYZ)
load_aspect_ratio_YX: get aspect ratio of a scape 3D video for Y/X (most common view)
load_max_proj: load the maximum projection of a SCAPE 3D video. If not already calculated, will do it and save it in the analysis folder.
"""

import nrrd
import numpy as np
import scapeio
from pathlib import Path
import tqdm
import h5py

ANALYSIS_PATH_NAME = "analysis"
PROJECTION_FILE_NAME = "projection_images.h5"


def load_napari_rois(roi_path: str | Path) -> np.ndarray:
    """
    Load manually segmented ROIs from napari saved in NRRD format.

    Parameters
    ----------
    roi_path : str | Path
        Path to the NRRD file containing the ROI labels.

    Returns
    -------
    np.ndarray
        ROI labels array with shape (z, y, x).
    """
    rois, _ = nrrd.read(roi_path)
    return np.transpose(rois, (2, 1, 0))


def _check_folder_structure(scape_path: str | Path):
    """
    Check if the folder structure is correct for SCAPE data path
    (whether the .dat files are inside a subfolder with the same name
    and whether there is a matching _info.mat file).
    If not, it will fix it by creating the subfolder and moving the files.
    Raises errors if the structure is not correct and cannot be fixed.
    
    Parameters
    ----------
    path : str | Path
        Path to the scape folder to check.   
    """
    scape_path = Path(scape_path)

    # check that the path name is sample#_run#
    base_name = scape_path.name
    if not base_name.startswith('sample') or '_run' not in base_name:
        raise ValueError(f"Path name {base_name} does not match expected format 'sample#_run#'.")
    
    # check that the scape_raw_path was not mistakenly passed to this function instead of the scape_path
    # (in that case, the parent folder has the same name)
    # to avoid recursively moving files inside further subfolders
    parent_path = scape_path.parent
    if parent_path.name == base_name:
        raise ValueError(f"Parent folder has the same name as the scape_path, probably scape_raw_path was passed. Please check the folder structure.")

    # check that there is a _info.mat file inside
    info_file = scape_path / f"{base_name}_info.mat"
    if not info_file.exists():
        print(f"Info file not found in scape_path. Checking parent folder...")
        
        info_file = parent_path / f"{base_name}_info.mat"
        if not info_file.exists():
            raise FileNotFoundError(f"Info file not found in parent folder either. Please check the folder structure.")
        
        else:
            # move the .dat files to a subfolder with the same name as the scape_path
            print(f"Info file found in parent folder. Moving .dat files to subfolder...")
            scape_raw_path = scape_path / base_name
            scape_raw_path.mkdir(exist_ok=True)
            # move _info.mat file and _stim_data.bin file
            info_file.rename(scape_raw_path / info_file.name)
            stim_file = parent_path / f"{base_name}_stim_data.bin"
            if stim_file.exists():
                stim_file.rename(scape_raw_path / stim_file.name)
            # move .dat files
            for n, dat_file in enumerate(scape_path.glob(f"*.dat")):
                dat_file.rename(scape_raw_path / dat_file.name)
            # move ModifiedData folder
            modified_data_folder = scape_path / "ModifiedData"
            if modified_data_folder.exists():
                modified_data_folder.rename(scape_raw_path / "ModifiedData")
            print(f"Files moved successfully ({n+1} .dat files moved).")

def load_voxel_size(scape_path: str | Path) -> dict:
    """
    Get the voxel size of a deskewed SCAPE 3D image (XYZ).

    Parameters
    ----------
    scape_path : str | Path
        Path to the folder containing the SCAPE data.
        (Will check that the folder structure is correct and fix it if not.)

    Returns
    -------
    dict
        Voxel size in micrometers for each axis (x, y, z).
    """

    scape_path = Path(scape_path)
    _check_folder_structure(scape_path)
    scape_raw_path = scape_path / scape_path.name

    tube_lens = '50 mm Navitar' #always!
    voxel_size = scapeio.get_voxel_size(scape_raw_path, tube_lens, deskewed=True)
    axes = {'z':0, 'x':1, 'y':2}
    voxel_size = {ax:voxel_size[axes[ax]] for ax in axes.keys()}
    return voxel_size

def load_aspect_ratio_YX(scape_path: str | Path) -> float:
    """
    Get the aspect ratio of a SCAPE 3D image for Y/X (most common view).

    Parameters
    ----------
    scape_path : str | Path
        Path to the folder containing the SCAPE data.
        (get_voxel_size will check that the folder structure is correct and fix it if not.)

    Returns
    -------
    float
        Aspect ratio of the image (Y/X).
    """
    voxel_size = load_voxel_size(scape_path)
    aspect_ratio = voxel_size['x'] / voxel_size['y']  # width / height
    return aspect_ratio

def load_proj(scape_path: str | Path, proj_type: str) -> np.ndarray:
    """
    Load the specified projection of a SCAPE 3D video.
    If not already calculated, will calculate all projection images.

    Parameters
    ----------  
    scape_path : str | Path
        Path to the folder containing the SCAPE data.
    proj_type : str
        Type of projection to load. Can be 'max_proj', 'min_proj', or 'std_image'.

    Returns
    -------
    np.ndarray
        Specified projection of the image along the z-axis.
        Also saved in the analysis folder as '{proj_type}_proj.'.
    """
    assert proj_type in ['max_proj', 'min_proj', 'std_image'], f"proj_type must be one of 'max_proj', 'min_proj', or 'std_image'. Got {proj_type} instead."

    scape_path = Path(scape_path)
    _check_folder_structure(scape_path)
    h5_path = scape_path / ANALYSIS_PATH_NAME / PROJECTION_FILE_NAME

    if not h5_path.exists():
        print(f"Projection images not found. Computing them...")
        _get_projection_images(scape_path)

    with h5py.File(h5_path, 'r') as f:
        proj = f[proj_type][:]
    return proj

def _load_num_volumes(scape_path: str | Path) -> int:
    """
    Get the number of volumes in a SCAPE recording.

    Parameters
    ----------
    scape_raw_path : str | Path
        Path to the raw SCAPE data folder.

    Returns
    -------
    int
    """
    scape_path = Path(scape_path)
    _check_folder_structure(scape_path)
    scape_raw_path = scape_path / scape_path.name
    scan_name = scapeio.get_scan_name(scape_raw_path)
    scan_metadata = scapeio.load_matlab_metadata(scan_name)
    return scan_metadata['daq']['numberOfScans']

def _get_projection_images(scape_path: str | Path, batch_size: int = 50):
    """
    Get the std, maximum and minimum projection images of a SCAPE 3D video
    and save them in the analysis folder as projection_images.h5.
    (for the std image, it will also save the sum_x, sum_x2 and frame_count to be able to combine the std images later)

    Parameters
    ----------
    scape_path : str | Path
        Path to the folder containing the SCAPE data.
    batch_size : int
        Number of volumes to load per batch.
    """
    # set save path
    scape_path = Path(scape_path)
    _check_folder_structure(scape_path)
    analysis_path = scape_path / ANALYSIS_PATH_NAME
    h5_path = analysis_path / PROJECTION_FILE_NAME

    sum_x: np.ndarray | None = None
    sum_x2: np.ndarray | None = None
    min_proj: np.ndarray | None = None
    max_proj: np.ndarray | None = None
    frame_count = 0

    scape_raw_path = str(scape_path / scape_path.name) # string needed for scapeio.load_scan
    num_volumes = _load_num_volumes(scape_path)

    for i in tqdm.tqdm(range(0, num_volumes, batch_size),
                       desc=f'Computing projection images ({scape_path.name})'):
        start_volume = i
        end_volume = min(i + batch_size, num_volumes)
        print(f'Loading volumes {start_volume} to {end_volume}...')
        video_batch, _ = scapeio.load_scan(scape_raw_path, colors=1, start_volume=start_volume,
                                           end_volume=end_volume - 1, return_axis_order=True)

        deskewed_batch = scapeio.operations.deskew_scan(scape_raw_path, images=video_batch)
        deskewed_batch = deskewed_batch.astype(np.float32)
        del video_batch

        if max_proj is None:
            max_proj = np.max(deskewed_batch, axis=0)
        else:
            max_proj = np.maximum(max_proj, np.max(deskewed_batch, axis=0))

        if min_proj is None:
            min_proj = np.min(deskewed_batch, axis=0)
        else:
            min_proj = np.minimum(min_proj, np.min(deskewed_batch, axis=0))

        if sum_x is None:
            sum_x = np.sum(deskewed_batch, axis=0)
            sum_x2 = np.sum(deskewed_batch ** 2, axis=0)
        else:
            sum_x += np.sum(deskewed_batch, axis=0)
            sum_x2 += np.sum(deskewed_batch ** 2, axis=0)
        frame_count += deskewed_batch.shape[0]

        del deskewed_batch

    mean = sum_x / frame_count
    mean_sq = sum_x2 / frame_count
    variance = np.maximum(0, mean_sq - np.square(mean))
    std_image = np.sqrt(variance)

    # SAVE the results
    analysis_path = scape_path / "analysis"
    analysis_path.mkdir(exist_ok=True)
    
    with h5py.File(h5_path, 'a') as f:
        for name, data in (('sum_x', sum_x), 
                           ('sum_x2', sum_x2), 
                           ('std_image', std_image), 
                           ('min_proj', min_proj), 
                           ('max_proj', max_proj),
                           ('frame_count', frame_count)):
            if name in f:
                del f[name]
            f.create_dataset(name, data=data)
    print(f'Saved min_proj, max_proj, sum_x, sum_x2, frame_count, std_image to {h5_path}')
    


