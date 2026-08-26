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
--- analysis_sample# : folder containing analysis results for several recordings for the 
    same sample (analysis_combined_path)

load_napari_rois: load manually segmented ROIs from napari saved in NRRD format
check_folder_structure: check if the folder structure is correct for SCAPE data path 
    (whether the .dat files are inside a subfolder with the same name and whether there is a matching _info.mat file). If not, it will fix it by creating the subfolder and moving the files. Raises errors if the structure is not correct and cannot be fixed.
load_voxel_size: get voxel size of a deskewed scape 3D video (XYZ)
load_aspect_ratio_YX: get aspect ratio of a scape 3D video for Y/X (most common view)
load_proj: load the specified projection ('max_proj', 'min_proj', or 'std_image') of a
    SCAPE 3D video. If not already calculated, will compute and save all projection images.
get_projection_images: compute the std, maximum, and minimum projection images, and an
    activity trace (mean fluorescence per volume, ignoring zero voxels), of a SCAPE 3D
    video. Saves the projections and activity trace in the analysis folder, both in an h5
    file and as OME-TIFFs (projections) / a PNG plot (activity) calibrated with the
    recording's voxel size / volume rate for napari and easy viewing
combine_sample_projections: combine projection images and activity traces from several
    runs of the same sample and save them (as h5, OME-TIFFs, and a combined activity PNG
    with dotted lines marking run boundaries) to a new sample#_combined folder
_plot_activity: plot an activity trace (mean fluorescence per volume) over time and save
    it as a PNG, optionally marking run boundaries with vertical dotted lines
"""

import nrrd
import numpy as np
import scapeio
import tifffile
from pathlib import Path
import tqdm
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MAIN_PATH = '/mnt/upramdya_data/MA/SCAPE'
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


def check_folder_structure(scape_path: str | Path):
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
    check_folder_structure(scape_path)
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
    check_folder_structure(scape_path)
    h5_path = scape_path / ANALYSIS_PATH_NAME / PROJECTION_FILE_NAME

    if not h5_path.exists():
        print(f"Projection images not found. Computing them...")
        get_projection_images(scape_path)

    with h5py.File(h5_path, 'r') as f:
        proj = f[proj_type][:]
    return proj

def get_projection_images(scape_path: str | Path, batch_size: int = 50):
    """
    Get the std, maximum and minimum projection images, and the activity trace
    (mean fluorescence per volume, ignoring zero voxels as if they were nan),
    of a SCAPE 3D video, and save them in the analysis folder as
    projection_images.h5.
    (for the std image, it will also save the sum_x, sum_x2 and frame_count to be able to combine the std images later)
    Also saves `std_image`, `min_proj`, and `max_proj` as OME-TIFFs
    (`{proj_type}.ome.tif`), calibrated with the recording's voxel size
    (see `load_voxel_size`) so they open correctly scaled in napari, and the
    activity trace as `activity.png` (see `_plot_activity`).

    Parameters
    ----------
    scape_path : str | Path
        Path to the folder containing the SCAPE data.
    batch_size : int
        Number of volumes to load per batch.
    """
    # set save path
    scape_path = Path(scape_path)
    check_folder_structure(scape_path)
    analysis_path = scape_path / ANALYSIS_PATH_NAME
    h5_path = analysis_path / PROJECTION_FILE_NAME

    sum_x: np.ndarray | None = None
    sum_x2: np.ndarray | None = None
    min_proj: np.ndarray | None = None
    max_proj: np.ndarray | None = None
    frame_count = 0
    activity_batches = []

    scape_raw_path = str(scape_path / scape_path.name) # string needed for scapeio.load_scan
    _, num_volumes, scan_rate = scapeio.get_scan_timing(scape_raw_path)

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

        # per-volume mean fluorescence, ignoring zero voxels as if they were nan
        # (zeros don't contribute to the sum, so only the nonzero count matters)
        spatial_axes = tuple(range(1, deskewed_batch.ndim))
        nonzero_sum = np.sum(deskewed_batch, axis=spatial_axes)
        nonzero_count = np.count_nonzero(deskewed_batch, axis=spatial_axes)
        with np.errstate(invalid='ignore'):
            activity_batches.append(nonzero_sum / nonzero_count)

        del deskewed_batch

    mean = sum_x / frame_count
    mean_sq = sum_x2 / frame_count
    variance = np.maximum(0, mean_sq - np.square(mean))
    std_image = np.sqrt(variance)
    activity = np.concatenate(activity_batches)

    # SAVE the results
    analysis_path = scape_path / "analysis"
    analysis_path.mkdir(exist_ok=True)

    with h5py.File(h5_path, 'a') as f:
        for name, data in (('sum_x', sum_x),
                           ('sum_x2', sum_x2),
                           ('std_image', std_image),
                           ('min_proj', min_proj),
                           ('max_proj', max_proj),
                           ('frame_count', frame_count),
                           ('activity', activity)):
            if name in f:
                del f[name]
            f.create_dataset(name, data=data)
        f.attrs['scan_rate'] = scan_rate
    print(f'Saved min_proj, max_proj, sum_x, sum_x2, frame_count, std_image, activity to {h5_path}')

    # also save std_image, min_proj and max_proj as OME-TIFFs, calibrated with the
    # recording's voxel size, so they open correctly scaled in napari
    voxel_size = load_voxel_size(scape_path)
    tiff_metadata = {'axes': 'ZYX',
                     'PhysicalSizeX': voxel_size['x'], 'PhysicalSizeXUnit': 'um',
                     'PhysicalSizeY': voxel_size['y'], 'PhysicalSizeYUnit': 'um',
                     'PhysicalSizeZ': voxel_size['z'], 'PhysicalSizeZUnit': 'um'}
    for proj_type, data in (('std_image', std_image), ('min_proj', min_proj), ('max_proj', max_proj)):
        tiff_path = analysis_path / f'{proj_type}.ome.tif'
        tifffile.imwrite(tiff_path, data.transpose(0, 2, 1), ome=True, metadata=tiff_metadata)
    print(f'Saved std_image, min_proj, max_proj as OME-TIFFs to {analysis_path}')

    # also plot and save the activity trace over time
    time_axis = np.arange(len(activity)) / scan_rate
    _plot_activity(activity, time_axis, analysis_path / 'activity.png',
                  title=f'Average fluorescence over time ({scape_path.name})')

def _plot_activity(activity: np.ndarray, time_axis: np.ndarray, save_path: str | Path,
                   title: str, run_boundaries: list[float] | None = None):
    """
    Plot an activity trace (mean fluorescence per volume) over time and save
    it as a PNG.

    Parameters
    ----------
    activity : np.ndarray
        Activity trace, one value per volume.
    time_axis : np.ndarray
        Time (in seconds) corresponding to each value in `activity`.
    save_path : str | Path
        Path to save the PNG figure to.
    title : str
        Title of the plot.
    run_boundaries : list of float or None
        Times (in seconds) at which to draw vertical dotted lines separating
        runs (used when combining multiple runs). None for a single run.
    """
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(time_axis, activity, lw=0.8, color='steelblue')
    if run_boundaries:
        for t in run_boundaries:
            ax.axvline(t, color='gray', linestyle=':', lw=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Average fluorescence')
    ax.set_title(title)
    ax.set_xlim(time_axis[0], time_axis[-1])
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved activity plot to {save_path}')

def combine_sample_projections(exp_folder: str | Path, sample_name: str, run_paths: list[Path]):
    """
    Combine projection images and activity traces from several runs of the
    same sample and save them to a new sample#_combined folder: the
    projections as an h5 file and calibrated OME-TIFFs, and the combined
    activity trace as an h5 dataset and a PNG plot (with vertical dotted
    lines marking the boundaries between runs).

    Parameters
    ----------
    exp_folder : str | Path
        Path to the experiment folder.
    sample_name : str
        Name of the sample (e.g. `'sample1'`).
    run_paths : list[Path]
        Paths to the `sample#_run#` folders to combine.
    """
    exp_folder = Path(exp_folder)
    combined_path = exp_folder / f'{sample_name}_combined'
    combined_path.mkdir(exist_ok=True)

    sum_x: np.ndarray | None = None
    sum_x2: np.ndarray | None = None
    min_proj: np.ndarray | None = None
    max_proj: np.ndarray | None = None
    frame_count = 0
    activity_runs = []
    time_axis_runs = []
    run_boundaries = []
    cumulative_time = 0.0

    for scape_path in run_paths:
        h5_path = scape_path / ANALYSIS_PATH_NAME / PROJECTION_FILE_NAME
        with h5py.File(h5_path, 'r') as f:
            run_sum_x = f['sum_x'][:]
            run_sum_x2 = f['sum_x2'][:]
            run_min_proj = f['min_proj'][:]
            run_max_proj = f['max_proj'][:]
            run_frame_count = int(f['frame_count'][()])
            run_activity = f['activity'][:]
            run_scan_rate = f.attrs['scan_rate']

        if sum_x is None:
            sum_x, sum_x2 = run_sum_x, run_sum_x2
            min_proj, max_proj = run_min_proj, run_max_proj
        else:
            sum_x += run_sum_x
            sum_x2 += run_sum_x2
            min_proj = np.minimum(min_proj, run_min_proj)
            max_proj = np.maximum(max_proj, run_max_proj)
        frame_count += run_frame_count

        activity_runs.append(run_activity)
        time_axis_runs.append(cumulative_time + np.arange(len(run_activity)) / run_scan_rate)
        cumulative_time += len(run_activity) / run_scan_rate
        run_boundaries.append(cumulative_time)

    mean = sum_x / frame_count
    mean_sq = sum_x2 / frame_count
    variance = np.maximum(0, mean_sq - np.square(mean))
    std_image = np.sqrt(variance)
    activity = np.concatenate(activity_runs)
    time_axis = np.concatenate(time_axis_runs)
    run_boundaries = run_boundaries[:-1] # no boundary line needed after the last run

    h5_path = combined_path / PROJECTION_FILE_NAME
    with h5py.File(h5_path, 'a') as f:
        for name, data in (('sum_x', sum_x),
                           ('sum_x2', sum_x2),
                           ('std_image', std_image),
                           ('min_proj', min_proj),
                           ('max_proj', max_proj),
                           ('frame_count', frame_count),
                           ('activity', activity)):
            if name in f:
                del f[name]
            f.create_dataset(name, data=data)
    print(f'Saved combined min_proj, max_proj, sum_x, sum_x2, frame_count, std_image, activity to {h5_path}')

    voxel_size = load_voxel_size(run_paths[0])
    tiff_metadata = {'axes': 'ZYX',
                     'PhysicalSizeX': voxel_size['x'], 'PhysicalSizeXUnit': 'um',
                     'PhysicalSizeY': voxel_size['y'], 'PhysicalSizeYUnit': 'um',
                     'PhysicalSizeZ': voxel_size['z'], 'PhysicalSizeZUnit': 'um'}
    for proj_type, data in (('std_image', std_image), ('min_proj', min_proj), ('max_proj', max_proj)):
        tiff_path = combined_path / f'{proj_type}.ome.tif'
        tifffile.imwrite(tiff_path, data.transpose(0, 2, 1), ome=True, metadata=tiff_metadata)
    print(f'Saved combined std_image, min_proj, max_proj as OME-TIFFs to {combined_path}')

    _plot_activity(activity, time_axis, combined_path / 'activity.png',
                  title=f'Average fluorescence over time ({sample_name} combined)',
                  run_boundaries=run_boundaries)

