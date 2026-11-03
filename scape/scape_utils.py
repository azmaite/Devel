""" 
GENERAL PURPOSE FUNCTIONS FOR SCAPE DATA ANALYSIS
Writen manually with limited help from CLAUDE (max single functions)

It assumes the following folder structure (if not, it will fix it):
- main_path : /mnt/labserver/data/MA/SCAPE (`common.paths.SCAPE_PATH`)
-- exp_folder : date_info folder than contains recordings from one day
--- sample#_run# : folder containing a single SCAPE recording (scape_path)
---- sample#_run# : subfolder folder containing raw data (.dat) files (scape_raw_path)
---- sample#_run#_info.mat : metadata file for the SCAPE recording
---- sample#_run#_stim_data.bin : DAQ voltage traces for the SCAPE recording
---- analysis : folder containing analysis results (analysis_path)
--- analysis_sample# : folder containing analysis results for several recordings for the 
    same sample (analysis_combined_path)

load_napari_rois: load manually segmented ROIs from napari saved in NRRD format
is_combined_path: check whether a path is a sample#_combined folder rather than a
    single-run sample#_run# folder
analysis_path_for: get the folder containing a SCAPE path's projection images/activity
    plot - itself for a sample#_combined folder, or its analysis subfolder for a
    sample#_run# folder
check_folder_structure: check if the folder structure is correct for SCAPE data path
    (whether the raw data (.dat files, .nrrd copy and acquisition metadata) are inside a
    subfolder with the same name and whether the matching _info.mat/_stim_data.bin files are
    next to it). If not, it will fix it by moving the files. Raises errors if the structure is
    not correct and cannot be fixed. A no-op for a sample#_combined folder, which has no raw
    data of its own.
load_voxel_size: get voxel size of a deskewed scape 3D video (XYZ). For a sample#_combined
    folder, read back from its max_proj.ome.tif metadata instead of raw data, which it
    doesn't have.
load_aspect_ratio_YX: get aspect ratio of a scape 3D video for Y/X (most common view)
load_proj: load the specified projection ('max_proj', 'min_proj', 'std_image' or
    'median_proj') of a SCAPE 3D video. If not already calculated, will compute and save all
    projection images (a missing 'median_proj' is computed on its own, see
    get_median_projection).
get_auto_batch_size: choose the number of volumes to load per batch from the recording's
    volume size (raw voxels per volume), to keep memory use roughly constant
get_projection_images: compute the std, maximum, and minimum projection images, and an
    activity trace (mean fluorescence per volume, ignoring zero voxels), of a SCAPE 3D
    video. Saves the projections, activity trace, and voxel size (`voxel_sizes_xyz`) in
    the analysis folder, both in an h5 file and as OME-TIFFs (projections) / a PNG plot
    (activity) calibrated with the recording's voxel size / volume rate for napari and
    easy viewing (the activity PNG can be saved elsewhere, e.g. the experiment's
    `00_activity` folder, see `preprocess_scape_exp`). Optionally (`median=True`) also the
    median projection, in the same pass
get_median_projection: compute only the median projection of a SCAPE 3D video and add it
    to its projection_images.h5 (and as an OME-TIFF), e.g. for runs whose other projections
    were computed before the median was added. The median needs every volume of the
    recording for each voxel, so the deskewed volumes are kept in RAM if they fit, else in a
    temporary memmap on the local disk (see _MedianStack)
combine_sample_projections: combine projection images and activity traces from several
    runs of the same sample and save them (as h5, OME-TIFFs, and a combined activity PNG
    with dotted lines marking run boundaries) to a new sample#_combined folder, recording
    the combined runs' names as a `source_runs` attribute on the h5 file (runs whose
    projection shape differs from the majority, e.g. a short test scan, are skipped). The
    median projection can't be combined from the runs' medians, so the combined folder has
    none
_plot_activity: plot an activity trace (mean fluorescence per volume) over time and save
    it as a PNG, optionally marking run boundaries with vertical dotted lines
"""

import os
import shutil
import tempfile

import nrrd
import numpy as np
import psutil
import scapeio
import tifffile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
import tqdm
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common.paths import SCAPE_PATH

MAIN_PATH = str(SCAPE_PATH)
ANALYSIS_PATH_NAME = "analysis"
PROJECTION_FILE_NAME = "projection_images.h5"
AUTO_BATCH_VOXELS = 1.4e9 # raw voxels loaded per batch by get_auto_batch_size
MEDIAN_RAM_FRACTION = 0.5 # median stack kept in RAM if under this fraction of available RAM
MEDIAN_SLAB_BYTES = 1e9   # bytes of the median stack processed at a time


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


def is_combined_path(scape_path: str | Path) -> bool:
    """
    Check whether `scape_path` is a `sample#_combined` folder (see
    `combine_sample_projections`), rather than a single-run
    `sample#_run#` folder.

    Parameters
    ----------
    scape_path : str | Path
        Path to check.

    Returns
    -------
    bool
    """
    return Path(scape_path).name.endswith('_combined')


def analysis_path_for(scape_path: str | Path) -> Path:
    """
    Get the folder containing a SCAPE path's projection images/activity
    plot: `scape_path` itself for a `sample#_combined` folder (see
    `combine_sample_projections`, which writes directly there), or
    `scape_path / ANALYSIS_PATH_NAME` for a single-run `sample#_run#`
    folder (see `get_projection_images`).

    Parameters
    ----------
    scape_path : str | Path
        Path to the SCAPE data folder.

    Returns
    -------
    Path
    """
    scape_path = Path(scape_path)
    return scape_path if is_combined_path(scape_path) else scape_path / ANALYSIS_PATH_NAME


def check_folder_structure(scape_path: str | Path):
    """
    Check if the folder structure is correct for a SCAPE data path.

    A `sample#_combined` folder (see `combine_sample_projections`) has
    no raw data of its own - it's just projections/activity combined
    from several runs - so nothing is checked for one beyond its name.

    For a `sample#_run#` folder (including `sample#_run#_HR` ones),
    checks whether the matching `_info.mat` and `_stim_data.bin` files
    are inside it, and whether the raw data (.dat files, a converted
    .nrrd, `acquisitionmetadata.ini`, `Spooled files.sifx` and the
    `ModifiedData` folder) are inside a subfolder with the same name.
    If not, it will fix it by moving `_info.mat`/`_stim_data.bin` in
    from the parent folder (as acquired) or up from the subfolder (where
    an older version of this function wrongly put them), and by creating
    the subfolder and moving the raw data into it. Raises errors if the
    structure is not correct and cannot be fixed.

    Parameters
    ----------
    path : str | Path
        Path to the scape folder to check.
    """
    scape_path = Path(scape_path)

    if is_combined_path(scape_path):
        return

    # check that the path name is sample#_run#
    base_name = scape_path.name
    if not base_name.startswith('sample') or '_run' not in base_name:
        raise ValueError(f"Path name {base_name} does not match expected format "
                         f"'sample#_run#' or 'sample#_combined'.")

    # check that the scape_raw_path was not mistakenly passed to this function instead of the scape_path
    # (in that case, the parent folder has the same name)
    # to avoid recursively moving files inside further subfolders
    parent_path = scape_path.parent
    if parent_path.name == base_name:
        raise ValueError(f"Parent folder has the same name as the scape_path, probably scape_raw_path was passed. Please check the folder structure.")

    # check that there is a _info.mat file inside scape_path (next to scape_raw_path, where
    # scapeio expects it). If not, look for it in the parent folder (as acquired) or inside
    # scape_raw_path (where an older version of this function wrongly moved it)
    scape_raw_path = scape_path / base_name
    for file_suffix in ('_info.mat', '_stim_data.bin'):
        file_name = f"{base_name}{file_suffix}"
        if (scape_path / file_name).exists():
            continue
        for other_path in (parent_path, scape_raw_path):
            if (other_path / file_name).exists():
                print(f"Moving {file_name} from {other_path} into {scape_path}...")
                (other_path / file_name).rename(scape_path / file_name)
                break
        else:
            if file_suffix == '_info.mat':
                raise FileNotFoundError(f"Info file {file_name} not found in {scape_path}, its parent "
                                        f"folder, or {scape_raw_path}. Please check the folder structure.")

    # move the raw data (and the other files created during acquisition, which scapeio
    # expects next to the .dat files) into a subfolder with the same name as the scape_path
    raw_files = [p for p in scape_path.iterdir()
                 if p.suffix in ('.dat', '.nrrd')
                 or p.name in ('ModifiedData', 'acquisitionmetadata.ini', 'Spooled files.sifx')]
    if raw_files:
        print(f"Moving raw data files into {scape_raw_path}...")
        scape_raw_path.mkdir(exist_ok=True)
        for raw_file in raw_files:
            raw_file.rename(scape_raw_path / raw_file.name)
        n_dat = sum(p.suffix == '.dat' for p in raw_files)
        print(f"Files moved successfully ({n_dat} .dat files and {len(raw_files) - n_dat} other files moved).")

def load_voxel_size(scape_path: str | Path) -> dict:
    """
    Get the voxel size of a deskewed SCAPE 3D image (XYZ).

    A `sample#_combined` folder (see `combine_sample_projections`) has
    no raw data of its own to query for this, so its voxel size is read
    back instead from `max_proj.ome.tif`'s embedded OME-XML metadata -
    the same values, written there (as `voxel_size['x']`/`['y']`/`['z']`
    verbatim) when the projections were computed/combined (see
    `get_projection_images`/`combine_sample_projections`).

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

    if is_combined_path(scape_path):
        tiff_path = analysis_path_for(scape_path) / 'max_proj.ome.tif'
        with tifffile.TiffFile(tiff_path) as tif:
            ome_root = ET.fromstring(tif.ome_metadata)
        ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
        pixels = ome_root.find('.//ome:Pixels', ns)
        return {'z': float(pixels.attrib['PhysicalSizeZ']),
                'x': float(pixels.attrib['PhysicalSizeX']),
                'y': float(pixels.attrib['PhysicalSizeY'])}

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
        Type of projection to load. Can be 'max_proj', 'min_proj', 'std_image' or
        'median_proj'.

    Returns
    -------
    np.ndarray
        Specified projection of the image along the z-axis.
        Also saved in the analysis folder as '{proj_type}_proj.'.
    """
    proj_types = ['max_proj', 'min_proj', 'std_image', 'median_proj']
    assert proj_type in proj_types, f"proj_type must be one of {proj_types}. Got {proj_type} instead."

    scape_path = Path(scape_path)
    check_folder_structure(scape_path)
    h5_path = analysis_path_for(scape_path) / PROJECTION_FILE_NAME

    if not h5_path.exists():
        print(f"Projection images not found. Computing them...")
        get_projection_images(scape_path, median=proj_type == 'median_proj')
    if proj_type == 'median_proj':
        with h5py.File(h5_path, 'r') as f:
            has_median = 'median_proj' in f
        if not has_median:
            print(f"Median projection not found. Computing it...")
            get_median_projection(scape_path)

    with h5py.File(h5_path, 'r') as f:
        proj = f[proj_type][:]
    return proj

def get_auto_batch_size(scape_path: str | Path) -> int:
    """
    Choose how many volumes to load per batch in `get_projection_images` so
    that the memory used per batch is about constant, whatever the size of
    the recording's volumes.

    Memory per batch scales with the number of raw voxels per volume
    (planes per volume x image height x image width, from the recording's
    metadata), so the batch size is `AUTO_BATCH_VOXELS` divided by that,
    rounded and clipped to between 1 and the number of volumes. The budget
    is calibrated on recordings that are known to work: batches of 50
    volumes of ~100 planes (~27 Mvoxels per volume) and batches of 7 volumes
    of ~750 planes (~200 Mvoxels per volume).

    Parameters
    ----------
    scape_path : str | Path
        Path to a single `sample#_run#` folder (with its folder structure
        already fixed by `check_folder_structure`).

    Returns
    -------
    int
        Number of volumes to load per batch.
    """
    scape_path = Path(scape_path)
    scape_raw_path = str(scape_path / scape_path.name) # string needed for scapeio
    scan_name = scapeio.get_scan_name(scape_raw_path)
    daq_metadata = scapeio.load_matlab_metadata(scan_name)['daq']
    camera_metadata = scapeio.load_camera_metadata(scan_name)['data']
    num_volumes = int(daq_metadata['numberOfScans'])
    voxels_per_volume = (int(daq_metadata['pixelsPerLine'])
                         * int(camera_metadata['aoiheight'])
                         * int(camera_metadata['aoiwidth']))
    batch_size = int(round(AUTO_BATCH_VOXELS / voxels_per_volume))
    batch_size = max(1, min(batch_size, num_volumes))
    print(f'Auto batch size for {scape_path.name}: {batch_size} volumes '
          f'({voxels_per_volume / 1e6:.0f} Mvoxels per volume)')
    return batch_size


def _tiff_metadata(voxel_size: dict) -> dict:
    """
    OME metadata of a projection OME-TIFF (axes ZYX, i.e. the h5 arrays
    transposed with `.transpose(0, 2, 1)`), calibrated with `voxel_size` (see
    `load_voxel_size`), so it opens correctly scaled in napari.
    """
    return {'axes': 'ZYX',
            'PhysicalSizeX': voxel_size['x'], 'PhysicalSizeXUnit': 'um',
            'PhysicalSizeY': voxel_size['y'], 'PhysicalSizeYUnit': 'um',
            'PhysicalSizeZ': voxel_size['z'], 'PhysicalSizeZUnit': 'um'}


def _deskewed_batches(scape_path: Path,
                      batch_size: int | None,
                      desc: str):
    """
    Load and deskew the volumes of a recording, a batch at a time.

    Parameters
    ----------
    scape_path : Path
        A single `sample#_run#` folder (folder structure already checked).
    batch_size : int | None
        Number of volumes per batch; None: see `get_auto_batch_size`.
    desc : str
        Description of the progress bar.

    Yields
    ------
    tuple[int, np.ndarray]
        Index of the first volume of the batch, and the deskewed batch
        (time, z, x, y), as returned by scapeio (uint16). A batch of a single
        volume, which scapeio returns without a time axis, gets it back.
    """
    if batch_size is None:
        batch_size = get_auto_batch_size(scape_path)
    scape_raw_path = str(scape_path / scape_path.name) # string needed for scapeio.load_scan
    _, num_volumes, _ = scapeio.get_scan_timing(scape_raw_path)
    for start_volume in tqdm.tqdm(range(0, num_volumes, batch_size), desc=desc):
        end_volume = min(start_volume + batch_size, num_volumes)
        print(f'Loading volumes {start_volume} to {end_volume}...')
        video_batch, axis_order = scapeio.load_scan(scape_raw_path, colors=1,
                                                    start_volume=start_volume,
                                                    end_volume=end_volume - 1,
                                                    return_axis_order=True)
        if 't' not in axis_order:
            # scapeio drops the time axis when a batch holds a single volume (e.g. a
            # last batch of 1 volume); restore it so axis 0 is always time
            video_batch = video_batch[np.newaxis]
        deskewed_batch = scapeio.operations.deskew_scan(scape_raw_path, images=video_batch)
        del video_batch
        yield start_volume, deskewed_batch


class _MedianStack:
    """
    All the deskewed volumes of a recording, filled batch by batch, to take
    their median over time voxel by voxel. Kept in RAM if the stack takes less
    than `MEDIAN_RAM_FRACTION` of the available RAM, else in a temporary
    memmap on the local disk (in `tempfile.gettempdir()`, deleted by `close`).
    The median is then taken one slab along z at a time (`MEDIAN_SLAB_BYTES`).
    """

    def __init__(self,
                 num_volumes: int,
                 volume_shape: tuple,
                 dtype: np.dtype,
                 name: str):
        """
        Parameters
        ----------
        num_volumes : int
            Number of volumes of the recording.
        volume_shape : tuple
            Shape of one deskewed volume (z, x, y).
        dtype : np.dtype
            Data type of the deskewed volumes.
        name : str
            Name of the recording (for messages and the memmap file name).
        """
        shape = (num_volumes, *volume_shape)
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        available_ram = psutil.virtual_memory().available
        self.path = None
        if nbytes < MEDIAN_RAM_FRACTION * available_ram:
            print(f'Median of {name}: keeping the {nbytes / 1e9:.1f} GB stack in RAM.')
            self.stack = np.empty(shape, dtype)
            return
        temp_dir = tempfile.gettempdir()
        free_disk = shutil.disk_usage(temp_dir).free
        if nbytes > 0.95 * free_disk:
            raise MemoryError(f'Median of {name}: the {nbytes / 1e9:.1f} GB stack fits neither '
                              f'in RAM ({available_ram / 1e9:.1f} GB available) nor on the '
                              f'local disk ({free_disk / 1e9:.1f} GB free in {temp_dir}).')
        handle, path = tempfile.mkstemp(prefix=f'median_{name}_', suffix='.dat')
        os.close(handle)
        self.path = Path(path)
        print(f'Median of {name}: the {nbytes / 1e9:.1f} GB stack does not fit in RAM, '
              f'using a temporary memmap ({self.path}).')
        self.stack = np.memmap(self.path, dtype=dtype, mode='w+', shape=shape)

    def add(self,
            start_volume: int,
            batch: np.ndarray) -> None:
        """
        Store a batch of deskewed volumes (time first) starting at volume
        `start_volume`.
        """
        self.stack[start_volume:start_volume + batch.shape[0]] = batch

    def median(self) -> np.ndarray:
        """
        Median over time of every voxel (float32; for an even number of
        volumes, the mean of the two middle values).
        """
        num_volumes, depth = self.stack.shape[:2]
        slab_bytes = num_volumes * int(np.prod(self.stack.shape[2:])) * self.stack.itemsize
        slab = max(1, int(MEDIAN_SLAB_BYTES // slab_bytes))
        median = np.empty(self.stack.shape[1:], np.float32)
        for z in tqdm.tqdm(range(0, depth, slab), desc='Median over time'):
            median[z:z + slab] = np.median(np.asarray(self.stack[:, z:z + slab]), axis=0)
        return median

    def close(self) -> None:
        """
        Free the stack and delete the memmap file, if any.
        """
        del self.stack
        if self.path is not None:
            self.path.unlink(missing_ok=True)


def _save_median(analysis_path: Path,
                 median_proj: np.ndarray,
                 voxel_size: dict) -> None:
    """
    Save the median projection in the analysis folder's projection_images.h5
    (`median_proj`) and as `median_proj.ome.tif`.
    """
    with h5py.File(analysis_path / PROJECTION_FILE_NAME, 'a') as f:
        if 'median_proj' in f:
            del f['median_proj']
        f.create_dataset('median_proj', data=median_proj)
    tifffile.imwrite(analysis_path / 'median_proj.ome.tif', median_proj.transpose(0, 2, 1),
                     ome=True, metadata=_tiff_metadata(voxel_size))
    print(f'Saved median_proj to {analysis_path / PROJECTION_FILE_NAME} and as an OME-TIFF')


def get_projection_images(scape_path: str | Path, batch_size: int | None = None,
                          activity_path: str | Path | None = None,
                          median: bool = False):
    """
    Get the std, maximum and minimum projection images, and the activity trace
    (mean fluorescence per volume, ignoring zero voxels as if they were nan),
    of a SCAPE 3D video, and save them in the analysis folder as
    projection_images.h5.
    (for the std image, it will also save the sum_x, sum_x2 and frame_count to be able to combine the std images later)
    Also saves the recording's voxel size as `voxel_sizes_xyz` (see `load_voxel_size`).
    Also saves `std_image`, `min_proj`, and `max_proj` as OME-TIFFs
    (`{proj_type}.ome.tif`), calibrated with the recording's voxel size
    (see `load_voxel_size`) so they open correctly scaled in napari, and the
    activity trace as `activity.png` (see `_plot_activity`), or at `activity_path`.
    With `median=True`, the median projection is computed in the same pass and
    saved too (`median_proj`, see `get_median_projection`).

    Parameters
    ----------
    scape_path : str | Path
        Path to the folder containing the SCAPE data. Must be a single
        `sample#_run#` folder, not a `sample#_combined` one - a
        combined folder has no raw data of its own to compute these
        from (see `combine_sample_projections` instead).
    batch_size : int | None
        Number of volumes to load per batch. If None (default), chosen
        automatically from the volume size (see `get_auto_batch_size`).
        A last batch holding a single
        volume is handled correctly (scapeio returns it without a time axis,
        which is re-added here).
    activity_path : str | Path | None
        Where to save the activity plot. If None (default), `activity.png` in
        the analysis folder (`preprocess_scape_exp` passes a path in the
        experiment's `00_activity` folder for long recordings).
    median : bool
        Also compute the median projection, which needs every volume at once
        (in RAM, or a temporary memmap on the local disk, see `_MedianStack`),
        so only for short recordings. If there is room for neither, the median
        is skipped with a warning. Default False.
    """
    # set save path
    scape_path = Path(scape_path)
    if is_combined_path(scape_path):
        raise ValueError(f"{scape_path} is a sample#_combined folder, which has no raw "
                         f"data of its own - use combine_sample_projections instead.")
    check_folder_structure(scape_path)
    analysis_path = scape_path / ANALYSIS_PATH_NAME
    h5_path = analysis_path / PROJECTION_FILE_NAME

    sum_x: np.ndarray | None = None
    sum_x2: np.ndarray | None = None
    min_proj: np.ndarray | None = None
    max_proj: np.ndarray | None = None
    median_stack: _MedianStack | None = None
    frame_count = 0
    activity_batches = []

    scape_raw_path = str(scape_path / scape_path.name) # string needed for scapeio
    _, num_volumes, scan_rate = scapeio.get_scan_timing(scape_raw_path)

    try:
        for start_volume, deskewed_batch in _deskewed_batches(
                scape_path, batch_size, f'Computing projection images ({scape_path.name})'):
            if median and median_stack is None:
                try:
                    median_stack = _MedianStack(num_volumes, deskewed_batch.shape[1:],
                                                deskewed_batch.dtype, scape_path.name)
                except MemoryError as error:
                    print(f'Warning: {error} Skipping the median projection.')
                    median = False
            if median:
                median_stack.add(start_volume, deskewed_batch)
            deskewed_batch = deskewed_batch.astype(np.float32)

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

        median_proj = median_stack.median() if median_stack is not None else None
    finally:
        if median_stack is not None:
            median_stack.close()

    mean = sum_x / frame_count
    mean_sq = sum_x2 / frame_count
    variance = np.maximum(0, mean_sq - np.square(mean))
    std_image = np.sqrt(variance)
    activity = np.concatenate(activity_batches)

    # SAVE the results
    analysis_path = scape_path / "analysis"
    analysis_path.mkdir(exist_ok=True)
    voxel_size = load_voxel_size(scape_path)

    with h5py.File(h5_path, 'a') as f:
        for name, data in (('sum_x', sum_x),
                           ('sum_x2', sum_x2),
                           ('std_image', std_image),
                           ('min_proj', min_proj),
                           ('max_proj', max_proj),
                           ('frame_count', frame_count),
                           ('activity', activity),
                           ('voxel_sizes_xyz', [voxel_size['x'],voxel_size['y'],voxel_size['z']])):
            if name in f:
                del f[name]
            f.create_dataset(name, data=data)
        f.attrs['scan_rate'] = scan_rate
    print(f'Saved min_proj, max_proj, sum_x, sum_x2, frame_count, std_image, activity to {h5_path}')

    # also save std_image, min_proj and max_proj as OME-TIFFs, calibrated with the
    # recording's voxel size, so they open correctly scaled in napari
    tiff_metadata = _tiff_metadata(voxel_size)
    for proj_type, data in (('std_image', std_image), ('min_proj', min_proj), ('max_proj', max_proj)):
        tiff_path = analysis_path / f'{proj_type}.ome.tif'
        tifffile.imwrite(tiff_path, data.transpose(0, 2, 1), ome=True, metadata=tiff_metadata)
    print(f'Saved std_image, min_proj, max_proj as OME-TIFFs to {analysis_path}')
    if median_proj is not None:
        _save_median(analysis_path, median_proj, voxel_size)

    # also plot and save the activity trace over time
    time_axis = np.arange(len(activity)) / scan_rate
    if activity_path is None:
        activity_path = analysis_path / 'activity.png'
    Path(activity_path).parent.mkdir(parents=True, exist_ok=True)
    _plot_activity(activity, time_axis, activity_path,
                  title=f'Average fluorescence over time ({scape_path.name})')


def get_median_projection(scape_path: str | Path,
                          batch_size: int | None = None) -> np.ndarray:
    """
    Compute only the median projection (median over time of every voxel) of a
    SCAPE 3D video, and add it to the analysis folder's projection_images.h5
    (`median_proj`, (z, x, y) like the other projections) and as
    `median_proj.ome.tif`. For runs whose other projections were computed
    without it (see `get_projection_images` to compute them all in one pass).

    Unlike the min/max/std, the median needs every volume for each voxel: the
    deskewed volumes are kept in RAM if they fit, else in a temporary memmap on
    the local disk (see `_MedianStack`), so this is meant for short recordings.

    Parameters
    ----------
    scape_path : str | Path
        A single `sample#_run#` folder (not a `sample#_combined` one).
    batch_size : int | None
        Number of volumes to load per batch. If None (default), chosen
        automatically from the volume size (see `get_auto_batch_size`).

    Returns
    -------
    np.ndarray
        The median projection (float32).
    """
    scape_path = Path(scape_path)
    if is_combined_path(scape_path):
        raise ValueError(f'{scape_path} is a sample#_combined folder, which has no raw data '
                         f'of its own.')
    check_folder_structure(scape_path)
    _, num_volumes, _ = scapeio.get_scan_timing(str(scape_path / scape_path.name))
    median_stack = None
    try:
        for start_volume, deskewed_batch in _deskewed_batches(
                scape_path, batch_size, f'Computing median projection ({scape_path.name})'):
            if median_stack is None:
                median_stack = _MedianStack(num_volumes, deskewed_batch.shape[1:],
                                            deskewed_batch.dtype, scape_path.name)
            median_stack.add(start_volume, deskewed_batch)
            del deskewed_batch
        median_proj = median_stack.median()
    finally:
        if median_stack is not None:
            median_stack.close()
    analysis_path = scape_path / ANALYSIS_PATH_NAME
    analysis_path.mkdir(exist_ok=True)
    _save_median(analysis_path, median_proj, load_voxel_size(scape_path))
    return median_proj

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
    lines marking the boundaries between runs). The names of the combined
    `sample#_run#` folders are saved as a `source_runs` attribute on the
    h5 file, in the same order they were combined.

    Only runs whose projections have the most common shape among `run_paths`
    are combined; any others (e.g. a short test scan with different scan
    settings) are skipped with a warning. If fewer than two runs remain, no
    `sample#_combined` folder is created.

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

    # only runs with the most common projection shape can be combined (e.g. a short
    # test scan with a different number of galvo steps is skipped)
    run_shapes = {}
    for scape_path in run_paths:
        with h5py.File(analysis_path_for(scape_path) / PROJECTION_FILE_NAME, 'r') as f:
            run_shapes[scape_path] = f['sum_x'].shape
    shape_counts = Counter(run_shapes.values())
    main_shape = max(shape_counts, key=shape_counts.get)
    skipped = [p for p in run_paths if run_shapes[p] != main_shape]
    for scape_path in skipped:
        print(f'Warning: skipping {scape_path.name} when combining {sample_name}: its '
              f'projection shape {run_shapes[scape_path]} differs from the other runs '
              f'({main_shape}).')
    run_paths = [p for p in run_paths if run_shapes[p] == main_shape]
    if len(run_paths) < 2:
        print(f'Fewer than two runs of {sample_name} with matching projection shapes, '
              f'not creating a {sample_name}_combined folder.')
        return

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
        h5_path = analysis_path_for(scape_path) / PROJECTION_FILE_NAME
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
        f.attrs['source_runs'] = [scape_path.name for scape_path in run_paths]
    print(f'Saved combined min_proj, max_proj, sum_x, sum_x2, frame_count, std_image, activity to {h5_path}')

    tiff_metadata = _tiff_metadata(load_voxel_size(run_paths[0]))
    for proj_type, data in (('std_image', std_image), ('min_proj', min_proj), ('max_proj', max_proj)):
        tiff_path = combined_path / f'{proj_type}.ome.tif'
        tifffile.imwrite(tiff_path, data.transpose(0, 2, 1), ome=True, metadata=tiff_metadata)
    print(f'Saved combined std_image, min_proj, max_proj as OME-TIFFs to {combined_path}')

    _plot_activity(activity, time_axis, combined_path / 'activity.png',
                  title=f'Average fluorescence over time ({sample_name} combined)',
                  run_boundaries=run_boundaries)

