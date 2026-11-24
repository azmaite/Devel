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
---- sample#_run#.ozx : (instead of the three above) the whole recording, losslessly
     compressed in a single file by `scapeio convert` (see RawRecording)
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
    data of its own. A run with a .ozx needs no _info.mat (it is inside the .ozx).
get_ozx_path: get the .ozx archive of a sample#_run# folder, if it has one
RawRecording: the raw volumes and metadata of a single run, read from its .ozx archive if it
    has one (only the volumes asked for, decoded in parallel; much faster from the server
    than the .dat files), else from its .dat files with scapeio. Volumes from both have
    camera rows cropped as `scapeio.load_scan` does, but a .ozx volume has 2 planes fewer
    (the flyback frames, which the .dat volumes include, see RawRecording), so projections
    from a .ozx and from .dat files of the same run differ in shape by 2 planes in x
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
    easy viewing (the activity PNG can be saved elsewhere, or skipped for long recordings,
    see `preprocess_scape_exp`). Optionally (`median=True`) also the
    median projection, in the same pass. Everything is computed on the raw volumes
    (uint16 data, exact uint64 sums, on several threads, loading the next batch while
    the current one is processed), and only the final projections are deskewed
    (more accurate, and much faster than deskewing every volume)
get_median_projection: compute only the median projection of a SCAPE 3D video and add it
    to its projection_images.h5 (and as an OME-TIFF), e.g. for runs whose other projections
    were computed before the median was added. The median is exact (over every volume) for
    runs under MEDIAN_MAX_VOLUMES volumes, and approximate (over MEDIAN_SAMPLES evenly spaced
    volumes) for longer ones; those raw volumes are kept in RAM if they fit (at most
    MEDIAN_MAX_RAM_BYTES), else the median is computed one x-section (range of planes) at a
    time, with one pass per section (see _median_sections; from a .ozx, a pass only reads its
    section's planes), so it stays exact
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
from concurrent.futures import ThreadPoolExecutor

import nrrd
import numpy as np
import psutil
import scapeio
import scapeio.operations
import tifffile
from scapeio.ozx import OzxArray
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
MEDIAN_MAX_RAM_BYTES = 8e9 # median stack held in RAM at once; bigger: split in x-sections
MEDIAN_RAM_FRACTION = 0.5 # ... and at most this fraction of the available RAM
MEDIAN_SLAB_BYTES = 1e9   # bytes of the median stack processed at a time
MEDIAN_MAX_VOLUMES = 75   # runs with fewer volumes get the exact median over all volumes
MEDIAN_SAMPLES = 50       # longer runs: approximate median from this many evenly spaced volumes
TUBE_LENS = '50 mm Navitar' # always!
STATS_THREADS = max(1, (os.cpu_count() or 4) - 2) # threads for the per-voxel reductions
CROP_ROWS = 2             # camera rows cropped at each edge of z' (as scapeio.load_scan does)


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
    structure is not correct and cannot be fixed. A run converted to
    `sample#_run#/sample#_run#.ozx` (see `get_ozx_path`) needs no
    `_info.mat`, which is inside the `.ozx`.

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
            # a run converted to .ozx has its _info.mat inside the .ozx
            if file_suffix == '_info.mat' and get_ozx_path(scape_path) is None:
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


def get_ozx_path(scape_path: str | Path) -> Path | None:
    """
    Get the `.ozx` archive of a run (`sample#_run#/sample#_run#.ozx`, as written by
    `scapeio convert <raw folder> -o <sample#_run# folder>`), if it has one.

    Parameters
    ----------
    scape_path : str | Path
        Path to a single `sample#_run#` folder.

    Returns
    -------
    Path or None
        Path to the `.ozx` file, or None if there is none.
    """
    scape_path = Path(scape_path)
    ozx_path = scape_path / f'{scape_path.name}.ozx'
    return ozx_path if ozx_path.is_file() else None


def _swap_deskewed_axes(images: np.ndarray) -> np.ndarray:
    """
    Reorder the spatial axes of volumes deskewed by `scapeio.operations.deskew_array`
    to (depth, galvo scan axis, camera image width), as `scapeio.operations.deskew_scan`
    does.
    """
    if images.ndim == 3:
        return images.swapaxes(0, 1)
    if images.ndim == 4:
        return images.swapaxes(1, 2)
    raise ValueError(f'Data has an unexpected shape: {images.shape}')


class RawRecording:
    """
    The raw volumes and metadata of a single `sample#_run#` recording, read from its
    `.ozx` archive if it has one (see `get_ozx_path`), else from its .dat files with
    scapeio.

    A `.ozx` is read with `scapeio.ozx.OzxArray`, which reads only the volumes asked for
    and decodes them in parallel, and holds all the metadata (so the run's `_info.mat`
    etc. are not needed). Its volumes are the .dat volumes with `CROP_ROWS` camera rows
    cropped at each edge (as `scapeio.load_scan` does) and without the 2 flyback frames
    of each volume: `.ozx` plane `p` is .dat plane `p + 1`, and a `.ozx` volume has 2
    planes fewer (along x, the galvo scan axis). Both are deskewed with the same
    `scapeio.operations.deskew_array`.

    Attributes
    ----------
    name : str
        Name of the run.
    ozx_path : Path or None
        The run's `.ozx`, or None if read from .dat files.
    num_volumes : int
        Number of volumes.
    scan_rate : float
        Volumes per second.
    volume_shape : tuple[int, int, int]
        Shape (x, z', y) of a raw volume from `load`.
    voxels_per_volume : int
        Raw voxels per volume, for `get_auto_batch_size`.
    voxel_size : dict
        `.ozx` only: deskewed voxel size in micrometers (keys 'z', 'x', 'y'; for .dat
        files see `load_voxel_size`).
    """

    def __init__(self, scape_path: str | Path):
        """
        Parameters
        ----------
        scape_path : str | Path
            Path to a single `sample#_run#` folder (folder structure already checked).
        """
        scape_path = Path(scape_path)
        self.name = scape_path.name
        self._raw_path = str(scape_path / scape_path.name)  # string needed for scapeio
        self.ozx_path = get_ozx_path(scape_path)
        if self.ozx_path is not None:
            self._ozx = OzxArray(self.ozx_path)
            meta = self._ozx.attrs['scape']
            if self._ozx.ndim != 4:
                raise ValueError(f'{self.ozx_path} is not a single-color recording '
                                 f'(shape {self._ozx.shape}).')
            if meta['calibration']['tube_lens'] != TUBE_LENS:
                raise ValueError(f"{self.ozx_path} was converted for the "
                                 f"{meta['calibration']['tube_lens']} tube lens, not {TUBE_LENS}.")
            pixel_size = meta['calibration']['pixel_size_um']
            dx = round(meta['scan']['dx_um'], 6)  # galvo step, from the recorded waveform
            self._raw_voxel_size = (dx, pixel_size["z'"], pixel_size['y'])  # (x, z', y)
            self._sheet_angle = meta['calibration']['light_sheet_angle_from_vertical_deg']
            self.num_volumes = self._ozx.shape[0]
            self.scan_rate = 1 / meta['scan']['volume_period_s']
            _, n_x, n_z, n_y = self._ozx.shape
            self.volume_shape = (n_x, n_z - 2 * CROP_ROWS, n_y)
            self.voxels_per_volume = int(np.prod(self.volume_shape))
            self.voxel_size = {'z': pixel_size['z'], 'x': dx, 'y': pixel_size['y']}
        else:
            _, self.num_volumes, self.scan_rate = scapeio.get_scan_timing(self._raw_path)
            scan_name = scapeio.get_scan_name(self._raw_path)
            daq_metadata = scapeio.load_matlab_metadata(scan_name)['daq']
            camera_metadata = scapeio.load_camera_metadata(scan_name)['data']
            self.volume_shape = (int(daq_metadata['pixelsPerLine']),
                                 int(camera_metadata['aoiheight']) - 2 * CROP_ROWS,
                                 int(camera_metadata['aoiwidth']))
            self.voxels_per_volume = int(np.prod(self.volume_shape))

    def load(self,
             start_volume: int,
             end_volume: int,
             x_range: tuple[int, int] | None = None) -> np.ndarray:
        """
        Load the raw volumes `start_volume` to `end_volume` (exclusive).

        Parameters
        ----------
        start_volume, end_volume : int
            The volumes to load (`end_volume` exclusive).
        x_range : tuple[int, int] | None
            Only load the planes `x_range[0]` to `x_range[1]` (exclusive) of each volume.
            From a `.ozx`, only those planes are read; from .dat files, the whole volumes
            are read and then cropped. Default None (all planes).

        Returns
        -------
        np.ndarray
            uint16 volumes (time, x, z', y), always with a time axis.
        """
        x_slice = slice(*x_range) if x_range is not None else slice(None)
        if self.ozx_path is not None:
            return self._ozx[start_volume:end_volume, x_slice][:, :, CROP_ROWS:-CROP_ROWS]
        video, axis_order = scapeio.load_scan(self._raw_path, colors=1,
                                              start_volume=start_volume,
                                              end_volume=end_volume,
                                              demotioned=False, try_loading_copies=None,
                                              return_axis_order=True)
        if 't' not in axis_order:
            # scapeio drops the time axis when loading a single volume; restore it
            video = video[np.newaxis]
        return video[:, x_slice]

    def deskew(self, video: np.ndarray) -> np.ndarray:
        """
        Deskew raw volumes from `load`, or a raw projection (x, z', y), as
        `scapeio.operations.deskew_scan` does.

        Returns
        -------
        np.ndarray
            Deskewed volumes (time, z, x, y), or projection (z, x, y).
        """
        if self.ozx_path is None:
            return scapeio.operations.deskew_scan(self._raw_path, TUBE_LENS, images=video)
        deskewed = scapeio.operations.deskew_array(video, self._raw_voxel_size,
                                                   sheet_angle_degrees=self._sheet_angle)
        return _swap_deskewed_axes(deskewed)


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

    if get_ozx_path(scape_path) is not None:
        return RawRecording(scape_path).voxel_size

    scape_raw_path = scape_path / scape_path.name
    voxel_size = scapeio.get_voxel_size(scape_raw_path, TUBE_LENS, deskewed=True)
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
    metadata or `.ozx`, see `RawRecording`), so the batch size is `AUTO_BATCH_VOXELS` divided by that,
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
    recording = RawRecording(scape_path)
    voxels_per_volume = recording.voxels_per_volume
    batch_size = int(round(AUTO_BATCH_VOXELS / voxels_per_volume))
    batch_size = max(1, min(batch_size, recording.num_volumes))
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


def _volume_ranges(volume_indices: np.ndarray,
                   batch_size: int) -> list[tuple[int, int]]:
    """
    Group sorted volume indices into ranges `(start, stop)` (stop exclusive) of
    consecutive volumes, each at most `batch_size` long (one per volume for
    indices that are not consecutive, e.g. the evenly spaced volumes of an
    approximate median).
    """
    ranges = []
    start = prev = int(volume_indices[0])
    for volume in volume_indices[1:]:
        volume = int(volume)
        if volume != prev + 1 or volume - start >= batch_size:
            ranges.append((start, prev + 1))
            start = volume
        prev = volume
    ranges.append((start, prev + 1))
    return ranges


def _raw_batches(recording: RawRecording,
                 ranges: list[tuple[int, int]],
                 desc: str,
                 x_range: tuple[int, int] | None = None):
    """
    Load raw batches of volumes (see `RawRecording.load`), loading the next
    batch in a background thread while the current one is being processed.

    Parameters
    ----------
    recording : RawRecording
        The recording.
    ranges : list[tuple[int, int]]
        The `(start, stop)` volumes of each batch (stop exclusive).
    desc : str
        Description of the progress bar.
    x_range : tuple[int, int] | None
        Only load these planes of each volume (see `RawRecording.load`).

    Yields
    ------
    tuple[int, np.ndarray]
        First volume of the batch, and the raw batch (time, x, z', y) (uint16).
    """
    source = recording.ozx_path.name if recording.ozx_path is not None else '.dat files'
    print(f'Reading {recording.name} from {source}')
    with ThreadPoolExecutor(max_workers=1) as prefetch:
        future = prefetch.submit(recording.load, *ranges[0], x_range)
        for i, (start, _) in enumerate(tqdm.tqdm(ranges, desc=desc)):
            batch = future.result()
            if i + 1 < len(ranges):
                future = prefetch.submit(recording.load, *ranges[i + 1], x_range)
            yield start, batch
            del batch


def _x_slabs(n_x: int) -> list[slice]:
    """
    Split `n_x` planes into one slab per stats thread (see `STATS_THREADS`).
    """
    bounds = np.linspace(0, n_x, min(n_x, STATS_THREADS) + 1).round().astype(int)
    return [slice(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]


class _RawProjections:
    """
    Max, min, sum and sum of squares over time of every raw voxel, and the
    activity trace (mean of the nonzero voxels of each volume), accumulated
    batch by batch from the raw uint16 volumes, without converting them to
    float. Max and min stay uint16, and the sums are uint64, which are exact
    and can't overflow (wrap around): a voxel's square is at most 65535^2 ~ 4.3e9,
    and uint64 holds up to ~1.8e19. Each batch is processed in one pass, split
    along x over `STATS_THREADS` threads (numpy releases the GIL).
    """

    def __init__(self, volume_shape: tuple):
        """
        Parameters
        ----------
        volume_shape : tuple
            Shape (x, z', y) of a raw volume.
        """
        self.max_proj = np.zeros(volume_shape, np.uint16)
        self.min_proj = np.full(volume_shape, np.iinfo(np.uint16).max, np.uint16)
        self.sum_x = np.zeros(volume_shape, np.uint64)
        self.sum_x2 = np.zeros(volume_shape, np.uint64)
        self.frame_count = 0
        self._activity = []
        self._slabs = _x_slabs(volume_shape[0])
        self._pool = ThreadPoolExecutor(max_workers=len(self._slabs))

    def _add_slab(self,
                  batch: np.ndarray,
                  x: slice) -> tuple[np.ndarray, np.ndarray]:
        """
        Accumulate the planes `x` of a batch; return each volume's sum and nonzero
        count over these planes.
        """
        slab = batch[:, x]
        np.maximum(self.max_proj[x], slab.max(axis=0), out=self.max_proj[x])
        np.minimum(self.min_proj[x], slab.min(axis=0), out=self.min_proj[x])
        np.add(self.sum_x[x], slab.sum(axis=0, dtype=np.uint64), out=self.sum_x[x])
        for volume in slab:
            square = volume.astype(np.uint32)
            square *= square
            np.add(self.sum_x2[x], square, out=self.sum_x2[x])
        spatial_axes = tuple(range(1, slab.ndim))
        return (slab.sum(axis=spatial_axes, dtype=np.uint64),
                np.count_nonzero(slab, axis=spatial_axes))

    def add(self, batch: np.ndarray) -> None:
        """
        Accumulate a raw batch (time, x, z', y) (uint16).
        """
        results = list(self._pool.map(lambda x: self._add_slab(batch, x), self._slabs))
        volume_sums = np.sum([r[0] for r in results], axis=0)
        nonzero_counts = np.sum([r[1] for r in results], axis=0)
        # per-volume mean fluorescence, ignoring zero voxels as if they were nan
        with np.errstate(invalid='ignore'):
            self._activity.append(volume_sums / nonzero_counts)
        self.frame_count += batch.shape[0]

    def std_image(self) -> np.ndarray:
        """
        Std over time of every raw voxel (float64).
        """
        mean = self.sum_x / self.frame_count
        mean_sq = self.sum_x2 / self.frame_count
        return np.sqrt(np.maximum(0, mean_sq - np.square(mean)))

    def activity(self) -> np.ndarray:
        """
        The activity trace (mean of the nonzero voxels of each volume).
        """
        return np.concatenate(self._activity)

    def close(self) -> None:
        """
        Stop the threads.
        """
        self._pool.shutdown()


def _median_volume_indices(num_volumes: int) -> np.ndarray:
    """
    Indices of the volumes the median projection is taken over: every volume
    for runs with fewer than `MEDIAN_MAX_VOLUMES` volumes, else
    `MEDIAN_SAMPLES` evenly spaced volumes (an approximate median, as the
    whole stack of a long run doesn't fit in memory).
    """
    if num_volumes < MEDIAN_MAX_VOLUMES:
        return np.arange(num_volumes)
    return np.unique(np.linspace(0, num_volumes - 1, MEDIAN_SAMPLES).round().astype(int))


def _median_sections(num_volumes: int,
                     volume_shape: tuple,
                     name: str) -> list[tuple[int, int]]:
    """
    Split the raw stack the median is taken over (`num_volumes` raw uint16
    volumes) into x-sections (ranges of planes) that each fit in RAM: at most
    `MEDIAN_MAX_RAM_BYTES`, and at most `MEDIAN_RAM_FRACTION` of the currently
    available RAM. The first section is filled during the pass that computes
    the other projections, and each further one takes one more pass (see
    `_median_by_sections`), which from a `.ozx` only reads that section's
    planes.

    Parameters
    ----------
    num_volumes : int
        Number of volumes the median is taken over.
    volume_shape : tuple
        Shape (x, z', y) of one raw volume.
    name : str
        Name of the recording (for messages).

    Returns
    -------
    list[tuple[int, int]]
        (start, stop) x-index of each section.
    """
    n_x = volume_shape[0]
    nbytes = num_volumes * int(np.prod(volume_shape)) * np.dtype(np.uint16).itemsize
    budget = min(MEDIAN_MAX_RAM_BYTES, MEDIAN_RAM_FRACTION * psutil.virtual_memory().available)
    num_sections = min(n_x, max(1, int(np.ceil(nbytes / budget))))
    bounds = np.linspace(0, n_x, num_sections + 1).round().astype(int)
    if num_sections == 1:
        print(f'Median of {name}: keeping the {nbytes / 1e9:.1f} GB stack in RAM.')
    else:
        print(f'Median of {name}: the {nbytes / 1e9:.1f} GB stack is split into '
              f'{num_sections} x-sections of {nbytes / num_sections / 1e9:.1f} GB.')
    return [(int(start), int(stop)) for start, stop in zip(bounds[:-1], bounds[1:])]


class _MedianStack:
    """
    One x-section (see `_median_sections`) of the raw volumes of a recording
    that the median is taken over (all, or a subsample, see
    `_median_volume_indices`), filled batch by batch and kept in RAM, to take
    their median over time voxel by voxel, a slab along x at a time
    (`MEDIAN_SLAB_BYTES`, on `STATS_THREADS` threads), to limit the memory of
    `np.median`.
    """

    def __init__(self,
                 volume_indices: np.ndarray,
                 volume_shape: tuple,
                 x_start: int,
                 x_stop: int):
        """
        Parameters
        ----------
        volume_indices : np.ndarray
            Sorted indices (in the recording) of the volumes to keep.
        volume_shape : tuple
            Shape (x, z', y) of one raw volume.
        x_start, x_stop : int
            The x-section of the volumes to keep.
        """
        self.volume_indices = np.asarray(volume_indices)
        self.x_start, self.x_stop = x_start, x_stop
        self.stack = np.empty((len(self.volume_indices), x_stop - x_start, *volume_shape[1:]),
                              np.uint16)

    def add(self,
            start_volume: int,
            batch: np.ndarray,
            batch_x_start: int = 0) -> None:
        """
        Store the x-section of the volumes to keep of a raw batch (time, x, z', y)
        starting at volume `start_volume`, whose planes start at `batch_x_start`
        (0 for whole volumes; `x_start` for a batch of just this section).
        """
        lo, hi = np.searchsorted(self.volume_indices, [start_volume, start_volume + batch.shape[0]])
        if hi > lo:
            x = slice(self.x_start - batch_x_start, self.x_stop - batch_x_start)
            self.stack[lo:hi] = batch[self.volume_indices[lo:hi] - start_volume, x]

    def median(self) -> np.ndarray:
        """
        Median over time of every voxel of the section (float32; for an even
        number of volumes, the mean of the two middle values).
        """
        num_volumes, n_x = self.stack.shape[:2]
        plane_bytes = num_volumes * int(np.prod(self.stack.shape[2:])) * self.stack.itemsize
        slab = max(1, int(MEDIAN_SLAB_BYTES // plane_bytes))
        median = np.empty(self.stack.shape[1:], np.float32)

        def median_slab(x):
            median[x:x + slab] = np.median(self.stack[:, x:x + slab], axis=0)

        with ThreadPoolExecutor(max_workers=STATS_THREADS) as pool:
            list(tqdm.tqdm(pool.map(median_slab, range(0, n_x, slab)),
                           total=len(range(0, n_x, slab)), desc='Median over time'))
        return median

    def close(self) -> None:
        """
        Free the stack.
        """
        del self.stack


def _median_by_sections(recording: RawRecording,
                        batch_size: int,
                        volume_indices: np.ndarray,
                        sections: list[tuple[int, int]],
                        median_proj: np.ndarray) -> None:
    """
    Fill in the raw median projection one x-section at a time, one pass over the
    volumes `volume_indices` per section, loading only the section's planes
    (from a `.ozx`; .dat files are read whole and cropped).

    Parameters
    ----------
    recording : RawRecording
        The recording.
    batch_size : int
        Number of volumes per batch.
    volume_indices : np.ndarray
        Sorted indices of the volumes the median is taken over.
    sections : list[tuple[int, int]]
        The x-sections to compute (see `_median_sections`).
    median_proj : np.ndarray
        Raw median projection (x, z', y) (float32), filled in place.
    """
    ranges = _volume_ranges(volume_indices, batch_size)
    for i, (x_start, x_stop) in enumerate(sections):
        median_stack = _MedianStack(volume_indices, recording.volume_shape, x_start, x_stop)
        try:
            desc = f'Median x-section {i + 1}/{len(sections)} ({recording.name})'
            for start_volume, batch in _raw_batches(recording, ranges, desc,
                                                    x_range=(x_start, x_stop)):
                median_stack.add(start_volume, batch, batch_x_start=x_start)
            median_proj[x_start:x_stop] = median_stack.median()
        finally:
            median_stack.close()


def _deskew_projection(recording: RawRecording,
                       projection: np.ndarray,
                       dtype: type = np.float32) -> np.ndarray:
    """
    Deskew a raw projection (x, z', y) to (z, x, y) (see `RawRecording.deskew`), in
    float64 so the deskew's interpolation isn't rounded, and return it as `dtype`.
    """
    return recording.deskew(projection.astype(np.float64)).astype(dtype)


def _save_median(analysis_path: Path,
                 median_proj: np.ndarray,
                 voxel_size: dict,
                 n_volumes: int) -> None:
    """
    Save the (deskewed) median projection in the analysis folder's
    projection_images.h5 (`median_proj`, with the number of volumes it was taken
    over as its `n_volumes` attribute) and as `median_proj.ome.tif`.
    """
    with h5py.File(analysis_path / PROJECTION_FILE_NAME, 'a') as f:
        if 'median_proj' in f:
            del f['median_proj']
        f.create_dataset('median_proj', data=median_proj)
        f['median_proj'].attrs['n_volumes'] = n_volumes
    tifffile.imwrite(analysis_path / 'median_proj.ome.tif', median_proj.transpose(0, 2, 1),
                     ome=True, metadata=_tiff_metadata(voxel_size))
    print(f'Saved median_proj to {analysis_path / PROJECTION_FILE_NAME} and as an OME-TIFF')


def get_projection_images(scape_path: str | Path, batch_size: int | None = None,
                          activity_path: str | Path | None = None,
                          median: bool = False,
                          plot_activity: bool = True):
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
    activity trace as `activity.png` (see `_plot_activity`), or at `activity_path`
    (unless `plot_activity` is False).
    With `median=True`, the median projection is computed in the same pass and
    saved too (`median_proj`: exact for runs under `MEDIAN_MAX_VOLUMES` volumes,
    else from `MEDIAN_SAMPLES` evenly spaced volumes, see `get_median_projection`).

    Everything is computed on the raw (not deskewed) volumes, read from the run's
    `.ozx` or .dat files (see `RawRecording`), which is more accurate than on
    deskewed ones (the deskew interpolates between planes), and much faster:
    only the final projections are deskewed (in float, see `_deskew_projection`),
    instead of every volume. The raw uint16 data is never converted to float:
    max/min stay uint16 and the sums are exact uint64 (see `_RawProjections`).
    The activity is the mean of each raw volume's nonzero voxels. The saved
    `sum_x`/`sum_x2` are deskewed (float64), for `combine_sample_projections`.

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
    activity_path : str | Path | None
        Where to save the activity plot. If None (default), `activity.png` in
        the analysis folder.
    median : bool
        Also compute the median projection, which needs the volumes it is taken
        over (all of them for short runs, `MEDIAN_SAMPLES` for longer ones) at
        at once. Its first x-section is filled in this pass; if the stack is
        too big for RAM (see `_median_sections`), each further x-section takes
        one more pass, which from a `.ozx` only reads that section's planes.
        Default False.
    plot_activity : bool
        Save the activity plot. Default True.
    """
    # set save path
    scape_path = Path(scape_path)
    if is_combined_path(scape_path):
        raise ValueError(f"{scape_path} is a sample#_combined folder, which has no raw "
                         f"data of its own - use combine_sample_projections instead.")
    check_folder_structure(scape_path)
    analysis_path = scape_path / ANALYSIS_PATH_NAME
    h5_path = analysis_path / PROJECTION_FILE_NAME

    if batch_size is None:
        batch_size = get_auto_batch_size(scape_path)
    recording = RawRecording(scape_path)
    num_volumes, scan_rate = recording.num_volumes, recording.scan_rate
    median_volume_indices = _median_volume_indices(num_volumes)
    median_sections: list[tuple[int, int]] = []
    median_stack: _MedianStack | None = None
    median_proj: np.ndarray | None = None
    if median:
        # this pass also fills the median's first x-section; any others take one
        # more pass each, below
        median_sections = _median_sections(len(median_volume_indices), recording.volume_shape,
                                           scape_path.name)
        median_proj = np.empty(recording.volume_shape, np.float32)
        median_stack = _MedianStack(median_volume_indices, recording.volume_shape,
                                    *median_sections[0])

    projections = _RawProjections(recording.volume_shape)
    ranges = [(start, min(start + batch_size, num_volumes))
              for start in range(0, num_volumes, batch_size)]
    try:
        for start_volume, batch in _raw_batches(
                recording, ranges, f'Computing projection images ({scape_path.name})'):
            if batch.shape[1:] != recording.volume_shape:
                raise ValueError(f'Volumes of {scape_path.name} have shape {batch.shape[1:]}, '
                                 f'not {recording.volume_shape} as expected from its metadata.')
            if median_stack is not None:
                median_stack.add(start_volume, batch)
            projections.add(batch)
        if median_stack is not None:
            x_start, x_stop = median_sections[0]
            median_proj[x_start:x_stop] = median_stack.median()
    finally:
        projections.close()
        if median_stack is not None:
            median_stack.close()
    if len(median_sections) > 1:
        _median_by_sections(recording, batch_size, median_volume_indices,
                            median_sections[1:], median_proj)

    # deskew only the final projections
    print(f'Deskewing the projections of {scape_path.name}...')
    max_proj = _deskew_projection(recording, projections.max_proj)
    min_proj = _deskew_projection(recording, projections.min_proj)
    std_image = _deskew_projection(recording, projections.std_image())
    sum_x = _deskew_projection(recording, projections.sum_x, np.float64)
    sum_x2 = _deskew_projection(recording, projections.sum_x2, np.float64)
    frame_count = projections.frame_count
    activity = projections.activity()

    # SAVE the results
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
        _save_median(analysis_path, _deskew_projection(recording, median_proj), voxel_size,
                     len(median_volume_indices))

    # also plot and save the activity trace over time
    if plot_activity:
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

    Unlike the min/max/std, the median needs all the volumes it is taken over
    at once: every volume for runs under `MEDIAN_MAX_VOLUMES` volumes, else
    `MEDIAN_SAMPLES` evenly spaced volumes (an approximate median; only those
    volumes are loaded). It is taken over the raw volumes, which are kept in RAM if
    they fit (see `_median_sections`), else one x-section at a time, with one pass
    per section (still exact; from a `.ozx`, each pass reads only its section's
    planes), and only the median is deskewed (see `_deskew_projection`).

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
        The (deskewed) median projection (float32).
    """
    scape_path = Path(scape_path)
    if is_combined_path(scape_path):
        raise ValueError(f'{scape_path} is a sample#_combined folder, which has no raw data '
                         f'of its own.')
    check_folder_structure(scape_path)
    if batch_size is None:
        batch_size = get_auto_batch_size(scape_path)
    recording = RawRecording(scape_path)
    volume_indices = _median_volume_indices(recording.num_volumes)
    if len(volume_indices) < recording.num_volumes:
        print(f'Approximate median projection of {scape_path.name} from '
              f'{len(volume_indices)} volumes')
    sections = _median_sections(len(volume_indices), recording.volume_shape, scape_path.name)
    median_proj = np.empty(recording.volume_shape, np.float32)
    _median_by_sections(recording, batch_size, volume_indices, sections, median_proj)
    median_proj = _deskew_projection(recording, median_proj)
    analysis_path = scape_path / ANALYSIS_PATH_NAME
    analysis_path.mkdir(exist_ok=True)
    _save_median(analysis_path, median_proj, load_voxel_size(scape_path), len(volume_indices))
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
            # float64, also when combining runs saved with float32 sums
            sum_x, sum_x2 = run_sum_x.astype(np.float64), run_sum_x2.astype(np.float64)
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

