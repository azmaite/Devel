"""
Script to rigidly register (3D rotation + translation, no scaling) the median projection of one
SCAPE run to the median projection of another run of the same sample, e.g. to bring two runs
recorded a bit apart in time into the same frame.
The moving run's median projection is resampled onto the fixed run's grid and saved in the
moving run's analysis folder as median_proj_aligned.ome.tif (calibrated with the voxel size, like
the other projection OME-TIFFs), together with the rigid transform as median_proj_aligned.tfm
(SimpleITK transform from fixed to moving physical space, see `register_median_proj`).

Registration is done in physical units (using `scape_utils.load_voxel_size`, so the anisotropic
voxels are handled correctly), with a correlation metric on a coarse-to-fine pyramid. Voxels
that are zero in either image (deskewing padding) are excluded from the metric.

Either run can also be given as an analysis folder inside its `sample#_run#` folder (e.g. an
older `analysis_dat_old`) to use the projections there instead of the current analysis folder.

Can be run from the terminal
(`python register_median_proj.py moving_run_folder fixed_run_folder`) or imported and called as a
function.

register_median_proj: register the median projection of the moving run to that of the fixed run,
    save the aligned median projection and the transform, and return the transform and the
    correlation after alignment.
save_aligned: save an aligned median projection and its transform.
"""

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import tifffile

import scape_utils


def _run_and_analysis(path: Path) -> tuple[Path, Path]:
    """
    Split `path` (a `sample#_run#` folder, or an analysis folder inside one) into the run folder
    and the analysis folder to use.
    """
    if path.name.startswith(scape_utils.ANALYSIS_PATH_NAME):
        return path.parent, path
    return path, scape_utils.analysis_path_for(path)


def _load_median_proj_sitk(path: Path) -> sitk.Image:
    """
    Load a run's median projection as a float32 SimpleITK image in the same ZYX
    orientation as the saved OME-TIFFs, with the voxel size as spacing.

    Parameters
    ----------
    path : Path
        Path to a `sample#_run#` folder with projections already computed, or to an analysis
        folder inside one with `median_proj.ome.tif`.

    Returns
    -------
    sitk.Image
        3D image, array axes (z, y, x), spacing in um.
    """
    scape_path, analysis_path = _run_and_analysis(path)
    if analysis_path == scape_utils.analysis_path_for(scape_path):
        median_proj = scape_utils.load_proj(scape_path, 'median_proj').transpose(0, 2, 1)
    else:
        median_proj = tifffile.imread(analysis_path / 'median_proj.ome.tif')
    voxel_size = scape_utils.load_voxel_size(scape_path)
    image = sitk.GetImageFromArray(np.ascontiguousarray(median_proj, dtype=np.float32))
    image.SetSpacing((voxel_size['x'], voxel_size['y'], voxel_size['z']))
    return image


def register_median_proj(moving_path: str | Path,
                         fixed_path: str | Path,
                         save: bool = True) -> tuple[sitk.Euler3DTransform, float]:
    """
    Rigidly register the median projection of `moving_path` to that of `fixed_path`, and save
    the moving median projection resampled onto the fixed grid as `median_proj_aligned.ome.tif`
    (plus the transform as `median_proj_aligned.tfm`) in the moving run's analysis folder.

    Parameters
    ----------
    moving_path : str | Path
        Path to the `sample#_run#` folder to align, or to an analysis folder inside it.
    fixed_path : str | Path
        Path to the `sample#_run#` folder to align to, or to an analysis folder inside it.
    save : bool
        Whether to save the aligned median projection and the transform (default True).

    Returns
    -------
    sitk.Euler3DTransform
        Rigid transform mapping points (in um) of the fixed image to the moving image.
    float
        Correlation of the aligned and fixed median projections within their overlap.
    """
    fixed = _load_median_proj_sitk(Path(fixed_path))
    moving = _load_median_proj_sitk(Path(moving_path))
    moving_path, analysis_path = _run_and_analysis(Path(moving_path))
    fixed_path = _run_and_analysis(Path(fixed_path))[0]
    # different sizes are fine (registration is in um, the moving image is resampled onto the
    # fixed grid), e.g. runs with fewer scan steps
    if fixed.GetSpacing() != moving.GetSpacing():
        raise ValueError(f'{moving_path.name} and {fixed_path.name} have different voxel sizes '
                         f'({moving.GetSpacing()} vs {fixed.GetSpacing()}).')

    # masks excluding the zero padding left by deskewing
    fixed_mask = sitk.Cast(fixed > 0, sitk.sitkUInt8)
    moving_mask = sitk.Cast(moving > 0, sitk.sitkUInt8)

    initial = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                                sitk.CenteredTransformInitializerFilter.MOMENTS)

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsCorrelation()
    registration.SetMetricFixedMask(fixed_mask)
    registration.SetMetricMovingMask(moving_mask)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.1, seed=0)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0,
                                                          minStep=1e-4,
                                                          numberOfIterations=300,
                                                          relaxationFactor=0.6,
                                                          gradientMagnitudeTolerance=1e-8)
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    registration.SetInitialTransform(initial, inPlace=False)
    registration.AddCommand(sitk.sitkMultiResolutionIterationEvent,
                            lambda: print(f'  level done, metric {registration.GetMetricValue():.4f}'))

    print(f'Registering {moving_path.name} -> {fixed_path.name}...')
    transform = registration.Execute(fixed, moving)
    print(f'Final metric {registration.GetMetricValue():.4f} '
          f'({registration.GetOptimizerStopConditionDescription()})')

    rigid = sitk.Euler3DTransform(transform.GetNthTransform(0)
                                  if isinstance(transform, sitk.CompositeTransform) else transform)
    angles_deg = np.degrees((rigid.GetAngleX(), rigid.GetAngleY(), rigid.GetAngleZ()))
    print(f'Rotation (deg, about x/y/z in sitk order): {np.round(angles_deg, 2)}, '
          f'translation (um): {np.round(rigid.GetTranslation(), 2)}')

    aligned = sitk.Resample(moving, fixed, rigid, sitk.sitkLinear, 0.0, sitk.sitkFloat32)

    # correlation within the overlap, before and after
    fixed_arr = sitk.GetArrayFromImage(fixed)
    for label, image in (('before', moving), ('after', aligned)):
        arr = sitk.GetArrayFromImage(image)
        valid = (fixed_arr > 0) & (arr > 0)
        correlation = np.corrcoef(fixed_arr[valid], arr[valid])[0, 1]
        print(f'Correlation with {fixed_path.name} {label}: {correlation:.4f}')

    if save:
        save_aligned(aligned, rigid, analysis_path / 'median_proj_aligned')
    return rigid, correlation


def save_aligned(aligned: sitk.Image,
                 transform: sitk.Transform,
                 out_stem: Path) -> None:
    """
    Save an aligned median projection as `{out_stem}.ome.tif` (ZYX, calibrated with its spacing,
    like the other projection OME-TIFFs) and its transform as `{out_stem}.tfm`.

    Parameters
    ----------
    aligned : sitk.Image
        Median projection resampled onto the fixed grid.
    transform : sitk.Transform
        Transform from fixed to moving physical space.
    out_stem : Path
        Output path without extension.

    Returns
    -------
    None
    """
    spacing = aligned.GetSpacing()
    tiff_metadata = {'axes': 'ZYX',
                     'PhysicalSizeX': spacing[0], 'PhysicalSizeXUnit': 'um',
                     'PhysicalSizeY': spacing[1], 'PhysicalSizeYUnit': 'um',
                     'PhysicalSizeZ': spacing[2], 'PhysicalSizeZUnit': 'um'}
    tifffile.imwrite(f'{out_stem}.ome.tif', sitk.GetArrayFromImage(aligned), ome=True,
                     metadata=tiff_metadata)
    sitk.WriteTransform(transform, f'{out_stem}.tfm')
    print(f'Saved {out_stem.name}.ome.tif and .tfm to {out_stem.parent}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rigidly register the median projection of one '
                                                 'SCAPE run to that of another.')
    parser.add_argument('moving_path', help='sample#_run# folder to align.')
    parser.add_argument('fixed_path', help='sample#_run# folder to align to.')
    args = parser.parse_args()
    register_median_proj(args.moving_path, args.fixed_path)
