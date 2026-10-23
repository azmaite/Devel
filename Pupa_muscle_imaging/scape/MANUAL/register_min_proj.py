"""
Script to rigidly register (3D rotation + translation, no scaling) the min projection of one
SCAPE run to the min projection of another run of the same sample, e.g. to bring two runs
recorded a bit apart in time into the same frame.
The moving run's min projection is resampled onto the fixed run's grid and saved in the
moving run's analysis folder as min_proj_aligned.ome.tif (calibrated with the voxel size, like
the other projection OME-TIFFs), together with the rigid transform as min_proj_aligned.tfm
(SimpleITK transform from fixed to moving physical space, see `register_min_proj`).

Registration is done in physical units (using `scape_utils.load_voxel_size`, so the anisotropic
voxels are handled correctly), with a correlation metric on a coarse-to-fine pyramid. Voxels
that are zero in either image (deskewing padding) are excluded from the metric.

Can be run from the terminal
(`python register_min_proj.py moving_run_folder fixed_run_folder`) or imported and called as a
function.

register_min_proj: register the min projection of the moving run to that of the fixed run,
    save the aligned min projection and the transform, and return the transform.
"""

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import tifffile

import scape_utils


def _load_min_proj_sitk(scape_path: Path) -> sitk.Image:
    """
    Load a run's min projection as a float32 SimpleITK image in the same ZYX
    orientation as the saved OME-TIFFs, with the voxel size as spacing.

    Parameters
    ----------
    scape_path : Path
        Path to a `sample#_run#` folder with projections already computed.

    Returns
    -------
    sitk.Image
        3D image, array axes (z, y, x), spacing in um.
    """
    min_proj = scape_utils.load_proj(scape_path, 'min_proj').transpose(0, 2, 1)
    voxel_size = scape_utils.load_voxel_size(scape_path)
    image = sitk.GetImageFromArray(np.ascontiguousarray(min_proj, dtype=np.float32))
    image.SetSpacing((voxel_size['x'], voxel_size['y'], voxel_size['z']))
    return image


def register_min_proj(moving_path: str | Path,
                      fixed_path: str | Path) -> sitk.Euler3DTransform:
    """
    Rigidly register the min projection of `moving_path` to that of `fixed_path`, and save
    the moving min projection resampled onto the fixed grid as `min_proj_aligned.ome.tif`
    (plus the transform as `min_proj_aligned.tfm`) in the moving run's analysis folder.

    Parameters
    ----------
    moving_path : str | Path
        Path to the `sample#_run#` folder to align.
    fixed_path : str | Path
        Path to the `sample#_run#` folder to align to.

    Returns
    -------
    sitk.Euler3DTransform
        Rigid transform mapping points (in um) of the fixed image to the moving image.
    """
    moving_path, fixed_path = Path(moving_path), Path(fixed_path)
    fixed = _load_min_proj_sitk(fixed_path)
    moving = _load_min_proj_sitk(moving_path)
    if fixed.GetSize() != moving.GetSize() or fixed.GetSpacing() != moving.GetSpacing():
        raise ValueError(f'{moving_path.name} and {fixed_path.name} have different grids '
                         f'({moving.GetSize()} vs {fixed.GetSize()}).')

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
        print(f'Correlation with {fixed_path.name} {label}: '
              f'{np.corrcoef(fixed_arr[valid], arr[valid])[0, 1]:.4f}')

    analysis_path = moving_path / scape_utils.ANALYSIS_PATH_NAME
    voxel_size = scape_utils.load_voxel_size(moving_path)
    tiff_metadata = {'axes': 'ZYX',
                     'PhysicalSizeX': voxel_size['x'], 'PhysicalSizeXUnit': 'um',
                     'PhysicalSizeY': voxel_size['y'], 'PhysicalSizeYUnit': 'um',
                     'PhysicalSizeZ': voxel_size['z'], 'PhysicalSizeZUnit': 'um'}
    tifffile.imwrite(analysis_path / 'min_proj_aligned.ome.tif',
                     sitk.GetArrayFromImage(aligned), ome=True, metadata=tiff_metadata)
    sitk.WriteTransform(rigid, str(analysis_path / 'min_proj_aligned.tfm'))
    print(f'Saved min_proj_aligned.ome.tif and min_proj_aligned.tfm to {analysis_path}')
    return rigid


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rigidly register the min projection of one '
                                                 'SCAPE run to that of another.')
    parser.add_argument('moving_path', help='sample#_run# folder to align.')
    parser.add_argument('fixed_path', help='sample#_run# folder to align to.')
    args = parser.parse_args()
    register_min_proj(args.moving_path, args.fixed_path)
