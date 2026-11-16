"""
Make videos (and slice overviews) to check a master reference and the hemibodies it was built
from, saved in `{master_path}/videos/{YYYYMMDD}`.

Volumes are shown as maximum-intensity projections along axis 0 (the depth of the hemibodies,
viewed from above, with the long anterior-posterior axis, axis 1, vertical), rotated about the
long axis (axis 1: the hemibodies were split by a plane containing it) from -45 to +45 degrees
and back, one sweep every `ROTATION_PERIOD_S` seconds (sinusoidally). Things flicker on and off
at `FLICKER_HZ` (half the period on, half off). All images are on the 1.5 um isotropic grid of
the hemibodies, so the projections are true to scale. All views are cropped to the non-zero
part of the reference or master (the grid is zero-padded). Videos are mp4 (H.264) at `FPS`.

Videos:
- `reference_segments_rotation.mp4`: the reference hemibody (round-1 template,
  `{reference}_image.tif` in `segment_masks`, gray) with its automatic muscle mask labelled by
  leg segment (one colour per segment: reds for the front leg, greens for the middle, blues for
  the hind; the head is left out) flickering on and off. The segments are those assigned by
  hand to the min-projection muscle clusters of the reference (`{recording}_segment_ids.json`
  in `TABLE_FOLDER`, `master_reference/segment_masks`): each voxel of the master's reference
  mask takes the segment of the nearest hand-labelled voxel
  (`build_master_mean.reference_segments`; the two are on the same grid, from the same split). The labels shown are those of the front-most
  labelled voxel along each line of sight,
- `master_hemibodies_rotation_overlay.mp4`: the master mean (`master_mean.tif`, magenta) with
  the registered hemibodies it averages (`{round_folder}/*_registered.tif`) flickering on, one
  after the other, in green (white where both are bright),
- `master_hemibodies_rotation_replace.mp4`: the same, but in gray, with each hemibody replacing
  the master while on (to compare them by blinking),
- `hemibodies_rotation_registered.mp4`, `hemibodies_rotation_rigid.mp4` and
  `hemibodies_rotation_rigid_vs_registered.mp4` (side by side), to compare before and after the
  registration: the hemibodies of the round, one after the other (each for one flicker period),
  without the master, either as registered (`{round_folder}/*_registered.tif`) or only rigidly
  aligned to the master: a rigid fit of the whole muscle mask of the hemibody to the round's
  template muscles (`reference_segments.tif`), from the same centroid translation as the
  registration (`deformation.register_hemibody`). The two videos are frame by frame in
  sync (same hemibody and angle) and use the same gray levels per hemibody. The rigid
  transforms are saved in `rigid_transforms.json` (and reused when rerun),
- `hemibodies_slices_{option}_rigid_vs_registered.mp4`: the same comparison (rigid left,
  registered right, no master) on the 6 slices of the slice option (see below), each slice
  cropped to where any hemibody has data,
- `muscle_labels_rotation.mp4`: the master mean (gray) with muscle label maps on the master
  grid shown one after the other, `MUSCLE_LAYER_S` (2 s) each: the master muscles split into
  muscles (`muscle_segmentation/{SPLIT_NAME}.tif`, see
  `master_reference_3d/muscle_segmentation/split_master_muscles.py`), the same
  split restricted to the segments compared with the CT (F femur, F tibia, M tibia, from its
  json), the split merged by CT muscle (`{SPLIT_NAME}_merged_to_ct.tif`) and the CT muscles
  mapped onto the master (`ct_alignment/ct_muscles_on_master.tif`, see
  `master_reference_3d/muscle_segmentation/register_ct_labels.py`),
  then the master alone for as long. The 10 s cycle is out of step with the 4 s rotation
  sweep (by half a sweep per cycle), so over the video (`MUSCLE_CYCLES` = 2 cycles, 20 s) each
  layer is seen at every angle. Labels are drawn
  at `SEGMENT_ALPHA` opacity, seen from the front (the label of the first labelled voxel along
  each line of sight), one colour per label id; the merged split and the CT muscles share their
  ids, so they share their colours, named in the legend while they are shown,
- `master_hemibodies_slices_{option}_{overlay|replace}.mp4`: the same two flickers on 6 single
  slices of the master in a 2x3 grid (no rotation). The slice options (`SLICE_OPTIONS`) are
  'depth' (6 slices along axis 0), 'per_axis' (2 along each axis) and 'long_axis' (6 cross-sections
  along axis 1), each placed at equal-mass quantiles of the master mean along its axis (so all
  are within the tissue). `--slice_pngs folder` saves one png per option, to choose one.

The hemibody captions name the hemibody shown. Gray levels are set per volume from the 99.7th
percentile of its projection (or slices), so the normalised registered images and the master
look alike.

Run from the terminal (`scapeio` conda environment):
`python make_master_videos.py --master_path .../master_reference/master_median --round_folder
registered_hemibodies_round3 [--videos rotate_reference rotate_master slices
rigid_vs_registered rigid_vs_registered_slices rotate_muscles] [--slices long_axis] [--slice_pngs folder]` (see `--help`).

make_master_videos: make the videos.
save_slice_pngs: save a png of the master slices of each slice option.
reference_segment_map: the reference muscle mask labelled by leg segment (no head).
video_rigid_vs_registered: the hemibodies before (rigid only) and after the registration.
video_rigid_vs_registered_slices: the same, on slices.
video_muscle_labels: the master with the muscle split and the CT muscles, one after the other.
"""

import argparse
import datetime
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk
import tifffile
from PIL import Image, ImageDraw, ImageFont

# the master reference modules (master_reference_3d/registration, master_mean_generation)
MASTER_REFERENCE_CODE = Path(__file__).parents[1] / 'master_reference_3d'
sys.path.insert(0, str(MASTER_REFERENCE_CODE / 'registration'))
sys.path.insert(0, str(MASTER_REFERENCE_CODE / 'master_mean_generation'))
import build_master_mean  # noqa: E402
import deformation as dfm  # noqa: E402
import register_segments as rs  # noqa: E402

TABLE_FOLDER = build_master_mean.REFERENCE_LABELS  # hand-made segment tables

FPS = 30
ROTATION_PERIOD_S = 4.0  # one -45 -> +45 -> -45 sweep
MAX_ANGLE_DEG = 45.0
FLICKER_HZ = 1.0
GRAY_PERCENTILE = 99.7
N_THREADS = 8
MAGENTA = np.array([1.0, 0.0, 1.0])
GREEN = np.array([0.0, 1.0, 0.0])
SEGMENT_ALPHA = 0.6
CAPTION_LINES = 4  # height of the caption band, in lines
SLICE_OPTIONS = ('depth', 'per_axis', 'long_axis')
AXIS_NAMES = {0: 'depth (axis 0)', 1: 'long axis (axis 1)', 2: 'left-right (axis 2)'}
SPLIT_NAME = 'master_muscle_split_h0.05_along9'  # muscle split shown by video_muscle_labels
CT_SEGMENTS = ('F_femur', 'F_tibia', 'M_tibia')  # segments of the split compared with the CT
MUSCLE_LAYER_S = 2.0  # seconds each label layer (and the master alone) is shown
MUSCLE_CYCLES = 2     # cycles of the layers in muscle_labels_rotation.mp4


def _frames_per_period() -> int:
    """
    Number of video frames in one rotation sweep (made even, so the sweep has a middle frame).

    Returns
    -------
    int
        Frames per `ROTATION_PERIOD_S`.
    """
    return 2 * round(ROTATION_PERIOD_S * FPS / 2)


def _rotation_angles() -> np.ndarray:
    """
    The distinct angles of a sweep: frame k of the period is at
    -MAX_ANGLE_DEG * cos(2 pi k / period), and frames k and period - k share an angle, so only
    the angles of frames 0 .. period / 2 are rendered (see `_angle_index`).

    Returns
    -------
    np.ndarray
        Angles in degrees, from -MAX_ANGLE_DEG to +MAX_ANGLE_DEG.
    """
    period = _frames_per_period()
    return -MAX_ANGLE_DEG * np.cos(2 * np.pi * np.arange(period // 2 + 1) / period)


def _angle_index(frame: int) -> int:
    """
    Index into `_rotation_angles()` of a video frame.

    Parameters
    ----------
    frame : int
        Frame number.

    Returns
    -------
    int
        Angle index.
    """
    period = _frames_per_period()
    k = frame % period
    return k if k <= period // 2 else period - k


def _flicker_on(frame: int) -> bool:
    """
    Whether the flickering layer is on in a frame (the first half of each flicker period).

    Parameters
    ----------
    frame : int
        Frame number.

    Returns
    -------
    bool
        True while on.
    """
    return (frame / FPS * FLICKER_HZ) % 1 < 0.5


def _bbox(image: np.ndarray,
          margin: int = 8) -> tuple:
    """
    Bounding box of the non-zero voxels (or pixels) of an array, with a margin, to crop away
    the zero padding of the grid.

    Parameters
    ----------
    image : np.ndarray
        Array.
    margin : int
        Margin added on each side, in voxels.

    Returns
    -------
    tuple of slice
        One slice per axis.
    """
    box = []
    for axis in range(image.ndim):
        nonzero = np.flatnonzero(np.any(image != 0, axis=tuple(a for a in range(image.ndim)
                                                                 if a != axis)))
        start, stop = (nonzero[0], nonzero[-1] + 1) if nonzero.size else (0, image.shape[axis])
        box.append(slice(max(start - margin, 0), min(stop + margin, image.shape[axis])))
    return tuple(box)


def _rotated_mips(volume: np.ndarray,
                  order: int = 1) -> list:
    """
    Maximum-intensity projections along axis 0 of a volume rotated about axis 1, at each angle
    of `_rotation_angles()`, padded (centred) to a common width.

    Parameters
    ----------
    volume : np.ndarray
        Volume (axis 0: depth, axis 1: long axis, axis 2: left-right).
    order : int
        Interpolation order (1 for images, 0 for masks).

    Returns
    -------
    list of np.ndarray
        One float32 projection (axis 1 x rotated axis 2) per angle.
    """
    chunks = np.array_split(volume.astype(np.float32), N_THREADS, axis=1)
    mips = []
    with ThreadPoolExecutor(N_THREADS) as pool:
        for angle in _rotation_angles():
            parts = pool.map(lambda chunk: ndi.rotate(chunk, angle, axes=(0, 2), reshape=True,
                                                      order=order).max(axis=0), chunks)
            mips.append(np.concatenate(list(parts), axis=0))
    width = max(mip.shape[1] for mip in mips)
    return [np.pad(mip, [(0, 0), ((width - mip.shape[1]) // 2,
                                  width - mip.shape[1] - (width - mip.shape[1]) // 2)])
            for mip in mips]


def _rotated_front_labels(labels: np.ndarray) -> list:
    """
    Labels of a label volume rotated as in `_rotated_mips`, seen from the front: for each line of
    sight (along axis 0), the label of the first labelled voxel.

    Parameters
    ----------
    labels : np.ndarray
        Integer label volume.

    Returns
    -------
    list of np.ndarray
        One label image per angle of `_rotation_angles()`, padded like `_rotated_mips`.
    """
    def front(chunk, angle):
        rotated = ndi.rotate(chunk, angle, axes=(0, 2), reshape=True, order=0)
        first = np.argmax(rotated > 0, axis=0)
        return np.take_along_axis(rotated, first[None], axis=0)[0]
    dtype = np.uint8 if labels.max() < 256 else np.uint16
    chunks = np.array_split(labels.astype(dtype), N_THREADS, axis=1)
    images = []
    with ThreadPoolExecutor(N_THREADS) as pool:
        for angle in _rotation_angles():
            images.append(np.concatenate(list(pool.map(lambda c: front(c, angle), chunks)),
                                         axis=0))
    width = max(image.shape[1] for image in images)
    return [np.pad(image, [(0, 0), ((width - image.shape[1]) // 2,
                                    width - image.shape[1] - (width - image.shape[1]) // 2)])
            for image in images]


def _segment_colours(segment_names: list) -> np.ndarray:
    """
    One RGB colour per leg segment: shades of red for the front leg (`F_...`), green for the
    middle (`M_...`) and blue for the hind leg (`H_...`), lighter towards the body.

    Parameters
    ----------
    segment_names : list of str
        Segment names, in label order.

    Returns
    -------
    np.ndarray
        (number of segments + 1, 3) colours in 0..1, row 0 (no label) black.
    """
    maps = {'F': plt.get_cmap('Reds'), 'M': plt.get_cmap('Greens'), 'H': plt.get_cmap('Blues')}
    colours = np.zeros((len(segment_names) + 1, 3))
    for leg, cmap in maps.items():
        members = [i for i, name in enumerate(segment_names, start=1) if name.startswith(leg)]
        for rank, index in enumerate(members):
            colours[index] = cmap(np.linspace(0.4, 0.95, max(len(members), 2))[rank])[:3]
    other = [i for i, name in enumerate(segment_names, start=1) if name[0] not in maps]
    for rank, index in enumerate(other):
        colours[index] = plt.get_cmap('Purples')(0.5 + 0.4 * rank / max(len(other), 1))[:3]
    return colours


def _legend(height: int,
            entries: list) -> np.ndarray:
    """
    A black strip with a colour legend, to put to the right of a frame.

    Parameters
    ----------
    height : int
        Height of the strip (the frame's).
    entries : list of (str, colour)
        Legend lines, each with an RGB colour (0..1).

    Returns
    -------
    np.ndarray
        uint8 RGB strip, even width.
    """
    font = _font(16)
    width = 2 * ((max(int(font.getlength(text)) for text, _ in entries) + 40) // 2)
    image = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(image)
    y = 10
    for text, colour in entries:
        fill = tuple(int(255 * c) for c in colour)
        draw.rectangle([8, y + 3, 22, y + 17], fill=fill)
        draw.text((28, y), text, font=font, fill=(255, 255, 255))
        y += 22
    return np.asarray(image)


def reference_segment_map(master_path: Path,
                          reference: str,
                          table_folder: Path = TABLE_FOLDER) -> tuple:
    """
    The automatic muscle mask of the reference hemibody (`segment_masks` of `master_path`),
    labelled by leg segment from the hand-made segment table (see the module docstring), without
    the head (`deformation.IGNORED_SEGMENTS`).

    Parameters
    ----------
    master_path : Path
        Master folder (with `segment_masks`).
    reference : str
        Reference hemibody name (`{recording}_{side}`).
    table_folder : Path
        Folder with `{recording}_segment_ids.json` and the `_components.tif` it refers to.

    Returns
    -------
    tuple
        (uint8 segment map on the hemibody grid, 1..; list of segment names in label order).
    """
    mask = tifffile.imread(master_path / 'segment_masks' / f'{reference}_components.tif') > 0
    segments, names = build_master_mean.reference_segments(mask, reference, table_folder)
    return segments.astype(np.uint8), names


def _gray(image: np.ndarray,
          vmax: float) -> np.ndarray:
    """
    Scale an image to 0..1.

    Parameters
    ----------
    image : np.ndarray
        Image.
    vmax : float
        Value shown as white.

    Returns
    -------
    np.ndarray
        Clipped scaled image (float32).
    """
    return np.clip(image / vmax, 0, 1).astype(np.float32)


def _vmax(images: list) -> float:
    """
    White level of a set of projections or slices: the `GRAY_PERCENTILE` of their non-zero
    values.

    Parameters
    ----------
    images : list of np.ndarray
        Images.

    Returns
    -------
    float
        White level.
    """
    values = np.concatenate([image[image > 0].ravel() for image in images])
    return float(np.percentile(values, GRAY_PERCENTILE)) if values.size else 1.0


def _font(size: int) -> ImageFont.FreeTypeFont:
    """
    The DejaVu Sans font shipped with matplotlib, at a size.

    Parameters
    ----------
    size : int
        Font size, in pixels.

    Returns
    -------
    ImageFont.FreeTypeFont
        The font.
    """
    return ImageFont.truetype(font_manager.findfont('DejaVu Sans'), size)


def _to_frame(rgb: np.ndarray,
              lines: list) -> np.ndarray:
    """
    Turn an RGB image (0..1) into a video frame: uint8, with the caption lines in a black band
    above the image (always `CAPTION_LINES` high, so all frames have the same size) and even
    height and width (for H.264).

    Parameters
    ----------
    rgb : np.ndarray
        RGB image, floats in 0..1.
    lines : list of (str, colour)
        Caption lines, each with an RGB colour (0..1).

    Returns
    -------
    np.ndarray
        uint8 frame.
    """
    font = _font(int(np.clip(rgb.shape[1] / 45, 12, 22)))
    band = 8 + int(font.size * 1.3) * CAPTION_LINES
    rgb = np.pad(rgb, [(band, (band + rgb.shape[0]) % 2), (0, rgb.shape[1] % 2), (0, 0)])
    image = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    y = 4
    for text, colour in lines:
        draw.text((8, y), text, font=font,
                  fill=tuple(int(255 * c) for c in colour), stroke_width=2, stroke_fill=(0, 0, 0))
        y += int(font.size * 1.3)
    return np.asarray(image)


def _write_video(path: Path,
                 frames) -> None:
    """
    Write frames to an mp4 (H.264, `FPS`).

    Parameters
    ----------
    path : Path
        Output path.
    frames : iterable of np.ndarray
        uint8 RGB frames, all the same (even) shape.

    Returns
    -------
    None
    """
    with imageio.get_writer(path, fps=FPS, codec='libx264', quality=8, macro_block_size=2,
                            pixelformat='yuv420p') as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f'Saved {path}')


def _n_frames(n_hemibodies: int) -> int:
    """
    Length of a hemibody-flicker rotation video: the cycle through all hemibodies (one flicker
    period each), repeated until the video also holds a whole number of rotation sweeps (so each
    hemibody is seen at several angles).

    Parameters
    ----------
    n_hemibodies : int
        Number of hemibodies.

    Returns
    -------
    int
        Number of frames.
    """
    flicker_frames = round(FPS / FLICKER_HZ)
    cycle = n_hemibodies * flicker_frames
    return cycle * _frames_per_period() // math.gcd(cycle, _frames_per_period())


def _hemibody_paths(round_folder: Path) -> list:
    """
    Registered hemibody images of a round.

    Parameters
    ----------
    round_folder : Path
        Output folder of a registration round.

    Returns
    -------
    list of Path
        `{name}_registered.tif` paths, sorted.
    """
    return sorted(p for p in round_folder.glob('*_registered.tif'))


def _hemibody_name(path: Path) -> str:
    """
    Hemibody name of a `{name}_registered.tif` path.

    Parameters
    ----------
    path : Path
        Path.

    Returns
    -------
    str
        `{name}`.
    """
    return path.name.removesuffix('_registered.tif')


def video_reference_segments(master_path: Path,
                             reference: str,
                             out_folder: Path,
                             table_folder: Path = TABLE_FOLDER) -> None:
    """
    `reference_segments_rotation.mp4` (see the module docstring).

    Parameters
    ----------
    master_path : Path
        Master folder (with `segment_masks`).
    reference : str
        Reference hemibody name.
    out_folder : Path
        Output folder.
    table_folder : Path
        Folder of the hand-made segment table (see `reference_segment_map`).

    Returns
    -------
    None
    """
    image = tifffile.imread(master_path / 'segment_masks' / f'{reference}_image.tif')
    segments, segment_names = reference_segment_map(master_path, reference, table_folder)
    crop = _bbox(image)
    image, segments = image[crop], segments[crop]
    print(f'Rendering {reference} and its segments...')
    image_mips = _rotated_mips(image)
    segment_views = _rotated_front_labels(segments)
    vmax = _vmax(image_mips)
    colours = _segment_colours(segment_names)
    legend_entries = [(name, colours[i]) for i, name in enumerate(segment_names, start=1)]

    def frames():
        for frame in range(2 * _frames_per_period()):
            index = _angle_index(frame)
            rgb = np.repeat(_gray(image_mips[index], vmax)[..., None], 3, axis=-1)
            lines = [(f'reference hemibody: {reference}', (1, 1, 1)),
                     (f'rotation about the long axis: {_rotation_angles()[index]:+.0f} deg',
                      (1, 1, 1))]
            if _flicker_on(frame):
                labels = segment_views[index]
                alpha = SEGMENT_ALPHA * (labels > 0)[..., None]
                rgb = rgb * (1 - alpha) + alpha * colours[labels]
                lines.append(('+ muscle mask by leg segment (no head)', (1, 1, 1)))
            legend = _legend(rgb.shape[0], legend_entries).astype(np.float32) / 255
            yield _to_frame(np.concatenate([rgb, legend], axis=1), lines)
    _write_video(out_folder / 'reference_segments_rotation.mp4', frames())


def _label_colours(n_labels: int,
                   seed: int = 0) -> np.ndarray:
    """
    One bright, random RGB colour per label id (fixed `seed`, so the same id gets the same
    colour in every layer).

    Parameters
    ----------
    n_labels : int
        Highest label id.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        (n_labels + 1, 3) colours in 0..1, row 0 (no label) black.
    """
    rng = np.random.default_rng(seed)
    hsv = np.stack([rng.random(n_labels + 1), rng.uniform(0.6, 1.0, n_labels + 1),
                    rng.uniform(0.75, 1.0, n_labels + 1)], axis=-1)
    colours = matplotlib.colors.hsv_to_rgb(hsv)
    colours[0] = 0
    return colours


def _muscle_layers(master_path: Path) -> list:
    """
    The label layers of `muscle_labels_rotation.mp4` (see the module docstring), on the master
    grid.

    Parameters
    ----------
    master_path : Path
        Master folder (with `muscle_segmentation` and `ct_alignment`).

    Returns
    -------
    list of dict
        One per layer: `caption`, `labels` (label volume) and `names` (label id -> name, for
        the legend, or None).
    """
    split_folder = master_path / 'muscle_segmentation'
    split = tifffile.imread(split_folder / f'{SPLIT_NAME}.tif')
    split_info = json.loads((split_folder / f'{SPLIT_NAME}.json').read_text())
    ct_labels = [int(label) for label, segment in split_info['label_to_segment'].items()
                 if segment in CT_SEGMENTS]
    merged_info = json.loads((split_folder / f'{SPLIT_NAME}_merged_to_ct.json').read_text())
    ct_info = json.loads((master_path / 'ct_alignment' / 'ct_muscles_on_master.json')
                         .read_text())
    return [{'caption': f'muscle split ({SPLIT_NAME})', 'labels': split, 'names': None},
            {'caption': f'muscle split, {", ".join(s.replace("_", " ") for s in CT_SEGMENTS)} '
                        f'only', 'labels': np.where(np.isin(split, ct_labels), split, 0),
             'names': None},
            {'caption': 'muscle split merged by CT muscle',
             'labels': tifffile.imread(split_folder / f'{SPLIT_NAME}_merged_to_ct.tif'),
             'names': merged_info['id_to_name']},
            {'caption': 'CT muscles mapped onto the master (ct_muscles_on_master)',
             'labels': tifffile.imread(master_path / 'ct_alignment' / 'ct_muscles_on_master.tif'),
             'names': ct_info['id_to_name']}]


def video_muscle_labels(master_path: Path,
                        out_folder: Path) -> None:
    """
    `muscle_labels_rotation.mp4` (see the module docstring).

    Parameters
    ----------
    master_path : Path
        Master folder (`master_mean.tif`, `muscle_segmentation`, `ct_alignment`).
    out_folder : Path
        Output folder.

    Returns
    -------
    None
    """
    mean = tifffile.imread(master_path / 'master_mean.tif')
    layers = _muscle_layers(master_path)
    for layer in layers:
        if layer['labels'].shape != mean.shape:
            raise ValueError(f'{layer["caption"]}: shape {layer["labels"].shape}, not on the '
                             f'master grid {mean.shape}')
    crop = _bbox(mean)
    print('Rendering the master and the muscle labels...')
    mean_mips = _rotated_mips(mean[crop])
    vmax = _vmax(mean_mips)
    for layer in layers:
        layer['views'] = _rotated_front_labels(layer['labels'][crop])
    colours = _label_colours(max(int(layer['labels'].max()) for layer in layers))
    height = mean_mips[0].shape[0]
    legends = [_legend(height, [(f'{int(i)}: {name}', colours[int(i)])
                                for i, name in layer['names'].items()])
               if layer['names'] else None for layer in layers]
    width = max(legend.shape[1] for legend in legends if legend is not None)
    legends = [np.pad(legend if legend is not None else np.zeros((height, 0, 3), np.uint8),
                      [(0, 0), (0, width - (legend.shape[1] if legend is not None else 0)),
                       (0, 0)]).astype(np.float32) / 255 for legend in legends]
    layer_frames = round(MUSCLE_LAYER_S * FPS)
    n_steps = len(layers) + 1  # the layers, then the master alone

    def frames():
        for frame in range(MUSCLE_CYCLES * n_steps * layer_frames):
            index = _angle_index(frame)
            step = (frame // layer_frames) % n_steps
            rgb = np.repeat(_gray(mean_mips[index], vmax)[..., None], 3, axis=-1)
            lines = [('median master mean', (1, 1, 1)),
                     (f'rotation about the long axis: {_rotation_angles()[index]:+.0f} deg',
                      (1, 1, 1))]
            if step < len(layers):
                labels = layers[step]['views'][index]
                alpha = SEGMENT_ALPHA * (labels > 0)[..., None]
                rgb = rgb * (1 - alpha) + alpha * colours[labels]
                lines.append((f'+ {layers[step]["caption"]}', (1, 1, 1)))
                legend = legends[step]
            else:
                lines.append(('(master alone)', (1, 1, 1)))
                legend = np.zeros((height, width, 3), np.float32)
            yield _to_frame(np.concatenate([rgb, legend], axis=1), lines)
    _write_video(out_folder / 'muscle_labels_rotation.mp4', frames())


def _flicker_frames(master_images: list,
                    hemibody_images: list,
                    names: list,
                    index_of_frame,
                    mode: str,
                    header: list,
                    n_frames: int):
    """
    Frames of a hemibody-flicker video: the master, with each hemibody in turn flickering on
    (overlaid in green on the magenta master, or replacing the gray master).

    Parameters
    ----------
    master_images : list of np.ndarray
        Master images (gray 0..1), indexed by `index_of_frame(frame)`.
    hemibody_images : list of list of np.ndarray
        Per hemibody, its images (gray 0..1), indexed the same way.
    names : list of str
        Hemibody names.
    index_of_frame : callable
        Frame -> image index (the angle; 0 for still images).
    mode : 'overlay' or 'replace'
        Flicker style.
    header : callable
        Frame -> list of caption lines shown on every frame.
    n_frames : int
        Number of frames.

    Yields
    ------
    np.ndarray
        uint8 frames.
    """
    flicker_frames = round(FPS / FLICKER_HZ)
    for frame in range(n_frames):
        index = index_of_frame(frame)
        hemibody = (frame // flicker_frames) % len(names)
        master = master_images[index]
        lines = header(frame)
        if mode == 'overlay':
            rgb = master[..., None] * MAGENTA
            lines = lines + [('master mean', MAGENTA)]
            if _flicker_on(frame):
                rgb = rgb + hemibody_images[hemibody][index][..., None] * GREEN
                lines.append((f'+ {names[hemibody]}', GREEN))
        else:
            on = _flicker_on(frame)
            gray = hemibody_images[hemibody][index] if on else master
            rgb = np.repeat(gray[..., None], 3, axis=-1)
            lines = lines + [(names[hemibody] if on else 'master mean',
                              GREEN if on else MAGENTA)]
        yield _to_frame(rgb, lines)


def video_master_rotation(master_path: Path,
                          round_folder: Path,
                          out_folder: Path) -> None:
    """
    `master_hemibodies_rotation_{overlay,replace}.mp4` (see the module docstring).

    Parameters
    ----------
    master_path : Path
        Master folder (with `master_mean.tif`).
    round_folder : Path
        Round whose registered hemibodies are shown.
    out_folder : Path
        Output folder.

    Returns
    -------
    None
    """
    print('Rendering the master mean...')
    master = tifffile.imread(master_path / 'master_mean.tif')
    crop = _bbox(master)  # all volumes are on the master grid
    master_mips = _rotated_mips(master[crop])
    del master
    master_gray = [_gray(m, _vmax(master_mips)) for m in master_mips]
    paths = _hemibody_paths(round_folder)
    names, hemibody_gray = [], []
    for path in paths:
        print(f'Rendering {path.name}...')
        mips = _rotated_mips(tifffile.imread(path)[crop])
        vmax = _vmax(mips)
        hemibody_gray.append([_gray(m, vmax) for m in mips])
        names.append(_hemibody_name(path))

    def header(frame):
        return [(f'{master_path.name} master, {len(names)} hemibodies', (1, 1, 1)),
                (f'rotation about the long axis: '
                 f'{_rotation_angles()[_angle_index(frame)]:+.0f} deg', (1, 1, 1))]
    for mode in ('overlay', 'replace'):
        _write_video(out_folder / f'master_hemibodies_rotation_{mode}.mp4',
                     _flicker_frames(master_gray, hemibody_gray, names, _angle_index, mode,
                                     header, _n_frames(len(names))))


def _rigid_aligned(master_path: Path,
                   round_folder: Path,
                   name: str,
                   shape: tuple,
                   transforms: dict) -> np.ndarray:
    """
    A hemibody only rigidly aligned to the round's template muscles (see the module docstring),
    normalised as the registered hemibodies (`deformation.normalise`), on the round's
    grid, 0 where it has no data.

    Parameters
    ----------
    master_path : Path
        Master folder (with `segment_masks`).
    round_folder : Path
        Registration round (with `reference_segments.tif`).
    name : str
        Hemibody name.
    shape : tuple of int
        Grid shape (at least the round's grid and the hemibody's shape).
    transforms : dict
        Name -> rigid transform (`deformation.affine_dict`, target -> template), read
        from and added to (the fit is skipped if `name` is in it).

    Returns
    -------
    np.ndarray
        float32 volume of shape `shape`.
    """
    folder = master_path / 'segment_masks'
    image = dfm.pad(tifffile.imread(folder / f'{name}_image.tif').astype(np.float32), shape)
    components = dfm.pad(tifffile.imread(folder / f'{name}_components.tif'), shape)
    if name in transforms:
        rigid = dfm.affine_from_dict(transforms[name])
    else:
        template_muscles = dfm.pad(tifffile.imread(round_folder / 'reference_segments.tif'),
                                   shape) > 0
        muscles = components > 0
        shift = dfm.physical_centroid(template_muscles) - dfm.physical_centroid(muscles)
        initial = sitk.TranslationTransform(3, shift.tolist())
        print(f'Fitting a rigid alignment of {name}...')
        rigid = rs.register_segment_shape(template_muscles, muscles, model='rigid',
                                          initial=initial, clip_um=45.0,
                                          target_valid=image > 0)
        transforms[name] = dfm.affine_dict(rigid)
    warped, view = dfm.warp_blended(
        [dfm.normalise(image, components > 0), (image > 0).astype(np.float32)],
        [rs.invert_affine(rigid)], [1.0], shape)
    return np.where(view > 0.5, warped / np.maximum(view, 1e-6), 0).astype(np.float32)


def _registered_and_rigid(master_path: Path,
                          round_folder: Path,
                          out_folder: Path) -> tuple:
    """
    The hemibodies of a round, registered (`*_registered.tif`) and only rigidly aligned (see
    `_rigid_aligned`; the rigid transforms are read from and saved to
    `out_folder / 'rigid_transforms.json'`), on a common grid (the round's grid, padded at the
    high end if a hemibody is larger).

    Parameters
    ----------
    master_path : Path
        Master folder (with `segment_masks`).
    round_folder : Path
        Registration round.
    out_folder : Path
        Folder of `rigid_transforms.json`.

    Returns
    -------
    tuple
        (list of hemibody names; list of registered volumes; list of rigid volumes).
    """
    paths = _hemibody_paths(round_folder)
    names = [_hemibody_name(path) for path in paths]
    grid = tifffile.TiffFile(round_folder / 'reference_segments.tif').series[0].shape
    shapes = [grid] + [tifffile.TiffFile(master_path / 'segment_masks' / f'{name}_image.tif')
                       .series[0].shape for name in names]
    shape = tuple(int(s) for s in np.max(shapes, axis=0))
    transforms_path = out_folder / 'rigid_transforms.json'
    transforms = (json.loads(transforms_path.read_text()) if transforms_path.exists() else {})
    registered = [dfm.pad(tifffile.imread(path), shape) for path in paths]
    rigid = []
    for name in names:
        rigid.append(_rigid_aligned(master_path, round_folder, name, shape, transforms))
        transforms_path.write_text(json.dumps(transforms, indent=1))
    return names, registered, rigid


def _side_by_side(panels: list) -> np.ndarray:
    """
    Put video frames side by side, 8 black pixels apart.

    Parameters
    ----------
    panels : list of np.ndarray
        uint8 frames of the same height.

    Returns
    -------
    np.ndarray
        The combined frame.
    """
    gap = np.zeros((panels[0].shape[0], 8, 3), np.uint8)
    return np.concatenate(sum(([panel, gap] for panel in panels), [])[:-1], axis=1)


def video_rigid_vs_registered(master_path: Path,
                              round_folder: Path,
                              out_folder: Path) -> None:
    """
    `hemibodies_rotation_{registered,rigid,rigid_vs_registered}.mp4` (see the module docstring).

    Parameters
    ----------
    master_path : Path
        Master folder (with `segment_masks`).
    round_folder : Path
        Round whose hemibodies are shown.
    out_folder : Path
        Output folder (also gets `rigid_transforms.json`).

    Returns
    -------
    None
    """
    names, registered, rigid = _registered_and_rigid(master_path, round_folder, out_folder)
    crop = _bbox(np.any([v != 0 for v in registered + rigid], axis=0))

    gray = {'registered': [], 'rigid': []}
    for name, volume, rigid_volume in zip(names, registered, rigid):
        print(f'Rendering {name} (registered and rigid)...')
        mips = _rotated_mips(volume[crop])
        rigid_mips = _rotated_mips(rigid_volume[crop])
        vmax = _vmax(mips)  # the same gray levels for both
        gray['registered'].append([_gray(m, vmax) for m in mips])
        gray['rigid'].append([_gray(m, vmax) for m in rigid_mips])
    del registered, rigid
    titles = {'registered': 'registered (segment by segment) to the master',
              'rigid': 'rigid alignment only (before registration)'}
    flicker_frames = round(FPS / FLICKER_HZ)

    def frames(kinds):
        for frame in range(_n_frames(len(names))):
            index = _angle_index(frame)
            hemibody = (frame // flicker_frames) % len(names)
            panels = []
            for kind in kinds:
                rgb = np.repeat(gray[kind][hemibody][index][..., None], 3, axis=-1)
                panels.append(_to_frame(rgb, [
                    (titles[kind], (1, 1, 1)),
                    (f'{master_path.name} master, {round_folder.name}', (1, 1, 1)),
                    (f'rotation about the long axis: {_rotation_angles()[index]:+.0f} deg',
                     (1, 1, 1)),
                    (f'{hemibody + 1}/{len(names)}: {names[hemibody]}', GREEN)]))
            yield _side_by_side(panels)
    for kinds, file_name in ((['registered'], 'hemibodies_rotation_registered.mp4'),
                             (['rigid'], 'hemibodies_rotation_rigid.mp4'),
                             (['rigid', 'registered'],
                              'hemibodies_rotation_rigid_vs_registered.mp4')):
        _write_video(out_folder / file_name, frames(kinds))


def video_rigid_vs_registered_slices(master_path: Path,
                                     round_folder: Path,
                                     out_folder: Path,
                                     option: str) -> None:
    """
    `hemibodies_slices_{option}_rigid_vs_registered.mp4` (see the module docstring).

    Parameters
    ----------
    master_path : Path
        Master folder (with `master_mean.tif`, which places the slices, and `segment_masks`).
    round_folder : Path
        Round whose hemibodies are shown.
    out_folder : Path
        Output folder (with `rigid_transforms.json`, made if missing).
    option : 'depth', 'per_axis' or 'long_axis'
        Slice option.

    Returns
    -------
    None
    """
    positions = slice_positions(tifffile.imread(master_path / 'master_mean.tif'), option)
    names, registered, rigid = _registered_and_rigid(master_path, round_folder, out_folder)
    # crop each slice to where any hemibody, registered or rigid, has data
    crops = [_bbox(np.any([_take(v, axis, index) != 0 for v in registered + rigid], axis=0))
             for axis, index in positions]

    def slices(volume):
        return [_take(volume, axis, index)[crop] for (axis, index), crop in zip(positions, crops)]
    grids = {'registered': [], 'rigid': []}
    for volume, rigid_volume in zip(registered, rigid):
        registered_slices, rigid_slices = slices(volume), slices(rigid_volume)
        vmax = _vmax(registered_slices)  # the same gray levels for both
        grids['registered'].append(_grid([_gray(s, vmax) for s in registered_slices]))
        grids['rigid'].append(_grid([_gray(s, vmax) for s in rigid_slices]))
    del registered, rigid
    axes = sorted({a for a, _ in positions})
    slices_text = '; '.join(f'{AXIS_NAMES[a].split(" (")[0]} '
                            f'{", ".join(str(i) for b, i in positions if b == a)}' for a in axes)
    titles = {'registered': 'registered (segment by segment) to the master',
              'rigid': 'rigid alignment only (before registration)'}
    flicker_frames = round(FPS / FLICKER_HZ)

    def frames():
        for frame in range(flicker_frames * len(names) * 2):  # every hemibody twice
            hemibody = (frame // flicker_frames) % len(names)
            yield _side_by_side([_to_frame(
                np.repeat(grids[kind][hemibody][..., None], 3, axis=-1),
                [(titles[kind], (1, 1, 1)),
                 (f'{master_path.name} master, {round_folder.name}', (1, 1, 1)),
                 (f'slices: {slices_text}', (1, 1, 1)),
                 (f'{hemibody + 1}/{len(names)}: {names[hemibody]}', GREEN)])
                for kind in ('rigid', 'registered')])
    _write_video(out_folder / f'hemibodies_slices_{option}_rigid_vs_registered.mp4', frames())


def slice_positions(mean: np.ndarray,
                    option: str) -> list:
    """
    Slices of a slice option (see the module docstring), at equal-mass quantiles of the master
    mean along their axis.

    Parameters
    ----------
    mean : np.ndarray
        Master mean.
    option : 'depth', 'per_axis' or 'long_axis'
        Slice option.

    Returns
    -------
    list of (int, int)
        (axis, index) of the 6 slices.
    """
    def quantile_positions(axis: int, quantiles: list) -> list:
        profile = np.clip(mean, 0, None).sum(axis=tuple(a for a in range(3) if a != axis))
        cumulative = np.cumsum(profile) / profile.sum()
        return [(axis, int(np.searchsorted(cumulative, q))) for q in quantiles]
    six = [(2 * i + 1) / 12 for i in range(6)]
    if option == 'depth':
        return quantile_positions(0, six)
    if option == 'long_axis':
        return quantile_positions(1, six)
    if option == 'per_axis':
        return sum((quantile_positions(axis, [1 / 3, 2 / 3]) for axis in range(3)), [])
    raise ValueError(f'Unknown slice option {option}, use one of {SLICE_OPTIONS}')


def _take(volume: np.ndarray,
          axis: int,
          index: int) -> np.ndarray:
    """
    One slice of a volume, oriented with the long axis (axis 1) vertical when it is in the
    slice.

    Parameters
    ----------
    volume : np.ndarray
        Volume.
    axis, index : int
        Slice axis and index.

    Returns
    -------
    np.ndarray
        2D slice.
    """
    image = np.take(volume, index, axis=axis)
    return image.T if axis == 2 else image  # (axis 1, axis 0) for axis-2 slices


def _grid(images: list,
          gap: int = 4) -> np.ndarray:
    """
    Put 6 2D images (or RGB images) in a 2 x 3 grid, each centred in a cell of the largest
    shape.

    Parameters
    ----------
    images : list of np.ndarray
        The images.
    gap : int
        Pixels between cells.

    Returns
    -------
    np.ndarray
        The grid image.
    """
    height = max(i.shape[0] for i in images)
    width = max(i.shape[1] for i in images)
    extra = images[0].shape[2:]
    grid = np.zeros((2 * height + gap, 3 * width + 2 * gap) + extra, np.float32)
    for n, image in enumerate(images):
        row, column = divmod(n, 3)
        y = row * (height + gap) + (height - image.shape[0]) // 2
        x = column * (width + gap) + (width - image.shape[1]) // 2
        grid[y:y + image.shape[0], x:x + image.shape[1]] = image
    return grid


def save_slice_pngs(master_path: Path,
                    out_folder: Path) -> None:
    """
    Save `master_slices_{option}.png`, the 6 master slices of each slice option, to choose one.

    Parameters
    ----------
    master_path : Path
        Master folder (with `master_mean.tif`).
    out_folder : Path
        Output folder.

    Returns
    -------
    None
    """
    mean = tifffile.imread(master_path / 'master_mean.tif')
    out_folder.mkdir(parents=True, exist_ok=True)
    for option in SLICE_OPTIONS:
        positions = slice_positions(mean, option)
        slices = [_take(mean, axis, index) for axis, index in positions]
        vmax = _vmax(slices)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10 if option != 'long_axis' else 7))
        for ax, image, (axis, index) in zip(axes.ravel(), slices, positions):
            ax.imshow(image, cmap='gray', vmin=0, vmax=vmax)
            ax.set_title(f'{AXIS_NAMES[axis]}, slice {index} of {mean.shape[axis]}', fontsize=9)
            ax.axis('off')
        fig.suptitle(f'{master_path.name} master mean: slice option "{option}"')
        fig.tight_layout()
        path = out_folder / f'master_slices_{option}.png'
        fig.savefig(path, dpi=100)
        plt.close(fig)
        print(f'Saved {path}')


def video_master_slices(master_path: Path,
                        round_folder: Path,
                        out_folder: Path,
                        option: str) -> None:
    """
    `master_hemibodies_slices_{option}_{overlay,replace}.mp4` (see the module docstring).

    Parameters
    ----------
    master_path : Path
        Master folder (with `master_mean.tif`).
    round_folder : Path
        Round whose registered hemibodies are shown.
    out_folder : Path
        Output folder.
    option : 'depth', 'per_axis' or 'long_axis'
        Slice option.

    Returns
    -------
    None
    """
    mean = tifffile.imread(master_path / 'master_mean.tif')
    positions = slice_positions(mean, option)

    crops = [_bbox(_take(mean, axis, index)) for axis, index in positions]

    def gray_grid(volume):
        slices = [_take(volume, axis, index)[crop]
                  for (axis, index), crop in zip(positions, crops)]
        vmax = _vmax(slices)
        return _grid([_gray(s, vmax) for s in slices])
    master_grid = gray_grid(mean)
    names, hemibody_grids = [], []
    for path in _hemibody_paths(round_folder):
        hemibody_grids.append([gray_grid(tifffile.imread(path))])
        names.append(_hemibody_name(path))
    axes = sorted({a for a, _ in positions})
    slices_text = '; '.join(f'{AXIS_NAMES[a].split(" (")[0]} '
                            f'{", ".join(str(i) for b, i in positions if b == a)}' for a in axes)

    def header(frame):
        return [(f'{master_path.name} master, {len(names)} hemibodies', (1, 1, 1)),
                (f'slices: {slices_text}', (1, 1, 1))]
    n_frames = round(FPS / FLICKER_HZ) * len(names) * 2  # every hemibody twice
    for mode in ('overlay', 'replace'):
        _write_video(out_folder / f'master_hemibodies_slices_{option}_{mode}.mp4',
                     _flicker_frames([master_grid], hemibody_grids, names, lambda frame: 0,
                                     mode, header, n_frames))


def make_master_videos(master_path: Path,
                       round_folder: Path,
                       reference: str,
                       videos: list,
                       slices: str = 'long_axis',
                       out_folder: Path | None = None) -> Path:
    """
    Make the videos (see the module docstring).

    Parameters
    ----------
    master_path : Path
        Master folder (`master_mean.tif`, `segment_masks`).
    round_folder : Path
        Round whose registered hemibodies are shown (relative paths are taken in `master_path`).
    reference : str
        Reference hemibody name (for `rotate_reference`).
    videos : list of str
        Which videos: 'rotate_reference', 'rotate_master', 'slices', 'rigid_vs_registered',
        'rigid_vs_registered_slices', 'rotate_muscles'.
    slices : 'depth', 'per_axis' or 'long_axis'
        Slice option of the 'slices' videos (default 'long_axis', chosen from the pngs).
    out_folder : Path or None
        Output folder. None: `{master_path}/videos/{today, YYYYMMDD}`.

    Returns
    -------
    Path
        The output folder.
    """
    master_path = Path(master_path)
    round_folder = master_path / round_folder
    out_folder = Path(out_folder or master_path / 'videos'
                      / datetime.date.today().strftime('%Y%m%d'))
    out_folder.mkdir(parents=True, exist_ok=True)
    if 'rotate_reference' in videos:
        video_reference_segments(master_path, reference, out_folder)
    if 'rotate_master' in videos:
        video_master_rotation(master_path, round_folder, out_folder)
    if 'slices' in videos:
        video_master_slices(master_path, round_folder, out_folder, slices)
    if 'rigid_vs_registered' in videos:
        video_rigid_vs_registered(master_path, round_folder, out_folder)
    if 'rigid_vs_registered_slices' in videos:
        video_rigid_vs_registered_slices(master_path, round_folder, out_folder, slices)
    if 'rotate_muscles' in videos:
        video_muscle_labels(master_path, out_folder)
    return out_folder


if __name__ == '__main__':
    matplotlib.use('Agg')
    parser = argparse.ArgumentParser(description='Videos of a master reference and the '
                                                 'hemibodies it was built from.')
    parser.add_argument('--master_path', type=Path, required=True,
                        help='Master folder (master_mean.tif, segment_masks).')
    parser.add_argument('--round_folder', type=Path, required=True,
                        help='Round whose registered hemibodies are shown (e.g. the last one).')
    parser.add_argument('--reference', default=build_master_mean.REFERENCE,
                        help='Reference hemibody (round-1 template).')
    parser.add_argument('--videos', nargs='*', default=None,
                        choices=['rotate_reference', 'rotate_master', 'slices',
                                 'rigid_vs_registered', 'rigid_vs_registered_slices',
                                 'rotate_muscles'],
                        help='Videos to make (default: rotate_reference rotate_master, or none '
                             'if --slice_pngs is given).')
    parser.add_argument('--slices', default='long_axis', choices=SLICE_OPTIONS,
                        help='Slice option of the slices videos.')
    parser.add_argument('--slice_pngs', type=Path, default=None,
                        help='Save a png of each slice option in this folder (and make no '
                             'video unless --videos is also given).')
    parser.add_argument('--out_folder', type=Path, default=None,
                        help='Output folder (default: {master_path}/videos/{YYYYMMDD}).')
    args = parser.parse_args()
    if args.slice_pngs:
        save_slice_pngs(args.master_path, args.slice_pngs)
    videos = args.videos if args.videos is not None else (
        [] if args.slice_pngs else ['rotate_reference', 'rotate_master'])
    if videos:
        make_master_videos(args.master_path, args.round_folder, args.reference, videos,
                           args.slices, args.out_folder)
