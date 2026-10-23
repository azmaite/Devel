"""
Register the muscle labels of a CT scan (traced in WebKnossos, exported by
`ct_scan/export_ct_muscles.py`) to the final master reference, leg segment by leg segment, and map
them onto the master: the first version of a master muscle map, with the identity of each muscle.

The CT holds the front and middle legs of one side (no hind leg), with every muscle traced, while
SCAPE only sees some of the coxa/thorax muscles (deeper muscles are dimmer, and often not seen).
Only the CT muscles listed in `SEGMENT_MUSCLES` are used: for the leg segments, all their muscles;
for the coxa/thorax segments, only those judged visible in SCAPE (by hand, from their overlap with
the master after a rough placement; extra, invisible CT muscles would pull the fits towards the
middle of the mass of CT muscles). The CT's thorax and coxa groups are merged, like the master's
coxa/thorax segments. Non-muscle labels and the other side's legs are ignored.

Steps (all fits with `register_segments.register_segment_shape`, shapes of the muscle masks):
1. initial similarity transform (rotation, uniform scale, translation; and whether a mirror flip
   is needed) from the centroids of the trochanter, femur and tibia segments of the CT and the
   master (the coxa/thorax segments are left out: the CT has many more thorax muscles there),
2. global affine fit of all the master leg segments to all the kept CT muscles, from 1,
3. one fit per segment, from 2: the master segment is the template and only that segment's kept
   CT muscles are the target. With the master as the template, only the CT muscle where the master
   has muscle counts: the parts of CT muscles that SCAPE does not see, and the CT muscles being
   thicker than the fluorescence masks, do not pull the fit. Affine for large segments, rigid for
   small ones (`MIN_AFFINE_VOXELS`),
4. the CT labels are brought onto the master grid with the segment transforms blended (as in
   `registration_round.warp_to_template`, nearest label): `ct_muscles_on_master.tif`,
5. the master muscle map: each voxel of a master leg segment gets the nearest kept CT muscle of
   the same segment within `MAP_DISTANCE_UM` (`master_muscle_map.tif`).
The labels that are parts of one muscle (`id_to_muscle` of the export, see
`ct_scan/ct_muscle_names.py`: `_pN` parts merged, except the accessory heart muscles) are merged
into one label in the outputs (`merge_muscle_parts`); muscles are numbered from 1 by name.

Outputs, in `out_folder` (default `master_reference/ct_alignment`), on the master grid (napari
labels, scale 1 like `master_mean.tif`):
- `ct_muscles_on_master.tif` / `.json`: the kept CT muscles, with names and the WebKnossos ids
  each muscle is made of,
- `master_muscle_map.tif` / `.json`: the master muscle map (same muscle ids),
- `ct_registration.json`: the transforms and, per segment, how much of the master segment is
  covered by its CT muscles and how much of those lies on master muscle,
- `ct_registration.png`: outlines of the muscles of the master muscle map on the master
  (`plot_muscle_map`).

Run from the terminal: `python register_ct_labels.py` (settings at the top of the script).

register_ct_labels: run all the steps.
plot_muscle_map: plot the muscle outlines of a muscle map on the master.
merge_muscle_parts: relabel WebKnossos ids into muscles (parts merged).
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk
import tifffile

import register_segments as rs
import registration_round as rr
from common.paths import MASTER_REFERENCE_PATH

CT_FOLDER = MASTER_REFERENCE_PATH / 'ct_alignment'  # `ct_labels_1p5um.tif`, see ct_scan/
VOXEL_SIZE_UM = 1.5
# CT muscles used for each master segment: WebKnossos groups (innermost group names, all their
# labels) and single labels by name
SEGMENT_MUSCLES = {
    'F_coxa/thorax': {'groups': ['RF_coxa'], 'names': ['RF_ThCo_sternal-anterior-rotator']},
    'F_trochanter': {'groups': ['RF_trochanter']},
    'F_femur': {'groups': ['RF_femur', 'FeTi_extensor']},
    'F_tibia': {'groups': ['RF_tibia', 'TiTa_LongTendon']},
    'M_coxa/thorax': {'names': ['RM_CoTr_1', 'RM_CoTr_2', 'RM_ThTr_1', 'RM_ThTr_2', 'RM_ThTr_3',
                                'RM_ThTr_4', 'RM_ThCo_3', 'RM_ThCo_4']},
    'M_trochanter': {'groups': ['RM_trochanter']},
    'M_femur': {'groups': ['RM_femur']},
    'M_tibia': {'groups': ['RM_tibia']},
}
CENTROID_SEGMENTS = ['F_trochanter', 'F_femur', 'F_tibia', 'M_trochanter', 'M_femur', 'M_tibia']
MIN_AFFINE_VOXELS = 30000
MAP_DISTANCE_UM = 10.0  # master voxels get the nearest CT muscle of their segment within this
NEAR_UM = 6.0  # tolerance of the coverage checks


def _physical(mask: np.ndarray) -> np.ndarray:
    """
    Centroid of a mask, in physical SimpleITK coordinates (x, y, z, um).

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask, array order (z, y, x).

    Returns
    -------
    np.ndarray
        Centroid (x, y, z).
    """
    return np.array(ndi.center_of_mass(mask))[::-1] * VOXEL_SIZE_UM


def _similarity(source: np.ndarray,
                target: np.ndarray) -> tuple:
    """
    Least-squares similarity transform (rotation, uniform scale, translation) mapping `source`
    points onto `target` points, with and without a mirror flip; the better one is returned.

    Parameters
    ----------
    source, target : np.ndarray
        Matching points, shape (n, 3).

    Returns
    -------
    tuple
        (`sitk.AffineTransform` source -> target, mirrored (bool), rms residual (um), scale).
    """
    s0, t0 = source - source.mean(0), target - target.mean(0)
    u, sv, vt = np.linalg.svd(s0.T @ t0)
    best = None
    for flip in (1, -1):
        d = np.eye(3)
        d[2, 2] = flip * np.sign(np.linalg.det(u @ vt))
        rotation = (u @ d @ vt).T
        scale = np.trace(np.diag(sv) @ d) / (s0 ** 2).sum()
        rms = np.sqrt((((t0 - scale * s0 @ rotation.T) ** 2).sum(1)).mean())
        if best is None or rms < best[2]:
            best = (rotation, scale, rms, bool(np.linalg.det(rotation) < 0))
    rotation, scale, rms, mirrored = best
    transform = sitk.AffineTransform(3)
    transform.SetMatrix((scale * rotation).ravel().tolist())
    transform.SetTranslation((target.mean(0) - scale * rotation @ source.mean(0)).tolist())
    return transform, mirrored, float(rms), float(scale)


def merge_muscle_parts(volumes: list,
                       id_to_muscle: dict,
                       id_to_segment: dict) -> tuple:
    """
    Relabel volumes of WebKnossos ids into muscles: the ids that are parts of one muscle get one
    label. Muscles are numbered from 1 in the order of their names.

    Parameters
    ----------
    volumes : list of np.ndarray
        Volumes of WebKnossos ids.
    id_to_muscle : dict
        WebKnossos id -> muscle name (ids without one keep their own id's name).
    id_to_segment : dict
        WebKnossos id -> master segment name.

    Returns
    -------
    tuple
        (list of relabelled uint32 volumes, dict with `id_to_name`, `id_to_segment` and
        `id_to_webknossos_ids` of the muscles).
    """
    present = sorted({int(i) for volume in volumes for i in np.unique(volume) if i})
    names = sorted({id_to_muscle.get(i, str(i)) for i in present})
    muscle_id = {name: k for k, name in enumerate(names, start=1)}
    lookup = np.zeros(max(present) + 1, dtype=np.uint32)
    parts = {k: [] for k in muscle_id.values()}
    segment_of = {}
    for i in present:
        k = muscle_id[id_to_muscle.get(i, str(i))]
        lookup[i] = k
        parts[k].append(i)
        segment_of[k] = id_to_segment[i]
    relabelled = [lookup[np.minimum(volume, max(present))] * (volume <= max(present))
                  for volume in volumes]
    return relabelled, {'id_to_name': {k: name for name, k in muscle_id.items()},
                        'id_to_segment': segment_of, 'id_to_webknossos_ids': parts}


def plot_muscle_map(muscle_map: np.ndarray,
                    mean: np.ndarray,
                    segments: np.ndarray,
                    path: Path) -> None:
    """
    Plot the outline of each muscle of a muscle map (max projections along the three axes) on the
    master image, one colour per muscle, with the master segments as faint white outlines.

    Parameters
    ----------
    muscle_map : np.ndarray
        Muscle labels on the master grid.
    mean : np.ndarray
        Master image.
    segments : np.ndarray
        Master segment map.
    path : Path
        Output png.

    Returns
    -------
    None
    """
    ids = [int(i) for i in np.unique(muscle_map) if i]
    colours = plt.cm.tab20(np.arange(len(ids)) % 20)
    fig, axes = plt.subplots(1, 3, figsize=(24, 10))
    for ax, axis in zip(axes, range(3)):
        ax.imshow(mean.max(axis=axis), cmap='gray', vmin=0, vmax=np.percentile(mean, 99.9),
                  aspect='auto')
        ax.contour((segments > 0).max(axis=axis), [0.5], colors='w', linewidths=0.4,
                   alpha=0.6)
        for muscle, colour in zip(ids, colours):
            ax.contour((muscle_map == muscle).max(axis=axis), [0.5], colors=[colour],
                       linewidths=0.9)
        ax.set_title(f'master muscle map: muscle outlines (max along axis {axis}); '
                     f'white: master segments', fontsize=9)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def register_ct_labels(ct_folder: Path = CT_FOLDER,
                       master_path: Path = MASTER_REFERENCE_PATH,
                       out_folder: Path = CT_FOLDER,
                       from_report: bool = False) -> None:
    """
    Register the CT muscle labels to the final master and map them onto it (see the module
    docstring).

    Parameters
    ----------
    ct_folder : Path
        Folder with `ct_labels_1p5um.tif` and `.json` (`ct_scan/export_ct_muscles.py`).
    master_path : Path
        Folder with the final master (`master_segments.tif`, `.json`).
    out_folder : Path
        Output folder.
    from_report : bool
        If True, skip the fits (steps 1-3) and use the transforms saved in
        `out_folder / 'ct_registration.json'` (e.g. after a change of the blending).

    Returns
    -------
    None
    """
    out_folder.mkdir(parents=True, exist_ok=True)
    master = tifffile.imread(master_path / 'master_segments.tif').astype(np.uint32)
    master_names = json.loads((master_path / 'master_segments.json').read_text())['id_to_name']
    master_index = {name: int(i) for i, name in master_names.items()}
    ct = tifffile.imread(ct_folder / 'ct_labels_1p5um.tif').astype(np.uint32)
    ct_info = json.loads((ct_folder / 'ct_labels_1p5um.json').read_text())
    id_to_name = {int(k): v for k, v in ct_info['id_to_name'].items()}
    id_to_group = {int(k): v for k, v in ct_info['id_to_group'].items()}
    by_name = {v: k for k, v in id_to_name.items()}

    segments = list(SEGMENT_MUSCLES)
    segment_ids = {}
    for segment, spec in SEGMENT_MUSCLES.items():
        ids = [i for i, g in id_to_group.items() if g in spec.get('groups', [])]
        ids += [by_name[n] for n in spec.get('names', [])]
        segment_ids[segment] = sorted(set(ids))
    kept = sorted({i for ids in segment_ids.values() for i in ids})
    print('CT muscles per segment:', {s: len(v) for s, v in segment_ids.items()})

    shape = tuple(int(s) for s in np.maximum(master.shape, ct.shape))
    master_p, ct_p = rr._pad(master, shape), rr._pad(ct, shape)
    ct_segment = {s: np.isin(ct_p, segment_ids[s]) for s in segments}
    master_segment = {s: master_p == master_index[s] for s in segments}

    if from_report:
        report = json.loads((out_folder / 'ct_registration.json').read_text())
        global_fit = rr._affine_from_dict(report['global'])
        fits = [rr._affine_from_dict(report['segments'][s]) for s in segments]
        print('Using the transforms of', out_folder / 'ct_registration.json')
    else:
        report, global_fit, fits = _fit_ct(master_segment, ct_segment, ct_p, kept, segments,
                                           shape)
    _map_ct(report, global_fit, fits, master, master_p, master_segment, master_index, ct_p,
            ct_info, kept, segments, segment_ids, shape, master_path, out_folder)


def _fit_ct(master_segment: dict,
            ct_segment: dict,
            ct_p: np.ndarray,
            kept: list,
            segments: list,
            shape: tuple) -> tuple:
    """
    Steps 1-3 of the module docstring: similarity from the segment centroids, global affine fit,
    and one fit per segment (CT -> master physical coordinates).

    Parameters
    ----------
    master_segment, ct_segment : dict
        Segment name -> boolean mask (common padded grid) of the master and of the kept CT
        muscles.
    ct_p : np.ndarray
        CT labels on the common grid.
    kept : list of int
        Kept CT ids.
    segments : list of str
        Segment names.
    shape : tuple of int
        Common grid shape.

    Returns
    -------
    tuple
        (report dict, global fit, list of segment fits).
    """
    # 1. similarity from segment centroids (CT -> master)
    similarity, mirrored, rms, scale = _similarity(
        np.array([_physical(ct_segment[s]) for s in CENTROID_SEGMENTS]),
        np.array([_physical(master_segment[s]) for s in CENTROID_SEGMENTS]))
    print(f'Centroid similarity: {"mirror flip, " if mirrored else ""}scale {scale:.3f}, '
          f'rms {rms:.1f} um')
    # 2. global affine fit
    all_master = np.any([master_segment[s] for s in segments], axis=0)
    all_ct = np.isin(ct_p, kept)
    global_fit = rs.register_segment_shape(all_master, all_ct, model='affine',
                                           initial=similarity, clip_um=45.0)
    print('Global fit done')
    # 3. segment fits
    near_master = ndi.distance_transform_edt(~all_master, sampling=VOXEL_SIZE_UM) <= NEAR_UM
    report = {'similarity': {'mirrored': mirrored, 'scale': scale, 'centroid_rms_um': rms},
              'global': rr._affine_dict(global_fit), 'segments': {}}
    fits = []
    for segment in segments:
        model = 'affine' if master_segment[segment].sum() >= MIN_AFFINE_VOXELS else 'rigid'
        fit = rs.register_segment_shape(master_segment[segment], ct_segment[segment], model=model,
                                        initial=global_fit)
        centre = tuple(_physical(master_segment[segment]))
        shift = float(np.linalg.norm(np.subtract(rs.invert_affine(fit).TransformPoint(centre),
                                                 rs.invert_affine(global_fit).TransformPoint(centre))))
        (placed,) = rr._warp_blended([ct_segment[segment].astype(np.float32)],
                                     [rs.invert_affine(fit)], [1.0], shape)
        placed = placed > 0.5
        near_ct = ndi.distance_transform_edt(~placed, sampling=VOXEL_SIZE_UM) <= NEAR_UM
        covered = float((master_segment[segment] & near_ct).sum() / master_segment[segment].sum())
        on_muscle = float((placed & near_master).sum() / max(placed.sum(), 1))
        report['segments'][segment] = {'model': model, 'shift_from_global_um': shift,
                                       'master_covered_by_ct': covered,
                                       'ct_near_master_muscle': on_muscle,
                                       **rr._affine_dict(fit)}
        fits.append(fit)
        print(f'  {segment}: {model}, {shift:.1f} um from global; master covered by CT '
              f'{covered:.2f}, CT near master muscle {on_muscle:.2f}')
    return report, global_fit, fits


def _map_ct(report: dict,
            global_fit: sitk.Transform,
            fits: list,
            master: np.ndarray,
            master_p: np.ndarray,
            master_segment: dict,
            master_index: dict,
            ct_p: np.ndarray,
            ct_info: dict,
            kept: list,
            segments: list,
            segment_ids: dict,
            shape: tuple,
            master_path: Path,
            out_folder: Path) -> None:
    """
    Steps 4-5 of the module docstring: bring the kept CT muscles onto the master grid with the
    segment transforms blended, build the master muscle map, and save the outputs.

    Parameters
    ----------
    report : dict
        Fit report (saved as `ct_registration.json`).
    global_fit, fits
        Global fit and segment fits (CT -> master).
    master, master_p : np.ndarray
        Master segment map, and the same on the common grid.
    master_segment : dict
        Segment name -> master segment mask (common grid).
    master_index : dict
        Segment name -> master segment id.
    ct_p : np.ndarray
        CT labels on the common grid.
    ct_info : dict
        Contents of `ct_labels_1p5um.json`.
    kept : list of int
        Kept CT ids.
    segments : list of str
        Segment names.
    segment_ids : dict
        Segment name -> its kept CT ids.
    shape : tuple of int
        Common grid shape.
    master_path, out_folder : Path
        Folder of the master, and output folder.

    Returns
    -------
    None
    """
    # 4. CT labels on the master grid, segment transforms blended (only the kept muscles)
    weights_segments = np.zeros(shape, np.uint32)
    for k, segment in enumerate(segments, start=1):
        weights_segments[master_segment[segment]] = k
    weights, _, global_weight = rr.segment_weights(weights_segments, len(segments))
    (on_master,) = rr._warp_blended([np.where(np.isin(ct_p, kept), ct_p, 0).astype(np.uint32)],
                                    [rs.invert_affine(f) for f in fits + [global_fit]],
                                    weights + [global_weight], shape, order=0)
    crop = tuple(slice(0, s) for s in master.shape)
    on_master = on_master[crop]
    # 5. master muscle map: nearest CT muscle of the same segment
    muscle_map = np.zeros(master.shape, np.uint32)
    for segment in segments:
        in_segment = np.isin(on_master, segment_ids[segment])
        distance, indices = ndi.distance_transform_edt(~in_segment, sampling=VOXEL_SIZE_UM,
                                                       return_indices=True)
        nearest = on_master[tuple(indices)]
        voxels = (master == master_index[segment]) & (distance <= MAP_DISTANCE_UM)
        muscle_map[voxels] = nearest[voxels]
    id_to_muscle = {int(k): v for k, v in ct_info.get('id_to_muscle', {}).items()}
    id_to_segment = {i: s for s in segments for i in segment_ids[s]}
    (on_master, muscle_map), muscles = merge_muscle_parts([on_master, muscle_map], id_to_muscle,
                                                          id_to_segment)
    for name, volume in (('ct_muscles_on_master', on_master), ('master_muscle_map', muscle_map)):
        tifffile.imwrite(out_folder / f'{name}.tif', volume.astype(np.uint32), compression='zlib')
        counts = np.bincount(volume.ravel(), minlength=len(muscles['id_to_name']) + 1)
        (out_folder / f'{name}.json').write_text(json.dumps(
            {**muscles, 'voxels': {k: int(counts[k]) for k in muscles['id_to_name']},
             'grid': 'master reference grid (1.5 um, as master_mean.tif)'}, indent=1))
    coverage = (muscle_map > 0).sum() / max((weights_segments[crop] > 0).sum(), 1)
    report['master_leg_segments_mapped'] = float(coverage)
    (out_folder / 'ct_registration.json').write_text(json.dumps(report, indent=1))
    print(f'Master muscle map: {len(np.unique(muscle_map)) - 1} muscles, covering {coverage:.2f} '
          f'of the master front/mid leg segments')

    plot_muscle_map(muscle_map, tifffile.imread(master_path / 'master_mean.tif'), master,
                    out_folder / 'ct_registration.png')


if __name__ == '__main__':
    matplotlib.use('Agg')
    parser = argparse.ArgumentParser(description='Register CT muscle labels to the master.')
    parser.add_argument('--ct_folder', type=Path, default=CT_FOLDER)
    parser.add_argument('--master_path', type=Path, default=MASTER_REFERENCE_PATH)
    parser.add_argument('--out_folder', type=Path, default=CT_FOLDER)
    parser.add_argument('--from_report', action='store_true',
                        help='Skip the fits and use the transforms of ct_registration.json.')
    args = parser.parse_args()
    register_ct_labels(args.ct_folder, args.master_path, args.out_folder, args.from_report)
