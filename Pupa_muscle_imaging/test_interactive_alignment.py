#!/usr/bin/env python3
import pickle
from pathlib import Path

import utils
from interactive_alignment import interactive_alignment

# load traced parts coordinates
main_path = Path('/mnt/labserver/data/MA/Development_project/')
traces_file = main_path / 'CT_PupaP15_traced' / 'pupa_traced_coordinates_rot.pkl'
with open(traces_file, 'rb') as f:
    coordinates = pickle.load(f)

# load min_projection image
#mkv_list = utils.get_mkv_list()
#mkv_file = mkv_list[30]
#mkv_file = '/mnt/labserver/data/MA/Development_project/Pupa_muscle_long_recordings/20260220_V7_prepup_02-18_11h45/20260221_213031/20260220_152804_V7_prepup_2-18_11h45_fly_1_V7_prepup_2-18_11h45_20260221_213031_raw_tiff.mkv'
#img, _ = utils.load_min_max_proj(mkv_file)

# load scape median volume
scape_path = Path('/mnt/labserver/data/MA/SCAPE')
img_path = scape_path / '260422_V-7_MHC-GCaMP8m_SCAPE' / 'sample1_run12' / 'figures' / 'median_volume_flat_img.pkl'
with open(img_path, 'rb') as f:
    (img, aspect_ratio, img2) = pickle.load(f)

# align
final_params = interactive_alignment(img2, coordinates)#, aspect_ratio=aspect_ratio)
print(final_params)
