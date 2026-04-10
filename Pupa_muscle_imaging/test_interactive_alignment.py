import os
import pickle

from NMF import utils
from interactive_alignment import interactive_alignment

# load traced parts coordinates
main_path = '/mnt/labserver/data/MA/Development_project/'
traces_folder = os.path.join(main_path, 'CT_PupaP15_traced')
traces_file = os.path.join(traces_folder, 'pupa_traced_coordinates_rot.pkl')
with open(traces_file, 'rb') as f:
    coordinates = pickle.load(f)

# load min_projection image
#mkv_list = utils.get_mkv_list()
#mkv_file = mkv_list[30]
mkv_file = '/mnt/labserver/data/MA/Development_project/Pupa_muscle_long_recordings/20260220_V7_prepup_02-18_11h45/20260221_213031/20260220_152804_V7_prepup_2-18_11h45_fly_1_V7_prepup_2-18_11h45_20260221_213031_raw_tiff.mkv'
min_proj, _ = utils.load_min_max_proj(mkv_file)

# align
final_params = interactive_alignment(min_proj, coordinates)
print(final_params)