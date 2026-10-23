"""
Muscle names of the CT-scan labels traced in WebKnossos: fixes for the known typos of the raw
label names, and the rules that merge the parts of a muscle (`_p1`, `_p2`, ...) into one muscle.

The labels of the export `RF-RM Leg muscles in pupa` were traced in parts for some muscles (e.g.
`RF_FeTi_extensor_p1` ... `_p8`, `RF_TiTa_long-tendon_p1` ... `_p9`): these parts are one muscle
each, except the accessory heart muscles of the tibia (`TiTa_acc-heart-muscle`), whose parts are
kept separate (p1 and p2), as they are seen as separate in the SCAPE recordings.

ID_TO_NAME_FIXES: raw name fixes of that export, by WebKnossos id.
KEEP_PARTS_SEPARATE: muscles whose parts are not merged.
muscle_name: merged muscle name of one label.
muscle_names: merged muscle name of every label of an export.
"""

import re

# known typos/duplicates in the raw webknossos names of the export 'RF-RM Leg muscles in pupa'
# (ids of that export):
# - id 31 is missing its 'RF_' prefix and uses another style than its siblings (ids 27-30, all
#   'RF_TiTa_long-tendon_pN')
# - ids 24/25 are BOTH named 'RF_FeTi-medial-acc-flexor_p2' (hyphen typo + duplicate part
#   number): 24 becomes _p1 so they form a proper p1/p2 pair
# - id 78 'RM_FeTi-lateral-acc-flexor' has the same hyphen typo
# - id 73 'RM_FeTi_long_tendon' uses an underscore instead of the hyphen used everywhere else
#   ('long-tendon')
# - id 50 'RF_ThCo_sternal-anterior-rotator' is the 'ThCo_rot' of the SCAPE (NMF) analysis
# - id 80 'RM_TiTa_acc-heart-muscle_p3' is renamed to _p2, joining it with id 79 into one part
#   (only 2 parts of this muscle are seen in SCAPE, not 3)
ID_TO_NAME_FIXES = {
    31: 'RF_TiTa_long-tendon_p3',
    24: 'RF_FeTi_medial-acc-flexor_p1',
    25: 'RF_FeTi_medial-acc-flexor_p2',
    78: 'RM_FeTi_lateral-acc-flexor',
    73: 'RM_FeTi_long-tendon',
    50: 'RF_ThCo_rot',
    80: 'RM_TiTa_acc-heart-muscle_p2',
}

# muscles (names without the leg prefix and the part suffix) whose _pN parts stay separate;
# the parts of every other muscle are merged into one
KEEP_PARTS_SEPARATE = {'TiTa_acc-heart-muscle'}


def muscle_name(label_id: int,
                raw_name: str) -> str:
    """
    Merged muscle name of one label: the raw name with its fix (`ID_TO_NAME_FIXES`), without the
    annotator's '?' marks, and without its `_pN` part suffix (unless the muscle is in
    `KEEP_PARTS_SEPARATE`).

    Parameters
    ----------
    label_id : int
        WebKnossos id of the label.
    raw_name : str
        Its raw WebKnossos name.

    Returns
    -------
    str
        Muscle name, e.g. 'RF_FeTi_extensor' or 'RM_TiTa_acc-heart-muscle_p2'.
    """
    name = ID_TO_NAME_FIXES.get(label_id, raw_name).replace('?', '')
    base = re.sub(r'_p\d+$', '', name)
    leg_and_muscle = base.split('_', 1)
    if len(leg_and_muscle) == 2 and leg_and_muscle[1] in KEEP_PARTS_SEPARATE:
        return name
    return base


def muscle_names(id_to_name: dict) -> dict:
    """
    Merged muscle name of every label (see `muscle_name`).

    Parameters
    ----------
    id_to_name : dict
        WebKnossos id -> raw name.

    Returns
    -------
    dict
        WebKnossos id -> muscle name (labels of one muscle share the same name).
    """
    return {int(i): muscle_name(int(i), name) for i, name in id_to_name.items()}
