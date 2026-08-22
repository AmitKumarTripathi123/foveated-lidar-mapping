"""
class_map.py
============
SemanticPOSS raw label -> project super-class remapping.

Raw class IDs follow SemanticPOSS's official label definitions (same
32-bit encoding as SemanticKITTI: the lower 16 bits of each label store
the semantic class ID).

Project super-classes (what the segmentation model will actually predict):
    0 = drivable terrain
    1 = non_drivable terrain      (currently unused by POSS's "ground"
                                    class, which doesn't distinguish
                                    road vs. sidewalk/curb -- see README)
    2 = static obstacle
    3 = dynamic object
    255 = ignore (excluded from loss / evaluation)
"""

import numpy as np

IGNORE_LABEL = 255

PROJECT_CLASSES = {
    0: "drivable_terrain",
    1: "non_drivable_terrain",
    2: "static_obstacle",
    3: "dynamic_object",
}

# Human-readable names for raw POSS label IDs, from the dataset's own
# documentation and empirical raw dataset scan.
POSS_RAW_CLASSES = {
    0:  "unlabeled",
    1:  "unlabeled_outlier",
    4:  "people",
    5:  "people",
    6:  "rider",
    7:  "car",
    8:  "trunk",
    9:  "plants",
    10: "traffic sign 1",
    11: "traffic sign 2",
    12: "traffic sign 3",
    13: "pole",
    14: "trashcan",
    15: "building",
    16: "cone/stone",
    17: "fence",
    18: "traffic sign 4",
    19: "other static",
    20: "unknown_20",
    21: "bike",
    22: "ground",
}

# Frozen Phase-2 SemanticPOSS raw label ID -> project super-class ID mapping:
# 21 -> 0 (drivable_terrain)
# 20 -> 1 (non_drivable_terrain)
# 19 -> 1 (non_drivable_terrain)
# 22 -> 255 (IGNORE_LABEL)
POSS_CLASS_REMAP = {
    0:  IGNORE_LABEL,  # unlabeled -> ignore
    1:  IGNORE_LABEL,  # outlier/unlabeled -> ignore
    22: IGNORE_LABEL,  # ground -> ignore (as per Phase-2 specification)

    21: 0,             # 21 -> 0 (drivable terrain)
    20: 1,             # 20 -> 1 (non-drivable terrain)
    19: 1,             # 19 -> 1 (non-drivable terrain)

    4:  3,  # people -> dynamic
    5:  3,  # 2+ people -> dynamic
    6:  3,  # rider -> dynamic
    7:  3,  # car -> dynamic

    8:  2,  # trunk -> static obstacle
    9:  2,  # plants -> static obstacle
    10: 2,  # traffic sign 1 -> static obstacle
    11: 2,  # traffic sign 2 -> static obstacle
    12: 2,  # traffic sign 3 -> static obstacle
    13: 2,  # pole -> static obstacle
    14: 2,  # trashcan -> static obstacle
    15: 2,  # building -> static obstacle
    16: 2,  # cone/stone -> static obstacle
    17: 2,  # fence -> static obstacle
    18: 2,  # traffic sign 4 -> static obstacle
}



def remap_labels(labels_raw: np.ndarray) -> np.ndarray:
    """
    Vectorized remap of raw POSS labels -> project super-classes.
    Any raw ID not present in POSS_CLASS_REMAP falls back to IGNORE_LABEL,
    so unexpected/unmapped classes never silently corrupt training.
    """
    remapped = np.full_like(labels_raw, IGNORE_LABEL, dtype=np.int64)
    for raw_id, proj_id in POSS_CLASS_REMAP.items():
        remapped[labels_raw == raw_id] = proj_id
    return remapped


def get_class_colors() -> dict:
    """Return RGB color mapping (0-255) for visualization of project super-classes."""
    return {
        0: [70, 130, 180],    # drivable_terrain: Steel Blue
        1: [218, 165, 32],    # non_drivable_terrain: Goldenrod
        2: [178, 34, 34],     # static_obstacle: Firebrick Red
        3: [50, 205, 50],     # dynamic_object: Lime Green
        255: [0, 0, 0],       # IGNORE: Black
    }


def compute_class_weights(class_counts: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Compute inverse-frequency class weights for active project super-classes.
    Only classes present in the dataset (count > 0) receive inverse-frequency weighting;
    unused classes receive weight 0.0.
    """
    num_classes = len(PROJECT_CLASSES)
    counts = class_counts[:num_classes].astype(np.float64)
    active_mask = counts > 0
    weights = np.zeros(num_classes, dtype=np.float64)
    
    if np.any(active_mask):
        total_active_points = counts[active_mask].sum()
        num_active = active_mask.sum()
        raw_weights = total_active_points / (num_active * counts[active_mask])
        weights[active_mask] = raw_weights / raw_weights.mean()

    return weights


