"""
Canonical Post-Processing & Validation Module (SIH PS 26130).
"""

from typing import Tuple
import numpy as np


def validate_predictions(
    xyz: np.ndarray,
    predicted_class: np.ndarray,
    confidence: np.ndarray,
) -> bool:
    """Validate prediction data integrity in O(N) linear time."""
    if xyz.shape[0] != predicted_class.shape[0] or xyz.shape[0] != confidence.shape[0]:
        return False
    if xyz.shape[0] == 0:
        return True

    # Check finite coordinates
    if not np.all(np.isfinite(xyz)):
        return False

    # Check class bounds {0, 1, 2, 3, 255}
    min_c = int(np.min(predicted_class))
    max_c = int(np.max(predicted_class))
    if min_c < 0 or (max_c > 3 and max_c != 255):
        return False

    # Check confidence bounds [0.0, 1.0]
    min_conf = float(np.min(confidence))
    max_conf = float(np.max(confidence))
    if min_conf < -1e-5 or max_conf > 1.0 + 1e-5:
        return False

    return True
