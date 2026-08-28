"""
Canonical Traversability Engine (SIH PS 26130).
Maps 4-class semantic predictions and elevation gradients to continuous traversability scores.
"""

from typing import Union
import numpy as np
from src.core.types import SuperClass


def compute_class_traversability(class_id: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
    """Map semantic class ID to base traversability score.

    Rules:
      - Drivable (0) -> +1.0 (Safe / Traversable)
      - Non-Drivable (1) -> -1.0 (Untraversable / Off-Road)
      - Static Obstacle (2) -> 0.0 (Blocked)
      - Dynamic Object (3) -> 0.0 (Hazard / Blocked)
      - Ignore / Unobserved (255) -> -1.0 (Unknown)
    """
    if isinstance(class_id, np.ndarray):
        trav = np.full(class_id.shape, -1.0, dtype=np.float32)
        trav[class_id == int(SuperClass.DRIVABLE_TERRAIN)] = 1.0
        trav[class_id == int(SuperClass.NON_DRIVABLE_TERRAIN)] = -1.0
        trav[class_id == int(SuperClass.STATIC_OBSTACLE)] = 0.0
        trav[class_id == int(SuperClass.DYNAMIC_OBJECT)] = 0.0
        return trav

    c = int(class_id)
    if c == int(SuperClass.DRIVABLE_TERRAIN):
        return 1.0
    elif c == int(SuperClass.NON_DRIVABLE_TERRAIN):
        return -1.0
    elif c in (int(SuperClass.STATIC_OBSTACLE), int(SuperClass.DYNAMIC_OBJECT)):
        return 0.0
    return -1.0
