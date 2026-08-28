"""
Canonical Range Filter (SIH PS 26130).
Filters points outside radial boundaries [min_range, max_range] and removes NaNs/Infs.
"""

from typing import Tuple
import numpy as np


class RangeFilter:
    """Filters point clouds to valid sensor radial boundaries."""

    def __init__(self, min_range: float = 0.5, max_range: float = 100.0):
        self.min_range = min_range
        self.max_range = max_range
        self.min_range_sq = min_range ** 2
        self.max_range_sq = max_range ** 2

    def filter(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Filter points array (N, 4).

        Returns:
            Tuple[filtered_points, valid_mask]
        """
        if points.shape[0] == 0:
            return points, np.array([], dtype=bool)

        xyz = points[:, :3]
        finite_mask = np.isfinite(xyz[:, 0]) & np.isfinite(xyz[:, 1]) & np.isfinite(xyz[:, 2])

        r_sq = np.zeros(points.shape[0], dtype=np.float32)
        r_sq[finite_mask] = xyz[finite_mask, 0]**2 + xyz[finite_mask, 1]**2 + xyz[finite_mask, 2]**2

        valid_mask = finite_mask & (r_sq >= self.min_range_sq) & (r_sq <= self.max_range_sq)
        return points[valid_mask], valid_mask
