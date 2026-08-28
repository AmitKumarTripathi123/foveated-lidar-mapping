"""
Canonical LiDAR Point Cloud Loader (SIH PS 26130).
Loads binary (.bin) SemanticPOSS / Velodyne scans and returns sanitized (N, 4) float32 arrays.
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np


def load_lidar_points(file_path: Union[str, Path]) -> np.ndarray:
    """Load raw LiDAR points from .bin file.

    Args:
        file_path: Path to .bin file containing float32 points [x, y, z, intensity].

    Returns:
        np.ndarray of shape (N, 4) with dtype float32.
    """
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"LiDAR file not found: {p}")

    raw_data = np.fromfile(str(p), dtype=np.float32)
    if raw_data.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    if raw_data.size % 4 != 0:
        # If not 4 channels, assume [x, y, z] and append 0.0 intensity
        if raw_data.size % 3 == 0:
            pts = raw_data.reshape(-1, 3)
            intensity = np.zeros((pts.shape[0], 1), dtype=np.float32)
            return np.hstack([pts, intensity])
        else:
            raise ValueError(f"Corrupt LiDAR point cloud file: {p} (size={raw_data.size})")

    return raw_data.reshape(-1, 4)
