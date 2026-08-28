"""
Canonical Core Data Types for Foveated LiDAR Mapping (SIH PS 26130).
Single source of truth for point definitions, hierarchical cell keys, and grid cell states.
"""

from dataclasses import dataclass, field
from enum import IntEnum
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class SuperClass(IntEnum):
    """SIH 4-Class Semantic Ontology + Ignore Label."""
    DRIVABLE_TERRAIN = 0
    NON_DRIVABLE_TERRAIN = 1
    STATIC_OBSTACLE = 2
    DYNAMIC_OBJECT = 3
    IGNORE_LABEL = 255


CLASS_NAMES: Dict[int, str] = {
    0: "drivable_terrain",
    1: "non_drivable_terrain",
    2: "static_obstacle",
    3: "dynamic_object",
    255: "ignore",
}

CLASS_HEX_COLORS: Dict[int, str] = {
    0: "#2ca02c",  # Green - Drivable
    1: "#d62728",  # Red - Non-Drivable
    2: "#1f77b4",  # Blue - Static Obstacle
    3: "#ff7f0e",  # Orange - Dynamic Object
    255: "#7f7f7f", # Gray - Ignore
}


@dataclass
class PointXYZL:
    """Canonical LiDAR point with geometry, intensity, semantic classification, and confidence."""
    x: float
    y: float
    z: float
    intensity: float = 0.0
    semantic: int = int(SuperClass.IGNORE_LABEL)
    confidence: float = 1.0


@dataclass(frozen=True)
class CellKey:
    """Hierarchical spatial hash key representing multiresolution grid cells."""
    level: int  # 0: Near (5cm), 1: Mid (15cm), 2: Far (50cm)
    ix: int
    iy: int

    def __repr__(self) -> str:
        return f"CellKey(L{self.level}:{self.ix},{self.iy})"


@dataclass
class GridCell:
    """Canonical 2.5D multiresolution grid cell holding elevation, semantics, and traversability."""
    key: CellKey
    resolution: float
    parent_id: int = -1
    elevation_mean: float = 0.0
    elevation_min: float = 0.0
    elevation_max: float = 0.0
    semantic: int = int(SuperClass.IGNORE_LABEL)
    confidence: float = 0.0
    traversability: float = -1.0
    point_count: int = 0


@dataclass
class GridMap25D:
    """Canonical 2.5D Multi-Layer Grid Map representation."""
    bounds_x: Tuple[float, float]
    bounds_y: Tuple[float, float]
    resolution: float
    grid_shape: Tuple[int, int]
    elevation_min: np.ndarray        # (H, W) float32
    elevation_max: np.ndarray        # (H, W) float32
    elevation_mean: np.ndarray       # (H, W) float32
    semantic_layer: np.ndarray       # (H, W) int64
    confidence_layer: np.ndarray     # (H, W) float32
    traversability_layer: np.ndarray # (H, W) float32
    point_count_layer: np.ndarray    # (H, W) int32


@dataclass
class ElevationCell:
    """Compact elevation and semantic cell representation."""
    elevation: float
    elevation_min: float
    elevation_max: float
    semantic: int
    traversability: float
    confidence: float
    point_count: int = 1


@dataclass
class FoveationZone:
    """Specification for a single distance-adaptive foveation tier."""
    name: str
    min_radius: float
    max_radius: float
    resolution: float
    level: int

    def contains(self, r: float) -> bool:
        return (r >= self.min_radius and r < self.max_radius)


# ============================================================
# Backward-Compatible Aliases for Legacy Phase-2 Modules
# ============================================================
@dataclass
class FoveationBand:
    name: str
    min_range: float
    max_range: float
    voxel_size: float


@dataclass
class CellState:
    points: List[Tuple[float, float, float]] = field(default_factory=list)
    intensities: List[float] = field(default_factory=list)
    classes: List[int] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)


@dataclass
class GridCell25D:
    band_name: str
    ix: int
    iy: int
    resolution: float
    point_count: int
    elevation_mean: float
    elevation_min: float
    elevation_max: float
    semantic_class: int
    confidence: float
    traversability: float


@dataclass
class PointCloudFrame:
    points: np.ndarray  # (N, 4) [x, y, z, intensity]
    frame_id: str = "frame"
    timestamp: float = 0.0
    ground_truth_labels: Optional[np.ndarray] = None


@dataclass
class FoveatedGridConfig:
    bands: List[FoveationBand] = field(default_factory=list)
    max_range: float = 100.0
    min_range: float = 0.5
