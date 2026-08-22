"""
Core data types and interfaces for the Foveated 3D LiDAR pipeline.
Strictly adheres to the project interface contract.
"""

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class SuperClass(IntEnum):
    """Standardized Super-classes for autonomous navigation."""
    DRIVABLE_TERRAIN = 0
    NON_DRIVABLE_TERRAIN = 1
    STATIC_OBSTACLE = 2
    DYNAMIC_OBJECT = 3
    IGNORE_LABEL = 255

    @classmethod
    def get_name(cls, value: int) -> str:
        try:
            return cls(value).name.lower()
        except ValueError:
            return f"unknown_{value}"


class AggregationPolicy(str, Enum):
    """Voxel aggregation policy."""
    NEAREST = "nearest"
    CENTROID = "centroid"
    MAJORITY = "majority"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    OBSTACLE_PRESERVING = "obstacle_preserving"


class ValidationPolicy(str, Enum):
    """Action policy when an invalid frame is detected."""
    STRICT_STOP = "strict_stop"
    SKIP_AND_WARN = "skip_and_warn"
    ISOLATE = "isolate"


@dataclass
class ClassifiedPoint:
    """
    Matches the C++ struct contract:
    struct ClassifiedPoint {
        float x;
        float y;
        float z;
        float intensity;
        uint8_t class_id;
        float confidence;
    };
    """
    x: float
    y: float
    z: float
    intensity: float
    class_id: int = SuperClass.IGNORE_LABEL
    confidence: float = 1.0


@dataclass
class FoveationBand:
    """A distance band configuration for foveated voxelization."""
    name: str
    min_range: float
    max_range: float
    voxel_size: float

    def contains(self, r: float) -> bool:
        """Check if radial distance falls within [min_range, max_range)."""
        return self.min_range <= r < self.max_range or (self.max_range >= 100.0 and r == self.max_range)


@dataclass
class PointCloudFrame:
    """
    LiDAR point cloud frame.
    points: float32[N, 4] -> [x, y, z, intensity]
    labels: uint32[N] -> raw or mapped labels
    """
    points: np.ndarray  # shape: (N, 4), dtype: float32
    labels: np.ndarray  # shape: (N,), dtype: uint32 or int32
    confidences: Optional[np.ndarray] = None  # shape: (N,), dtype: float32
    frame_id: str = "000000"
    timestamp: float = 0.0
    sequence_id: str = "00"
    is_valid: bool = True
    validation_notes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.points is not None and not isinstance(self.points, np.ndarray):
            self.points = np.asarray(self.points, dtype=np.float32)
        if self.labels is not None and not isinstance(self.labels, np.ndarray):
            self.labels = np.asarray(self.labels, dtype=np.uint32)
        if self.confidences is None and self.points is not None:
            self.confidences = np.ones(len(self.points), dtype=np.float32)

    @property
    def num_points(self) -> int:
        return len(self.points) if self.points is not None else 0

    @property
    def xyz(self) -> np.ndarray:
        return self.points[:, :3] if self.points is not None and len(self.points) > 0 else np.empty((0, 3), dtype=np.float32)

    @property
    def intensity(self) -> np.ndarray:
        return self.points[:, 3] if self.points is not None and len(self.points) > 0 else np.empty((0,), dtype=np.float32)

    @property
    def ranges_2d(self) -> np.ndarray:
        """Horizontal radial distance: r = sqrt(x^2 + y^2)"""
        if self.points is None or len(self.points) == 0:
            return np.empty((0,), dtype=np.float32)
        x = self.points[:, 0]
        y = self.points[:, 1]
        return np.sqrt(x * x + y * y)

    @property
    def ranges_3d(self) -> np.ndarray:
        """Spherical distance: r_3d = sqrt(x^2 + y^2 + z^2)"""
        if self.points is None or len(self.points) == 0:
            return np.empty((0,), dtype=np.float32)
        return np.linalg.norm(self.points[:, :3], axis=1)

    def copy(self) -> 'PointCloudFrame':
        return PointCloudFrame(
            points=self.points.copy() if self.points is not None else None,
            labels=self.labels.copy() if self.labels is not None else None,
            confidences=self.confidences.copy() if self.confidences is not None else None,
            frame_id=self.frame_id,
            timestamp=self.timestamp,
            sequence_id=self.sequence_id,
            is_valid=self.is_valid,
            validation_notes=list(self.validation_notes),
            metadata=dict(self.metadata)
        )

import ctypes


class CClassifiedPoint(ctypes.Structure):
    """C-compatible struct matching ClassifiedPoint."""
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
        ("intensity", ctypes.c_float),
        ("class_id", ctypes.c_uint8),
        ("confidence", ctypes.c_float)
    ]


class CClassifiedPointPacked(ctypes.Structure):
    """Packed C-compatible struct matching #pragma pack(push, 1)."""
    _pack_ = 1
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
        ("intensity", ctypes.c_float),
        ("class_id", ctypes.c_uint8),
        ("confidence", ctypes.c_float)
    ]

# ==============================================================================
# Phase-2 2.5D Foveated Grid Data Types
# ==============================================================================

class CellState(IntEnum):
    """Occupancy and observation state of a 2.5D grid cell."""
    UNKNOWN = 0   # Unobserved cell (no LiDAR points received)
    OCCUPIED = 1  # Observed cell containing LiDAR point(s)
    FREE = 2      # Proven free space via ray-tracing/sensor clearance


@dataclass
class GridCell25D:
    """
    Standardized 2.5D Foveated Spatial Cell.
    Spatial identity is strictly 2D (ix, iy). Elevation Z is a cell attribute.
    """
    ix: int
    iy: int
    resolution: float
    point_count: int = 0
    elevation_mean: float = float("nan")
    elevation_min: float = float("nan")
    elevation_max: float = float("nan")
    semantic_class: int = SuperClass.IGNORE_LABEL
    confidence: float = 0.0
    traversability: float = 0.0
    state: CellState = CellState.UNKNOWN
    band_name: str = ""

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Returns (min_x, max_x, min_y, max_y) in physical meters."""
        min_x = self.ix * self.resolution
        max_x = (self.ix + 1) * self.resolution
        min_y = self.iy * self.resolution
        max_y = (self.iy + 1) * self.resolution
        return (min_x, max_x, min_y, max_y)

    @property
    def center_xy(self) -> Tuple[float, float]:
        """Returns cell center (x, y) in physical meters."""
        return ((self.ix + 0.5) * self.resolution, (self.iy + 0.5) * self.resolution)

    def contains_point(self, x: float, y: float) -> bool:
        """Verifies spatial invariant: ix*s <= x < (ix+1)*s and iy*s <= y < (iy+1)*s."""
        min_x, max_x, min_y, max_y = self.bounds
        return (min_x <= x < max_x) and (min_y <= y < max_y)


@dataclass
class FoveatedGridConfig:
    """
    Phase-2 Frozen 4-Band Distance-Aware Spatial Grid Specification:
      0–10 m   -> 0.05 m  (5 cm)
      10–30 m  -> 0.10 m  (10 cm)
      30–60 m  -> 0.25 m  (25 cm)
      60–100 m -> 0.50 m  (50 cm)
      >= 100 m -> Out of range
    """
    name: str = "phase2_frozen_4band"
    max_range: float = 100.0
    bands: List[FoveationBand] = field(default_factory=lambda: [
        FoveationBand(name="near_field", min_range=0.0, max_range=10.0, voxel_size=0.05),
        FoveationBand(name="mid_near_field", min_range=10.0, max_range=30.0, voxel_size=0.10),
        FoveationBand(name="mid_far_field", min_range=30.0, max_range=60.0, voxel_size=0.25),
        FoveationBand(name="far_field", min_range=60.0, max_range=100.0, voxel_size=0.50),
    ])
