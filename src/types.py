"""
Core data types and interfaces for the Foveated 3D LiDAR pipeline.
Strictly adheres to the project interface contract.
"""

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Tuple, Any
import math
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
    semantic_counts: Dict[int, int] = field(default_factory=dict)

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

    @property
    def height_range(self) -> float:
        """Calculates vertical geometric span: height_range = elevation_max - elevation_min."""
        if math.isnan(self.elevation_max) or math.isnan(self.elevation_min):
            return float("nan")
        return float(self.elevation_max - self.elevation_min)

    @property
    def valid_semantic_count(self) -> int:
        """Returns count of valid semantic points (excluding IGNORE_LABEL)."""
        if not self.semantic_counts:
            return self.point_count if self.point_count > 0 else 0
        valid = sum(cnt for cid, cnt in self.semantic_counts.items() if cid != SuperClass.IGNORE_LABEL)
        return valid if valid > 0 else self.semantic_counts.get(SuperClass.IGNORE_LABEL, 0)

    @property
    def dominant_class(self) -> Optional[int]:
        """Returns the dominant class based on semantic counts with priority tie-breaking."""
        if not self.semantic_counts:
            return self.semantic_class if self.semantic_class != SuperClass.IGNORE_LABEL else None
        
        p_weights = {
            SuperClass.DYNAMIC_OBJECT: 4,
            SuperClass.STATIC_OBSTACLE: 3,
            SuperClass.NON_DRIVABLE_TERRAIN: 2,
            SuperClass.DRIVABLE_TERRAIN: 1,
            SuperClass.IGNORE_LABEL: 0,
        }
        best_c = None
        max_c = 0
        best_p = -1
        for cid, cnt in self.semantic_counts.items():
            if cid == SuperClass.IGNORE_LABEL:
                continue
            if cnt <= 0:
                continue
            p = p_weights.get(cid, 0)
            if cnt > max_c or (cnt == max_c and p > best_p):
                max_c = cnt
                best_c = cid
                best_p = p
        if best_c is not None:
            return best_c
        if self.semantic_counts.get(SuperClass.IGNORE_LABEL, 0) > 0:
            return SuperClass.IGNORE_LABEL
        return self.semantic_class if self.semantic_class != SuperClass.IGNORE_LABEL else None


    def class_probability(self, class_id: int) -> float:
        """Calculates class probability: semantic_counts[c] / valid_semantic_count."""
        total_valid = self.valid_semantic_count
        if total_valid <= 0:
            if class_id == SuperClass.IGNORE_LABEL and self.semantic_counts.get(SuperClass.IGNORE_LABEL, 0) > 0:
                return 1.0
            return 0.0
        return float(self.semantic_counts.get(class_id, 0)) / float(total_valid)

    @property
    def semantic_confidence(self) -> float:
        """Returns mean confidence associated with the cell."""
        return float(self.confidence)

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
