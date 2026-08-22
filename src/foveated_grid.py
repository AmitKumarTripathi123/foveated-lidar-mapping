"""
Phase-2 Foveated 2.5D Spatial Grid Correctness Engine.
Establishes the exact, deterministic mathematical mapping:
  LiDAR Point (x, y, z, intensity)
      ↓
  Horizontal Distance r = sqrt(x^2 + y^2)
      ↓
  Distance Band Resolution s(r)
      ↓
  2D XY Cell (ix = floor(x/s), iy = floor(y/s))
      ↓
  Elevation Aggregation (mean, min, max) + Semantic Priority Aggregation
      ↓
  2.5D Multi-Layer Grid Map (GridMap25D)
"""

import math
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd

from src.types import (
    SuperClass,
    PointCloudFrame,
    FoveationBand,
    CellState,
    GridCell25D,
    FoveatedGridConfig
)


DEFAULT_FROZEN_BANDS: List[FoveationBand] = [
    FoveationBand(name="near_field", min_range=0.0, max_range=10.0, voxel_size=0.05),
    FoveationBand(name="mid_near_field", min_range=10.0, max_range=30.0, voxel_size=0.10),
    FoveationBand(name="mid_far_field", min_range=30.0, max_range=60.0, voxel_size=0.25),
    FoveationBand(name="far_field", min_range=60.0, max_range=100.0, voxel_size=0.50),
]


def distance_to_band(
    distance: float,
    bands: Optional[List[FoveationBand]] = None
) -> Optional[FoveationBand]:
    """
    Resolves the exact FoveationBand for a given horizontal LiDAR distance r = sqrt(x^2 + y^2).
    Uses strict half-open intervals: [min_range, max_range).
    Returns None if distance is invalid (NaN, Inf, negative, or >= 100.0m).
    """
    if not isinstance(distance, (int, float)) or not math.isfinite(distance):
        return None
    if distance < 0.0 or distance >= 100.0:
        return None

    active_bands = bands if bands is not None else DEFAULT_FROZEN_BANDS
    for band in active_bands:
        if band.min_range <= distance < band.max_range:
            return band

    return None


def distance_to_resolution(
    distance: float,
    bands: Optional[List[FoveationBand]] = None
) -> Optional[float]:
    """
    Maps horizontal distance r to its cell resolution:
      [0.0, 10.0)  -> 0.05 m
      [10.0, 30.0) -> 0.10 m
      [30.0, 60.0) -> 0.25 m
      [60.0, 100.0)-> 0.50 m
      >= 100.0 m   -> None (Out of operational range)
    """
    band = distance_to_band(distance, bands=bands)
    return band.voxel_size if band is not None else None


def xy_to_cell(x: float, y: float, resolution: float) -> Tuple[int, int]:
    """
    Maps continuous 2D coordinates (x, y) to discrete cell index (ix, iy)
    using mathematical floor:
      ix = floor(x / resolution)
      iy = floor(y / resolution)
    Correctly handles negative coordinates without truncation towards zero.
    """
    if not math.isfinite(x) or not math.isfinite(y) or resolution <= 0.0:
        raise ValueError(f"Invalid inputs to xy_to_cell: x={x}, y={y}, resolution={resolution}")
    ix = int(math.floor(x / resolution))
    iy = int(math.floor(y / resolution))
    return (ix, iy)


def cell_to_bounds(ix: int, iy: int, resolution: float) -> Tuple[float, float, float, float]:
    """Returns spatial bounding box (min_x, max_x, min_y, max_y) for cell (ix, iy)."""
    min_x = ix * resolution
    max_x = (ix + 1) * resolution
    min_y = iy * resolution
    max_y = (iy + 1) * resolution
    return (min_x, max_x, min_y, max_y)


def point_to_cell(
    x: float,
    y: float,
    z: float,
    semantic_class: int = SuperClass.IGNORE_LABEL,
    confidence: float = 1.0,
    bands: Optional[List[FoveationBand]] = None
) -> Optional[GridCell25D]:
    """
    Transforms a single 3D LiDAR point to its corresponding 2.5D spatial cell.
    Returns None if point is out of range or non-finite.
    """
    if not math.isfinite(x) or not math.isfinite(y) or not math.isfinite(z):
        return None

    r = math.sqrt(x * x + y * y)
    band = distance_to_band(r, bands=bands)
    if band is None:
        return None

    ix, iy = xy_to_cell(x, y, band.voxel_size)

    # Calculate traversability score
    trav = 0.0
    if semantic_class == SuperClass.DRIVABLE_TERRAIN:
        trav = 1.0
    elif semantic_class == SuperClass.NON_DRIVABLE_TERRAIN:
        trav = 0.2
    else:
        trav = 0.0

    return GridCell25D(
        ix=ix,
        iy=iy,
        resolution=band.voxel_size,
        point_count=1,
        elevation_mean=float(z),
        elevation_min=float(z),
        elevation_max=float(z),
        semantic_class=int(semantic_class),
        confidence=float(confidence),
        traversability=trav,
        state=CellState.OCCUPIED,
        band_name=band.name
    )


class GridMap25D:
    """
    Multi-Layer 2.5D Foveated Grid Map Representation.
    Stores spatial cells indexed by (band_name, ix, iy).
    Spatial identity is strictly 2D (ix, iy). Elevation Z is an aggregated cell attribute.
    """
    def __init__(
        self,
        bands: Optional[List[FoveationBand]] = None,
        frame_id: str = "000000",
        timestamp: float = 0.0,
        sequence_id: str = "00"
    ):
        self.bands = list(bands) if bands is not None else list(DEFAULT_FROZEN_BANDS)
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.sequence_id = sequence_id
        self.cells: Dict[Tuple[str, int, int], GridCell25D] = {}

    @property
    def num_cells(self) -> int:
        return len(self.cells)

    @property
    def num_occupied_cells(self) -> int:
        return sum(1 for c in self.cells.values() if c.state == CellState.OCCUPIED)

    def get_cell(self, band_name: str, ix: int, iy: int) -> GridCell25D:
        """
        Retrieves cell at (band_name, ix, iy).
        If cell was never observed, returns an UNKNOWN state cell with NaN elevation.
        """
        key = (band_name, ix, iy)
        if key in self.cells:
            return self.cells[key]

        # Find band resolution
        res = 0.05
        for b in self.bands:
            if b.name == band_name:
                res = b.voxel_size
                break

        return GridCell25D(
            ix=ix,
            iy=iy,
            resolution=res,
            point_count=0,
            elevation_mean=float("nan"),
            elevation_min=float("nan"),
            elevation_max=float("nan"),
            semantic_class=SuperClass.IGNORE_LABEL,
            confidence=0.0,
            traversability=0.0,
            state=CellState.UNKNOWN,
            band_name=band_name
        )

    def get_cell_at_xy(self, x: float, y: float) -> Optional[GridCell25D]:
        """Queries the observed 2.5D cell at real-world coordinates (x, y)."""
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        r = math.sqrt(x * x + y * y)
        band = distance_to_band(r, bands=self.bands)
        if band is None:
            return None
        ix, iy = xy_to_cell(x, y, band.voxel_size)
        return self.get_cell(band.name, ix, iy)

    def insert_cell(self, cell: GridCell25D):
        """Inserts or overwrites a cell in the grid map."""
        key = (cell.band_name, cell.ix, cell.iy)
        self.cells[key] = cell

    def to_dataframe(self) -> pd.DataFrame:
        """Exports all observed cells into a tabular pandas DataFrame."""
        rows = []
        for c in self.cells.values():
            min_x, max_x, min_y, max_y = c.bounds
            rows.append({
                "band_name": c.band_name,
                "ix": c.ix,
                "iy": c.iy,
                "resolution": c.resolution,
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
                "center_x": (min_x + max_x) / 2.0,
                "center_y": (min_y + max_y) / 2.0,
                "point_count": c.point_count,
                "elevation_mean": c.elevation_mean,
                "elevation_min": c.elevation_min,
                "elevation_max": c.elevation_max,
                "semantic_class": c.semantic_class,
                "confidence": c.confidence,
                "traversability": c.traversability,
                "state": c.state.name
            })
        return pd.DataFrame(rows)


class FoveatedGrid25D:
    """
    Core Phase-2 Foveated 2.5D Grid Builder.
    Processes LiDAR points, executes distance-aware 2D XY cell indexing, and performs:
      1. Elevation aggregation: mean(z), min(z), max(z)
      2. Deterministic obstacle-preserving semantic priority aggregation:
         dynamic_object (3) > static_obstacle (2) > non_drivable (1) > drivable (0) > ignore (255)
      3. Traversability estimation and confidence propagation.
    """
    def __init__(
        self,
        bands: Optional[List[FoveationBand]] = None,
        max_range: float = 100.0
    ):
        self.bands = list(bands) if bands is not None else list(DEFAULT_FROZEN_BANDS)
        self.max_range = float(max_range)

    def build_grid(
        self,
        points: np.ndarray,
        labels: Optional[np.ndarray] = None,
        confidences: Optional[np.ndarray] = None,
        frame_id: str = "000000",
        timestamp: float = 0.0,
        sequence_id: str = "00"
    ) -> GridMap25D:
        """
        Constructs a GridMap25D from raw or preprocessed LiDAR arrays.
        points: float32 [N, 4] -> (x, y, z, intensity)
        labels: int/uint32 [N] -> super-classes
        confidences: float32 [N] -> prediction confidence
        """
        grid_map = GridMap25D(
            bands=self.bands,
            frame_id=frame_id,
            timestamp=timestamp,
            sequence_id=sequence_id
        )

        if points is None or len(points) == 0:
            return grid_map

        N = len(points)
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        r = np.sqrt(x * x + y * y)

        lbls = labels if labels is not None else np.full(N, SuperClass.IGNORE_LABEL, dtype=np.int64)
        confs = confidences if confidences is not None else np.ones(N, dtype=np.float32)

        # Semantic priority weights for aggregation:
        # dynamic (4) > static (3) > non-drivable (2) > drivable (1) > ignore (0)
        p_weights = np.zeros(256, dtype=np.int32)
        p_weights[SuperClass.DYNAMIC_OBJECT] = 4
        p_weights[SuperClass.STATIC_OBSTACLE] = 3
        p_weights[SuperClass.NON_DRIVABLE_TERRAIN] = 2
        p_weights[SuperClass.DRIVABLE_TERRAIN] = 1
        p_weights[SuperClass.IGNORE_LABEL] = 0

        # Process each band independently
        for band in self.bands:
            mask = (r >= band.min_range) & (r < band.max_range) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            if np.sum(mask) == 0:
                continue

            bx = x[mask]
            by = y[mask]
            bz = z[mask]
            bl = lbls[mask]
            bc = confs[mask]
            res = band.voxel_size

            # Mathematical floor for 2D XY cell indexing
            b_ix = np.floor(bx / res).astype(np.int64)
            b_iy = np.floor(by / res).astype(np.int64)

            # Unique 2D XY cells
            cell_keys = np.column_stack([b_ix, b_iy])
            unique_keys, inverse_idx = np.unique(cell_keys, axis=0, return_inverse=True)
            num_cells = len(unique_keys)

            # 1. Point counts per cell
            counts = np.bincount(inverse_idx, minlength=num_cells)

            # 2. Elevation Mean
            sum_z = np.bincount(inverse_idx, weights=bz, minlength=num_cells)
            mean_z = sum_z / counts

            # 3. Elevation Min & Max (using np.minimum.at / np.maximum.at)
            min_z = np.full(num_cells, np.inf, dtype=np.float32)
            max_z = np.full(num_cells, -np.inf, dtype=np.float32)
            np.minimum.at(min_z, inverse_idx, bz)
            np.maximum.at(max_z, inverse_idx, bz)

            # 4. Confidence Mean
            sum_c = np.bincount(inverse_idx, weights=bc, minlength=num_cells)
            mean_c = sum_c / counts

            # 5. Obstacle-preserving semantic class aggregation
            ranks = p_weights[np.clip(bl, 0, 255)]
            # Sort by rank descending so first unique per cell has highest priority
            sort_order = np.lexsort((-ranks, inverse_idx))
            _, first_idx = np.unique(inverse_idx[sort_order], return_index=True)
            best_idx = sort_order[first_idx]
            agg_labels = bl[best_idx]

            # Populate GridMap25D cells
            for c_idx in range(num_cells):
                ix_val = int(unique_keys[c_idx, 0])
                iy_val = int(unique_keys[c_idx, 1])
                s_cls = int(agg_labels[c_idx])

                # Traversability mapping
                if s_cls == SuperClass.DRIVABLE_TERRAIN:
                    trav = 1.0
                elif s_cls == SuperClass.NON_DRIVABLE_TERRAIN:
                    trav = 0.2
                else:
                    trav = 0.0

                cell = GridCell25D(
                    ix=ix_val,
                    iy=iy_val,
                    resolution=res,
                    point_count=int(counts[c_idx]),
                    elevation_mean=float(mean_z[c_idx]),
                    elevation_min=float(min_z[c_idx]),
                    elevation_max=float(max_z[c_idx]),
                    semantic_class=s_cls,
                    confidence=float(mean_c[c_idx]),
                    traversability=trav,
                    state=CellState.OCCUPIED,
                    band_name=band.name
                )
                grid_map.insert_cell(cell)

        return grid_map
