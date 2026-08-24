"""
Canonical 2.5D Foveated Grid Engine (SIH PS 26130).
Transforms 3D semantic LiDAR point clouds into multiresolution 2.5D elevation and traversability grid maps.
"""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import yaml

from src.core.types import (
    SuperClass,
    CellKey,
    GridCell,
    FoveationZone,
    FoveationBand,
    GridCell25D,
    CLASS_NAMES,
)
from src.core.hierarchy import FoveatedHierarchyEngine, CANONICAL_ZONES
from src.core.traversability import compute_class_traversability


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


class HierarchicalFoveatedGridEngine:
    """Canonical Grid Engine generating multiresolution cells and rasterized 2.5D layers."""

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        bounds_x: Tuple[float, float] = (-50.0, 50.0),
        bounds_y: Tuple[float, float] = (-50.0, 50.0),
        resolution: float = 0.20,
    ):
        self.bounds_x = bounds_x
        self.bounds_y = bounds_y
        self.resolution = resolution
        self.hierarchy = FoveatedHierarchyEngine(config_path)

        if config_path is not None:
            p = Path(config_path)
            if p.is_file():
                with open(p, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                if "grid" in cfg:
                    g = cfg["grid"]
                    self.bounds_x = tuple(g.get("bounds_x", bounds_x))
                    self.bounds_y = tuple(g.get("bounds_y", bounds_y))
                    self.resolution = float(g.get("resolution", resolution))

        self.width = int(round((self.bounds_x[1] - self.bounds_x[0]) / self.resolution))
        self.height = int(round((self.bounds_y[1] - self.bounds_y[0]) / self.resolution))
        self.grid_shape = (self.height, self.width)

    def build_hierarchical_cells(
        self,
        xyz: np.ndarray,
        classes: np.ndarray,
        confidences: np.ndarray,
    ) -> Dict[CellKey, GridCell]:
        """Aggregate points into multiresolution hierarchical GridCells."""
        cells: Dict[CellKey, GridCell] = {}
        if xyz.shape[0] == 0:
            return cells

        for i in range(xyz.shape[0]):
            x, y, z = float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2])
            res = self.hierarchy.point_to_cell_key(x, y)
            if res is None:
                continue
            key, zone = res

            c_id = int(classes[i]) if i < len(classes) else int(SuperClass.IGNORE_LABEL)
            conf = float(confidences[i]) if i < len(confidences) else 1.0

            if key not in cells:
                cells[key] = GridCell(
                    key=key,
                    resolution=zone.resolution,
                    elevation_mean=z,
                    elevation_min=z,
                    elevation_max=z,
                    semantic=c_id,
                    confidence=conf,
                    traversability=compute_class_traversability(c_id),
                    point_count=1,
                )
            else:
                c = cells[key]
                prev_n = c.point_count
                c.point_count += 1
                c.elevation_mean = (c.elevation_mean * prev_n + z) / c.point_count
                c.elevation_min = min(c.elevation_min, z)
                c.elevation_max = max(c.elevation_max, z)
                c.confidence = (c.confidence * prev_n + conf) / c.point_count

        return cells

    def build_25d_grid(
        self,
        xyz: np.ndarray,
        classes: np.ndarray,
        confidences: np.ndarray,
    ) -> GridMap25D:
        """Vectorized compilation of multi-layer GridMap25D from point predictions."""
        if xyz.shape[0] == 0:
            return GridMap25D(
                bounds_x=self.bounds_x, bounds_y=self.bounds_y, resolution=self.resolution,
                grid_shape=self.grid_shape,
                elevation_min=np.full(self.grid_shape, np.nan, dtype=np.float32),
                elevation_max=np.full(self.grid_shape, np.nan, dtype=np.float32),
                elevation_mean=np.full(self.grid_shape, np.nan, dtype=np.float32),
                semantic_layer=np.full(self.grid_shape, 255, dtype=np.int64),
                confidence_layer=np.zeros(self.grid_shape, dtype=np.float32),
                traversability_layer=np.full(self.grid_shape, -1.0, dtype=np.float32),
                point_count_layer=np.zeros(self.grid_shape, dtype=np.int32),
            )

        # 1. Bounds Masking
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        valid_bounds = (
            (x >= self.bounds_x[0]) & (x < self.bounds_x[1]) &
            (y >= self.bounds_y[0]) & (y < self.bounds_y[1])
        )

        x_v, y_v, z_v = x[valid_bounds], y[valid_bounds], z[valid_bounds]
        c_v = classes[valid_bounds]
        conf_v = confidences[valid_bounds]

        ix = np.floor((x_v - self.bounds_x[0]) / self.resolution).astype(np.int64)
        iy = np.floor((y_v - self.bounds_y[0]) / self.resolution).astype(np.int64)
        np.clip(ix, 0, self.width - 1, out=ix)
        np.clip(iy, 0, self.height - 1, out=iy)

        flat_idx = iy * self.width + ix
        num_cells = self.height * self.width

        # 2. Point Counts
        point_count_flat = np.bincount(flat_idx, minlength=num_cells).astype(np.int32)
        occupied_mask = point_count_flat > 0

        # 3. Elevation Mean, Min, Max
        sum_z = np.bincount(flat_idx, weights=z_v, minlength=num_cells)
        mean_elev_flat = np.full(num_cells, np.nan, dtype=np.float32)
        mean_elev_flat[occupied_mask] = sum_z[occupied_mask] / point_count_flat[occupied_mask]

        min_elev_flat = np.full(num_cells, np.inf, dtype=np.float32)
        max_elev_flat = np.full(num_cells, -np.inf, dtype=np.float32)
        np.minimum.at(min_elev_flat, flat_idx, z_v)
        np.maximum.at(max_elev_flat, flat_idx, z_v)
        min_elev_flat[~occupied_mask] = np.nan
        max_elev_flat[~occupied_mask] = np.nan

        # 4. Confidence Mean
        sum_conf = np.bincount(flat_idx, weights=conf_v, minlength=num_cells)
        mean_conf_flat = np.zeros(num_cells, dtype=np.float32)
        mean_conf_flat[occupied_mask] = sum_conf[occupied_mask] / point_count_flat[occupied_mask]

        # 5. Dominant Semantic Class Voting
        valid_c_mask = (c_v >= 0) & (c_v <= 3)
        sem_flat = np.full(num_cells, 255, dtype=np.int64)

        if np.any(valid_c_mask):
            flat_v = flat_idx[valid_c_mask]
            classes_v = c_v[valid_c_mask]
            joint_keys = flat_v * 4 + classes_v
            joint_counts = np.bincount(joint_keys, minlength=num_cells * 4).reshape(num_cells, 4)
            best_c = np.argmax(joint_counts, axis=-1)
            has_votes = np.max(joint_counts, axis=-1) > 0
            sem_flat[has_votes] = best_c[has_votes]

        # 6. Traversability Layer
        trav_flat = compute_class_traversability(sem_flat)

        return GridMap25D(
            bounds_x=self.bounds_x,
            bounds_y=self.bounds_y,
            resolution=self.resolution,
            grid_shape=self.grid_shape,
            elevation_min=min_elev_flat.reshape(self.grid_shape),
            elevation_max=max_elev_flat.reshape(self.grid_shape),
            elevation_mean=mean_elev_flat.reshape(self.grid_shape),
            semantic_layer=sem_flat.reshape(self.grid_shape),
            confidence_layer=mean_conf_flat.reshape(self.grid_shape),
            traversability_layer=trav_flat.reshape(self.grid_shape),
            point_count_layer=point_count_flat.reshape(self.grid_shape),
        )


# Backward Compatibility Function Bridges
def distance_to_band(distance: float, bands=None) -> Optional[FoveationBand]:
    """Compatibility alias for legacy tests."""
    engine = FoveatedHierarchyEngine()
    zone = engine.resolve_zone(distance)
    if zone is None:
        return None
    return FoveationBand(name=zone.name, min_range=zone.min_radius, max_range=zone.max_radius, voxel_size=zone.resolution)
