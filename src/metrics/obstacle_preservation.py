"""
Obstacle and Dynamic Object Preservation Validation Module.
Quantifies obstacle recall, loss, spatial grid IoU, and dynamic object retention across distance bands.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.types import PointCloudFrame, SuperClass


@dataclass
class DynamicObjectBandStats:
    band_name: str
    min_range: float
    max_range: float
    raw_points: int
    foveated_points: int
    retention_percentage: float
    is_preserved: bool  # >= at least 1 representative point if raw > 0


@dataclass
class ObstaclePreservationReport:
    raw_obstacle_points: int
    foveated_obstacle_points: int
    obstacle_point_retention: float
    raw_obstacle_grid_cells: int
    foveated_obstacle_grid_cells: int
    obstacle_grid_recall: float     # % of raw obstacle 2D cells that remain occupied
    obstacle_grid_iou: float        # IoU of obstacle occupancy grid
    obstacle_loss_percentage: float # 100 - obstacle_grid_recall
    dynamic_objects_by_band: List[DynamicObjectBandStats] = field(default_factory=list)
    far_field_dynamic_survival_rate: float = 100.0
    findings: List[str] = field(default_factory=list)


class ObstaclePreservationValidator:
    """
    Measures spatial coverage and semantic retention of safety-critical obstacles and dynamic objects.
    """

    def __init__(self, grid_resolution: float = 0.25, max_range: float = 100.0):
        self.grid_resolution = float(grid_resolution)
        self.max_range = float(max_range)

    def _get_obstacle_occupancy_grid(self, frame: PointCloudFrame) -> np.ndarray:
        """Generates a boolean 2D BEV grid of obstacle presence (SuperClasses 2 and 3)."""
        if frame.points is None or len(frame.points) == 0 or frame.labels is None:
            return np.zeros((1, 1), dtype=bool)

        pts = frame.points
        lbls = frame.labels

        obs_mask = (lbls == SuperClass.STATIC_OBSTACLE) | (lbls == SuperClass.DYNAMIC_OBJECT)
        obs_pts = pts[obs_mask]

        grid_cells = int(np.ceil((2.0 * self.max_range) / self.grid_resolution))
        if len(obs_pts) == 0:
            return np.zeros((grid_cells, grid_cells), dtype=bool)

        ix = np.floor((obs_pts[:, 0] + self.max_range) / self.grid_resolution).astype(np.int64)
        iy = np.floor((obs_pts[:, 1] + self.max_range) / self.grid_resolution).astype(np.int64)

        valid = (ix >= 0) & (ix < grid_cells) & (iy >= 0) & (iy < grid_cells)
        ix = ix[valid]
        iy = iy[valid]

        grid = np.zeros((grid_cells, grid_cells), dtype=bool)
        grid[ix, iy] = True
        return grid

    def evaluate(
        self,
        raw_frame: PointCloudFrame,
        foveated_frame: PointCloudFrame
    ) -> ObstaclePreservationReport:
        """
        Evaluates obstacle recall, IoU, and dynamic object retention.
        """
        raw_lbls = raw_frame.labels if raw_frame.labels is not None else np.zeros(len(raw_frame.points), dtype=np.uint32)
        fov_lbls = foveated_frame.labels if foveated_frame.labels is not None else np.zeros(len(foveated_frame.points), dtype=np.uint32)

        # 1. Point counts for obstacles (classes 2 and 3)
        raw_obs_mask = (raw_lbls == SuperClass.STATIC_OBSTACLE) | (raw_lbls == SuperClass.DYNAMIC_OBJECT)
        fov_obs_mask = (fov_lbls == SuperClass.STATIC_OBSTACLE) | (fov_lbls == SuperClass.DYNAMIC_OBJECT)

        raw_obs_pts = int(np.sum(raw_obs_mask))
        fov_obs_pts = int(np.sum(fov_obs_mask))
        pt_retention = (fov_obs_pts / max(raw_obs_pts, 1)) * 100.0 if raw_obs_pts > 0 else 100.0

        # 2. 2D Occupancy Grid Recall & IoU
        raw_grid = self._get_obstacle_occupancy_grid(raw_frame)
        fov_grid = self._get_obstacle_occupancy_grid(foveated_frame)

        raw_occ_cells = int(np.sum(raw_grid))
        fov_occ_cells = int(np.sum(fov_grid))

        intersection = int(np.sum(raw_grid & fov_grid))
        union = int(np.sum(raw_grid | fov_grid))

        grid_recall = (intersection / max(raw_occ_cells, 1)) * 100.0 if raw_occ_cells > 0 else 100.0
        grid_iou = (intersection / max(union, 1)) if union > 0 else 1.0
        loss_pct = 100.0 - grid_recall

        # 3. Dynamic object breakdown across distance bands
        raw_pts = raw_frame.points
        fov_pts = foveated_frame.points

        raw_r = np.sqrt(raw_pts[:, 0]**2 + raw_pts[:, 1]**2) if len(raw_pts) > 0 else np.empty(0)
        fov_r = np.sqrt(fov_pts[:, 0]**2 + fov_pts[:, 1]**2) if len(fov_pts) > 0 else np.empty(0)

        raw_dyn_mask = (raw_lbls == SuperClass.DYNAMIC_OBJECT)
        fov_dyn_mask = (fov_lbls == SuperClass.DYNAMIC_OBJECT)

        band_defs = [
            ("near_field (0-10m)", 0.0, 10.0),
            ("mid_field (10-40m)", 10.0, 40.0),
            ("far_field (40-100m)", 40.0, 100.0)
        ]

        dynamic_bands: List[DynamicObjectBandStats] = []
        far_survival = 100.0

        for b_name, r_min, r_max in band_defs:
            if len(raw_r) > 0:
                b_raw_mask = raw_dyn_mask & (raw_r >= r_min) & (raw_r < r_max if r_max < 100.0 else raw_r <= r_max)
                b_raw_cnt = int(np.sum(b_raw_mask))
            else:
                b_raw_cnt = 0

            if len(fov_r) > 0:
                b_fov_mask = fov_dyn_mask & (fov_r >= r_min) & (fov_r < r_max if r_max < 100.0 else fov_r <= r_max)
                b_fov_cnt = int(np.sum(b_fov_mask))
            else:
                b_fov_cnt = 0

            ret_pct = (b_fov_cnt / max(b_raw_cnt, 1)) * 100.0 if b_raw_cnt > 0 else 100.0
            preserved = (b_fov_cnt > 0) if b_raw_cnt > 0 else True

            if "far_field" in b_name and b_raw_cnt > 0:
                far_survival = ret_pct

            dynamic_bands.append(DynamicObjectBandStats(
                band_name=b_name,
                min_range=r_min,
                max_range=r_max,
                raw_points=b_raw_cnt,
                foveated_points=b_fov_cnt,
                retention_percentage=round(ret_pct, 2),
                is_preserved=preserved
            ))

        findings = []
        if grid_recall >= 90.0:
            findings.append(f"High obstacle occupancy recall achieved: {round(grid_recall, 1)}% (IoU: {round(grid_iou, 3)}).")
        else:
            findings.append(f"Obstacle recall reduced to {round(grid_recall, 1)}%. Check voxel aggregation policy.")

        if far_survival < 5.0 and any(b.raw_points > 0 for b in dynamic_bands if "far_field" in b.band_name):
            findings.append("WARNING: Distant dynamic objects (40-100m) suffer heavy point depletion under 0.50m voxel size.")
        else:
            findings.append("Dynamic objects maintain representative points across all distance bands.")

        return ObstaclePreservationReport(
            raw_obstacle_points=raw_obs_pts,
            foveated_obstacle_points=fov_obs_pts,
            obstacle_point_retention=round(pt_retention, 2),
            raw_obstacle_grid_cells=raw_occ_cells,
            foveated_obstacle_grid_cells=fov_occ_cells,
            obstacle_grid_recall=round(grid_recall, 2),
            obstacle_grid_iou=round(grid_iou, 4),
            obstacle_loss_percentage=round(loss_pct, 2),
            dynamic_objects_by_band=dynamic_bands,
            far_field_dynamic_survival_rate=round(far_survival, 2),
            findings=findings
        )
