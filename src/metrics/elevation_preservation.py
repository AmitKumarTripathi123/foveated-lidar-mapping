"""
Elevation Preservation Validation Module.
Evaluates 2.5D elevation loss (MAE, RMSE, p95) across near, mid, and far distance bands.
Specifically quantifies the impact of far-field 50cm voxelization on terrain and obstacle heights.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.types import PointCloudFrame


@dataclass
class BandElevationError:
    band_name: str
    min_range: float
    max_range: float
    evaluated_cells: int
    mae: float   # Mean Absolute Error (meters)
    rmse: float  # Root Mean Square Error (meters)
    p95_error: float  # 95th percentile error (meters)
    max_error: float  # Maximum error (meters)
    elevation_loss_acceptable: bool
    warning_notes: List[str] = field(default_factory=list)


@dataclass
class ElevationPreservationReport:
    grid_resolution: float
    total_evaluated_cells: int
    overall_mae: float
    overall_rmse: float
    overall_p95_error: float
    overall_max_error: float
    near_field_error: BandElevationError
    mid_field_error: BandElevationError
    far_field_error: BandElevationError
    far_field_50cm_assessment: str
    raw_elevation_grid: Optional[np.ndarray] = None
    fov_elevation_grid: Optional[np.ndarray] = None


class ElevationPreservationValidator:
    """
    Computes 2.5D elevation grid maps and validates vertical spatial fidelity.
    """

    def __init__(self, grid_resolution: float = 0.20, max_range: float = 100.0):
        self.grid_resolution = float(grid_resolution)
        self.max_range = float(max_range)

    def _compute_elevation_grid(
        self,
        points: np.ndarray,
        grid_min: float,
        grid_max: float,
        stat: str = "max"
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        Constructs a 2.5D raster grid of elevation Z values over [-max_range, +max_range].
        """
        if points is None or len(points) == 0:
            grid_cells = int(np.ceil((grid_max - grid_min) / self.grid_resolution))
            empty_grid = np.full((grid_cells, grid_cells), np.nan, dtype=np.float32)
            return empty_grid, empty_grid, grid_cells, grid_cells

        xyz = points[:, :3]
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        grid_cells_x = int(np.ceil((grid_max - grid_min) / self.grid_resolution))
        grid_cells_y = int(np.ceil((grid_max - grid_min) / self.grid_resolution))

        ix = np.floor((x - grid_min) / self.grid_resolution).astype(np.int64)
        iy = np.floor((y - grid_min) / self.grid_resolution).astype(np.int64)

        valid_idx = (ix >= 0) & (ix < grid_cells_x) & (iy >= 0) & (iy < grid_cells_y)
        ix = ix[valid_idx]
        iy = iy[valid_idx]
        z = z[valid_idx]
        x_val = x[valid_idx]
        y_val = y[valid_idx]

        flat_idx = ix * grid_cells_y + iy
        total_cells = grid_cells_x * grid_cells_y

        grid_z_flat = np.full(total_cells, -np.inf, dtype=np.float32)
        np.maximum.at(grid_z_flat, flat_idx, z)
        grid_z_flat[grid_z_flat == -np.inf] = np.nan
        grid_z = grid_z_flat.reshape((grid_cells_x, grid_cells_y))

        r_all = np.sqrt(x_val * x_val + y_val * y_val)
        sum_r = np.bincount(flat_idx, weights=r_all, minlength=total_cells)
        counts = np.bincount(flat_idx, minlength=total_cells)
        with np.errstate(divide="ignore", invalid="ignore"):
            grid_r_flat = sum_r / counts
            grid_r_flat[counts == 0] = np.nan
        grid_r = grid_r_flat.reshape((grid_cells_x, grid_cells_y)).astype(np.float32)

        return grid_z, grid_r, grid_cells_x, grid_cells_y

    def evaluate(
        self,
        raw_frame: PointCloudFrame,
        foveated_frame: PointCloudFrame,
        return_grids: bool = False
    ) -> ElevationPreservationReport:
        """
        Measures MAE, RMSE, and p95 elevation errors across distance bands.
        """
        grid_min = -self.max_range
        grid_max = self.max_range

        raw_grid, raw_r, _, _ = self._compute_elevation_grid(raw_frame.points, grid_min, grid_max)
        fov_grid, fov_r, _, _ = self._compute_elevation_grid(foveated_frame.points, grid_min, grid_max)

        # Evaluate on mutually occupied cells
        valid_mask = np.isfinite(raw_grid) & np.isfinite(fov_grid)
        total_cells = int(np.sum(valid_mask))

        if total_cells == 0:
            empty_band = BandElevationError("empty", 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, True)
            return ElevationPreservationReport(
                grid_resolution=self.grid_resolution,
                total_evaluated_cells=0,
                overall_mae=0.0,
                overall_rmse=0.0,
                overall_p95_error=0.0,
                overall_max_error=0.0,
                near_field_error=empty_band,
                mid_field_error=empty_band,
                far_field_error=empty_band,
                far_field_50cm_assessment="No overlapping cells found."
            )

        z_diff = np.abs(raw_grid[valid_mask] - fov_grid[valid_mask])
        r_vals = raw_r[valid_mask]

        overall_mae = float(np.mean(z_diff))
        overall_rmse = float(np.sqrt(np.mean(z_diff ** 2)))
        overall_p95 = float(np.percentile(z_diff, 95))
        overall_max = float(np.max(z_diff))

        def _eval_band(name: str, r_min: float, r_max: float, max_acceptable_rmse: float) -> BandElevationError:
            b_mask = (r_vals >= r_min) & (r_vals < r_max if r_max < self.max_range else r_vals <= r_max)
            b_diff = z_diff[b_mask]
            b_count = len(b_diff)

            if b_count == 0:
                return BandElevationError(name, r_min, r_max, 0, 0.0, 0.0, 0.0, 0.0, True)

            b_mae = float(np.mean(b_diff))
            b_rmse = float(np.sqrt(np.mean(b_diff ** 2)))
            b_p95 = float(np.percentile(b_diff, 95))
            b_max = float(np.max(b_diff))

            acceptable = b_rmse <= max_acceptable_rmse
            warnings = []
            if not acceptable:
                warnings.append(
                    f"Band {name} RMSE ({round(b_rmse, 3)}m) exceeds threshold ({max_acceptable_rmse}m)."
                )

            return BandElevationError(
                band_name=name,
                min_range=r_min,
                max_range=r_max,
                evaluated_cells=b_count,
                mae=round(b_mae, 4),
                rmse=round(b_rmse, 4),
                p95_error=round(b_p95, 4),
                max_error=round(b_max, 4),
                elevation_loss_acceptable=acceptable,
                warning_notes=warnings
            )

        # Band thresholds: Near (0.05m voxel) <= 0.04m, Mid (0.15m voxel) <= 0.12m, Far (0.50m voxel) <= 0.35m
        near_res = _eval_band("near_field (0-10m)", 0.0, 10.0, 0.04)
        mid_res = _eval_band("mid_field (10-40m)", 10.0, 40.0, 0.12)
        far_res = _eval_band("far_field (40-100m)", 40.0, 100.0, 0.35)

        # Far field 50cm impact investigation
        if far_res.rmse > 0.25:
            assessment = (
                f"WARNING: Far-field 0.50m voxelization shows noticeable vertical smoothing (RMSE {far_res.rmse}m, p95 {far_res.p95_error}m). "
                "Low obstacles (< 0.3m) at 40-100m may blend with ground plane; obstacle-preserving aggregation policy strongly recommended."
            )
        else:
            assessment = (
                f"PASSED: Far-field 0.50m voxelization preserves gross terrain profile and tall obstacles with RMSE {far_res.rmse}m (p95 {far_res.p95_error}m)."
            )

        return ElevationPreservationReport(
            grid_resolution=self.grid_resolution,
            total_evaluated_cells=total_cells,
            overall_mae=round(overall_mae, 4),
            overall_rmse=round(overall_rmse, 4),
            overall_p95_error=round(overall_p95, 4),
            overall_max_error=round(overall_max, 4),
            near_field_error=near_res,
            mid_field_error=mid_res,
            far_field_error=far_res,
            far_field_50cm_assessment=assessment,
            raw_elevation_grid=raw_grid if return_grids else None,
            fov_elevation_grid=fov_grid if return_grids else None
        )
