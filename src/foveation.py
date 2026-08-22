"""
Distance-Aware Foveated Voxelization Module.
Implements multi-band distance-aware voxelization, uniform baselines, and multiple aggregation policies.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import yaml
import numpy as np
from scipy import stats

from src.types import PointCloudFrame, FoveationBand, AggregationPolicy, SuperClass


@dataclass
class BandFoveationStats:
    band_name: str
    min_range: float
    max_range: float
    voxel_size: float
    raw_points: int
    foveated_points: int
    point_reduction_percentage: float
    compression_ratio: float


@dataclass
class FoveationResult:
    foveated_frame: PointCloudFrame
    raw_points: int
    foveated_points: int
    point_reduction_percentage: float
    compression_ratio: float
    processing_time_ms: float
    fps: float
    aggregation_policy: str
    band_stats: List[BandFoveationStats] = field(default_factory=list)
    configuration_name: str = "default"


class FoveatedVoxelizer:
    """
    Performs distance-aware foveated voxelization or uniform baseline voxelization.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        bands: Optional[List[FoveationBand]] = None,
        default_policy: AggregationPolicy = AggregationPolicy.OBSTACLE_PRESERVING,
        max_range: float = 100.0
    ):
        self.max_range = float(max_range)
        self.default_policy = default_policy
        self.bands: List[FoveationBand] = []

        if bands is not None:
            self.bands = list(bands)
        elif config_path and Path(config_path).exists():
            self.load_config(config_path)
        else:
            # Default experimental configuration
            self.bands = [
                FoveationBand(name="near_field", min_range=0.0, max_range=10.0, voxel_size=0.05),
                FoveationBand(name="mid_field", min_range=10.0, max_range=40.0, voxel_size=0.15),
                FoveationBand(name="far_field", min_range=40.0, max_range=100.0, voxel_size=0.50),
            ]

    def load_config(self, config_path: Union[str, Path]):
        """Loads band configuration from YAML."""
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        self.max_range = float(data.get("max_range", 100.0))
        pol_str = data.get("default_aggregation_policy", "obstacle_preserving")
        self.default_policy = AggregationPolicy(pol_str)

        self.bands = []
        for b in data.get("bands", []):
            self.bands.append(FoveationBand(
                name=b.get("name", f"band_{b['min_range']}_{b['max_range']}"),
                min_range=float(b["min_range"]),
                max_range=float(b["max_range"]),
                voxel_size=float(b["voxel_size"])
            ))

    def _aggregate_voxels(
        self,
        pts: np.ndarray,
        labels: np.ndarray,
        confidences: np.ndarray,
        voxel_size: float,
        policy: AggregationPolicy
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Voxelizes a point cloud chunk with the specified aggregation policy.
        pts: [N, 4] (x, y, z, intensity)
        labels: [N]
        confidences: [N]
        """
        if len(pts) == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.uint32),
                np.empty((0,), dtype=np.float32)
            )

        xyz = pts[:, :3]
        voxel_coords = np.floor(xyz / voxel_size).astype(np.int64)
        unique_voxels, inverse_indices = np.unique(voxel_coords, axis=0, return_inverse=True)
        num_voxels = len(unique_voxels)

        counts = np.bincount(inverse_indices, minlength=num_voxels)
        sum_x = np.bincount(inverse_indices, weights=xyz[:, 0], minlength=num_voxels)
        sum_y = np.bincount(inverse_indices, weights=xyz[:, 1], minlength=num_voxels)
        sum_z = np.bincount(inverse_indices, weights=xyz[:, 2], minlength=num_voxels)
        sum_i = np.bincount(inverse_indices, weights=pts[:, 3], minlength=num_voxels)

        out_pts = np.column_stack([
            sum_x / counts,
            sum_y / counts,
            sum_z / counts,
            sum_i / counts
        ]).astype(np.float32)

        if policy == AggregationPolicy.NEAREST:
            # Nearest point to voxel center
            v_centers = (unique_voxels + 0.5) * voxel_size
            # Distance of each point to its corresponding voxel center
            pt_centers = v_centers[inverse_indices]
            dists = np.sum((xyz - pt_centers) ** 2, axis=1)
            sort_order = np.lexsort((dists, inverse_indices))
            _, first_idx = np.unique(inverse_indices[sort_order], return_index=True)
            best_idx = sort_order[first_idx]
            out_pts = pts[best_idx].astype(np.float32)
            out_labels = labels[best_idx].astype(np.uint32)
            out_conf = confidences[best_idx].astype(np.float32)

        elif policy == AggregationPolicy.OBSTACLE_PRESERVING:
            # Priority: Dynamic (3) -> Static (2) -> Non-drivable (1) -> Drivable (0) -> Ignore (255)
            priority_map = np.zeros(256, dtype=np.int32)
            priority_map[SuperClass.DYNAMIC_OBJECT] = 4
            priority_map[SuperClass.STATIC_OBSTACLE] = 3
            priority_map[SuperClass.NON_DRIVABLE_TERRAIN] = 2
            priority_map[SuperClass.DRIVABLE_TERRAIN] = 1
            priority_map[SuperClass.IGNORE_LABEL] = 0

            ranks = priority_map[np.clip(labels, 0, 255)]
            sort_order = np.lexsort((-ranks, inverse_indices))
            _, first_idx = np.unique(inverse_indices[sort_order], return_index=True)
            best_idx = sort_order[first_idx]
            out_labels = labels[best_idx].astype(np.uint32)
            out_conf = confidences[best_idx].astype(np.float32)

        elif policy == AggregationPolicy.CONFIDENCE_WEIGHTED:
            # Sort by confidence descending
            sort_order = np.lexsort((-confidences, inverse_indices))
            _, first_idx = np.unique(inverse_indices[sort_order], return_index=True)
            best_idx = sort_order[first_idx]
            out_labels = labels[best_idx].astype(np.uint32)
            out_conf = confidences[best_idx].astype(np.float32)

        else:
            # CENTROID / MAJORITY default
            # Vectorized dominant label selection
            sort_order = np.argsort(inverse_indices)
            sorted_inv = inverse_indices[sort_order]
            sorted_lbl = labels[sort_order]
            sorted_cnf = confidences[sort_order]

            _, first_idx = np.unique(sorted_inv, return_index=True)
            best_idx = sort_order[first_idx]
            out_labels = sorted_lbl[first_idx].astype(np.uint32)
            out_conf = sorted_cnf[first_idx].astype(np.float32)

        return out_pts, out_labels, out_conf

    def voxelize(
        self,
        frame: PointCloudFrame,
        policy: Optional[AggregationPolicy] = None,
        config_name: str = "foveated"
    ) -> FoveationResult:
        """
        Executes distance-aware foveated voxelization across defined distance bands.
        """
        start_t = time.perf_counter()
        active_policy = policy if policy is not None else self.default_policy

        if frame.points is None or len(frame.points) == 0:
            return FoveationResult(
                foveated_frame=frame.copy(),
                raw_points=0,
                foveated_points=0,
                point_reduction_percentage=0.0,
                compression_ratio=1.0,
                processing_time_ms=0.0,
                fps=0.0,
                aggregation_policy=active_policy.value,
                configuration_name=config_name
            )

        pts = frame.points
        lbls = frame.labels if frame.labels is not None else np.zeros(len(pts), dtype=np.uint32)
        conf = frame.confidences if frame.confidences is not None else np.ones(len(pts), dtype=np.float32)

        # Calculate horizontal radial distance r = sqrt(x^2 + y^2)
        x = pts[:, 0]
        y = pts[:, 1]
        r = np.sqrt(x * x + y * y)

        foveated_pts_list = []
        foveated_lbls_list = []
        foveated_conf_list = []
        band_stats_list: List[BandFoveationStats] = []

        total_raw = len(pts)

        for band in self.bands:
            # Mask points in this band
            if band.max_range >= self.max_range:
                mask = (r >= band.min_range) & (r <= band.max_range)
            else:
                mask = (r >= band.min_range) & (r < band.max_range)

            raw_in_band = int(np.sum(mask))
            if raw_in_band == 0:
                band_stats_list.append(BandFoveationStats(
                    band_name=band.name,
                    min_range=band.min_range,
                    max_range=band.max_range,
                    voxel_size=band.voxel_size,
                    raw_points=0,
                    foveated_points=0,
                    point_reduction_percentage=0.0,
                    compression_ratio=1.0
                ))
                continue

            b_pts = pts[mask]
            b_lbls = lbls[mask]
            b_conf = conf[mask]

            b_out_pts, b_out_lbls, b_out_conf = self._aggregate_voxels(
                b_pts, b_lbls, b_conf, band.voxel_size, active_policy
            )

            foveated_pts_list.append(b_out_pts)
            foveated_lbls_list.append(b_out_lbls)
            foveated_conf_list.append(b_out_conf)

            fov_in_band = len(b_out_pts)
            reduction = ((raw_in_band - fov_in_band) / raw_in_band) * 100.0 if raw_in_band > 0 else 0.0
            comp_ratio = (raw_in_band / max(fov_in_band, 1))

            band_stats_list.append(BandFoveationStats(
                band_name=band.name,
                min_range=band.min_range,
                max_range=band.max_range,
                voxel_size=band.voxel_size,
                raw_points=raw_in_band,
                foveated_points=fov_in_band,
                point_reduction_percentage=round(reduction, 2),
                compression_ratio=round(comp_ratio, 2)
            ))

        if foveated_pts_list:
            all_fov_pts = np.vstack(foveated_pts_list).astype(np.float32)
            all_fov_lbls = np.concatenate(foveated_lbls_list).astype(np.uint32)
            all_fov_conf = np.concatenate(foveated_conf_list).astype(np.float32)
        else:
            all_fov_pts = np.empty((0, 4), dtype=np.float32)
            all_fov_lbls = np.empty((0,), dtype=np.uint32)
            all_fov_conf = np.empty((0,), dtype=np.float32)

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0

        total_fov = len(all_fov_pts)
        total_reduction = ((total_raw - total_fov) / total_raw) * 100.0 if total_raw > 0 else 0.0
        total_comp = (total_raw / max(total_fov, 1))

        foveated_frame = PointCloudFrame(
            points=all_fov_pts,
            labels=all_fov_lbls,
            confidences=all_fov_conf,
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            sequence_id=frame.sequence_id,
            is_valid=frame.is_valid,
            validation_notes=list(frame.validation_notes),
            metadata=dict(
                frame.metadata,
                foveated=True,
                policy=active_policy.value,
                point_reduction=total_reduction
            )
        )

        return FoveationResult(
            foveated_frame=foveated_frame,
            raw_points=total_raw,
            foveated_points=total_fov,
            point_reduction_percentage=round(total_reduction, 2),
            compression_ratio=round(total_comp, 2),
            processing_time_ms=round(elapsed_ms, 3),
            fps=round(fps, 2),
            aggregation_policy=active_policy.value,
            band_stats=band_stats_list,
            configuration_name=config_name
        )

    def uniform_voxelize(
        self,
        frame: PointCloudFrame,
        voxel_size: float,
        policy: Optional[AggregationPolicy] = None
    ) -> FoveationResult:
        """
        Baseline B: Uniform voxelization across the entire range with a fixed voxel size.
        """
        start_t = time.perf_counter()
        active_policy = policy if policy is not None else self.default_policy

        if frame.points is None or len(frame.points) == 0:
            return FoveationResult(
                foveated_frame=frame.copy(),
                raw_points=0,
                foveated_points=0,
                point_reduction_percentage=0.0,
                compression_ratio=1.0,
                processing_time_ms=0.0,
                fps=0.0,
                aggregation_policy=active_policy.value,
                configuration_name=f"uniform_{voxel_size:.2f}m"
            )

        pts = frame.points
        lbls = frame.labels if frame.labels is not None else np.zeros(len(pts), dtype=np.uint32)
        conf = frame.confidences if frame.confidences is not None else np.ones(len(pts), dtype=np.float32)

        out_pts, out_lbls, out_conf = self._aggregate_voxels(
            pts, lbls, conf, voxel_size, active_policy
        )

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0

        total_raw = len(pts)
        total_fov = len(out_pts)
        reduction = ((total_raw - total_fov) / total_raw) * 100.0 if total_raw > 0 else 0.0
        comp = (total_raw / max(total_fov, 1))

        foveated_frame = PointCloudFrame(
            points=out_pts,
            labels=out_lbls,
            confidences=out_conf,
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            sequence_id=frame.sequence_id,
            is_valid=frame.is_valid,
            validation_notes=list(frame.validation_notes),
            metadata=dict(frame.metadata, uniform_voxel_size=voxel_size)
        )

        return FoveationResult(
            foveated_frame=foveated_frame,
            raw_points=total_raw,
            foveated_points=total_fov,
            point_reduction_percentage=round(reduction, 2),
            compression_ratio=round(comp, 2),
            processing_time_ms=round(elapsed_ms, 3),
            fps=round(fps, 2),
            aggregation_policy=active_policy.value,
            band_stats=[BandFoveationStats(
                band_name="uniform_full_range",
                min_range=0.0,
                max_range=self.max_range,
                voxel_size=voxel_size,
                raw_points=total_raw,
                foveated_points=total_fov,
                point_reduction_percentage=round(reduction, 2),
                compression_ratio=round(comp, 2)
            )],
            configuration_name=f"uniform_{voxel_size:.2f}m"
        )
