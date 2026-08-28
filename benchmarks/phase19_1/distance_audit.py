"""
Phase 19.1 Distance-Wise mIoU & Confidence Auditor.
Partitions point cloud into Near (0-10m), Mid (10-40m), and Far (40-100m) zones
and computes per-zone mIoU, class IoUs, confidence distributions, and grid occupancy metrics.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from benchmarks.phase19_1.accuracy_audit import compute_multiclass_metrics, update_confusion_matrix


ZONE_SPECS = {
    "near_0_10m": {"min_r": 0.0, "max_r": 10.0, "level": 0, "voxel_res": 0.05},
    "mid_10_40m": {"min_r": 10.0, "max_r": 40.0, "level": 1, "voxel_res": 0.15},
    "far_40_100m": {"min_r": 40.0, "max_r": 100.0, "level": 2, "voxel_res": 0.50},
}


class DistanceWiseAuditor:
    """Accumulates distance-stratified metrics across multiple evaluation frames."""

    def __init__(self):
        self.zone_cms = {k: np.zeros((4, 4), dtype=np.int64) for k in ZONE_SPECS}
        self.zone_confidences = {k: [] for k in ZONE_SPECS}
        self.zone_total_points = {k: 0 for k in ZONE_SPECS}
        self.zone_occupied_cells = {k: 0 for k in ZONE_SPECS}
        self.filtered_points = 0

    def add_frame(
        self,
        xyz: np.ndarray,
        preds: np.ndarray,
        targets: np.ndarray,
        confs: np.ndarray,
    ):
        """Partition points by horizontal radial distance and accumulate zone metrics."""
        dists = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2)

        for z_key, spec in ZONE_SPECS.items():
            min_r = spec["min_r"]
            max_r = spec["max_r"]
            if z_key == "far_40_100m":
                mask = (dists >= min_r) & (dists <= max_r)
            else:
                mask = (dists >= min_r) & (dists < max_r)

            if np.any(mask):
                self.zone_total_points[z_key] += int(np.sum(mask))
                z_preds = preds[mask]
                z_targets = targets[mask]
                z_confs = confs[mask]

                update_confusion_matrix(self.zone_cms[z_key], z_preds, z_targets)
                self.zone_confidences[z_key].extend(z_confs.tolist())

                # Estimate occupied cells in zone at native zone voxel resolution
                z_xyz = xyz[mask]
                v_res = spec["voxel_res"]
                v_coords = np.floor(z_xyz / v_res).astype(np.int64)
                _, u_idx = np.unique(v_coords, axis=0, return_index=True)
                self.zone_occupied_cells[z_key] += len(u_idx)

        # Count filtered (>100m)
        self.filtered_points += int(np.sum(dists > 100.0))

    def compute_summary(self) -> Dict[str, Any]:
        """Compute full distance-wise audit dictionary."""
        summary = {}

        for z_key, spec in ZONE_SPECS.items():
            metrics = compute_multiclass_metrics(self.zone_cms[z_key])
            conf_list = self.zone_confidences[z_key]

            if conf_list:
                conf_arr = np.array(conf_list, dtype=np.float32)
                mean_c = float(np.mean(conf_arr))
                med_c = float(np.median(conf_arr))
                p10_c = float(np.percentile(conf_arr, 10))
                p90_c = float(np.percentile(conf_arr, 90))
            else:
                mean_c = med_c = p10_c = p90_c = 0.0

            total_pts = self.zone_total_points[z_key]
            occ_cells = self.zone_occupied_cells[z_key]
            pts_per_cell = float(total_pts / max(occ_cells, 1))

            summary[z_key] = {
                "points": total_pts,
                "valid_points": metrics["overall"]["total_valid_points"],
                "miou": metrics["overall"]["miou"],
                "point_accuracy": metrics["overall"]["point_accuracy"],
                "mean_class_accuracy": metrics["overall"]["mean_class_accuracy"],
                "drivable_iou": metrics["classes"]["drivable"]["iou"],
                "non_drivable_iou": metrics["classes"]["non_drivable"]["iou"],
                "static_iou": metrics["classes"]["static"]["iou"],
                "dynamic_iou": metrics["classes"]["dynamic"]["iou"],
                "confidence": {
                    "mean": round(mean_c, 4),
                    "median": round(med_c, 4),
                    "p10": round(p10_c, 4),
                    "p90": round(p90_c, 4),
                },
                "grid_statistics": {
                    "occupied_cells": occ_cells,
                    "mean_points_per_cell": round(pts_per_cell, 2),
                }
            }

        return summary
