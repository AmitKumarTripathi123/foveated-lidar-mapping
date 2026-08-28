"""
Phase 20 Foveation Boundary, Epsilon Coordinate, and Edge Case Audit (SIH PS 26130).
Tests:
1. Exact float epsilon boundary thresholds (0.5m, 10m, 40m, 100m).
2. Origin, negative coordinates, and voxel boundary corner cases.
3. Empty point clouds and extreme spatial coordinates.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.range_filter import RangeFilter
from src.core.native_foveation import NativeFoveationAccelerator
from ml.data.spvcnn_adapter import SPVCNNInputAdapter
from src.core.native_grid import NativeGridMapRasterizer


def run_boundary_audit() -> Dict[str, Any]:
    range_filter = RangeFilter(min_range=0.5, max_range=100.0)
    fov_sampler = NativeFoveationAccelerator()
    adapter = SPVCNNInputAdapter(voxel_size=0.05)
    grid_rasterizer = NativeGridMapRasterizer()

    results = {}

    # ------------------------------------------------------------
    # 1. Range Filter Boundary Epsilon Tests
    # ------------------------------------------------------------
    eps_points = np.array([
        [0.4999, 0.0, 0.0, 1.0],   # Below min range (filter out)
        [0.5000, 0.0, 0.0, 1.0],   # Exact min range (keep)
        [0.5001, 0.0, 0.0, 1.0],   # Above min range (keep)
        [99.9999, 0.0, 0.0, 1.0],  # Below max range (keep)
        [100.0000, 0.0, 0.0, 1.0], # Exact max range (keep)
        [100.0001, 0.0, 0.0, 1.0], # Above max range (filter out)
    ], dtype=np.float32)

    filtered_pts, mask = range_filter.filter(eps_points)
    rf_checks = {
        "0_4999m_filtered_out": bool(not mask[0]),
        "0_5000m_kept": bool(mask[1]),
        "0_5001m_kept": bool(mask[2]),
        "99_9999m_kept": bool(mask[3]),
        "100_0000m_kept": bool(mask[4]),
        "100_0001m_filtered_out": bool(not mask[5]),
    }
    results["range_filter_boundaries"] = {
        "checks": rf_checks,
        "status": "PASS" if all(rf_checks.values()) else "FAIL",
    }

    # ------------------------------------------------------------
    # 2. Foveation Zone Boundary Epsilon Tests (Orthogonal Azimuths)
    # ------------------------------------------------------------
    zone_pts = np.array([
        [9.9999, 0.0, 0.0, 1.0],    # Near zone (< 10m) on +X
        [0.0, 10.0000, 0.0, 1.0],   # Near/Mid boundary on +Y
        [-10.0001, 0.0, 0.0, 1.0],  # Mid zone (> 10m) on -X
        [0.0, -39.9999, 0.0, 1.0],  # Mid zone (< 40m) on -Y
        [40.0000, 0.0, 0.0, 1.0],   # Mid/Far boundary on +X
        [0.0, 40.0001, 0.0, 1.0],   # Far zone (> 40m) on +Y
        [99.9999, 0.0, 0.0, 1.0],   # Far zone max bound on +X
        [0.0, 100.0000, 0.0, 1.0],  # Far zone exact limit on +Y
    ], dtype=np.float32)

    fov_out, _, rep = fov_sampler.sample(zone_pts)
    fov_checks = {
        "near_boundary_sampled": len(fov_out) >= 1,
        "all_orthogonal_boundary_points_retained": len(fov_out) == len(zone_pts),
    }
    results["foveation_zone_boundaries"] = {
        "checks": fov_checks,
        "foveated_count": len(fov_out),
        "status": "PASS" if all(fov_checks.values()) else "FAIL",
    }

    # ------------------------------------------------------------
    # 3. Coordinate Edge Cases: Origin, Negative, Extreme
    # ------------------------------------------------------------
    edge_pts = np.array([
        [0.0, 0.0, 0.0, 0.0],          # Exact origin
        [-5.0, -5.0, -1.0, 0.5],       # Negative coordinates Near
        [-25.0, -15.0, 0.0, 0.8],      # Negative coordinates Mid
        [-75.0, -40.0, 2.0, 0.2],      # Negative coordinates Far
        [49.99, 49.99, 0.0, 1.0],      # Grid upper boundary
        [-49.99, -49.99, 0.0, 1.0],    # Grid lower boundary
    ], dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pts_t = torch.from_numpy(edge_pts).to(device).float()
    bundle = adapter.prepare_input(pts_t, device=device)

    fake_preds = torch.zeros(len(edge_pts), dtype=torch.int64, device=device)
    fake_confs = torch.ones(len(edge_pts), dtype=torch.float32, device=device)
    grid = grid_rasterizer.rasterize(bundle["xyz"], fake_preds, fake_confs, mode="cuda" if device.type == "cuda" else "cpu")

    edge_checks = {
        "origin_quantized": bundle["num_voxels"] > 0,
        "negative_coords_indexed": bundle["point_to_voxel_idx"].min().item() >= 0,
        "grid_rasterized_valid": grid.semantic_layer.shape == (500, 500),
    }
    results["coordinate_edge_cases"] = {
        "checks": edge_checks,
        "status": "PASS" if all(edge_checks.values()) else "FAIL",
    }

    # ------------------------------------------------------------
    # 4. Empty Cloud Invariant Test
    # ------------------------------------------------------------
    empty_pts = np.zeros((0, 4), dtype=np.float32)
    empty_f, _ = range_filter.filter(empty_pts)
    empty_fov, _, _ = fov_sampler.sample(empty_f)
    empty_t = torch.zeros((0, 4), device=device, dtype=torch.float32)
    empty_bundle = adapter.prepare_input(empty_t, device=device)

    empty_checks = {
        "empty_range_filter_zero": len(empty_f) == 0,
        "empty_foveation_zero": len(empty_fov) == 0,
        "empty_bundle_zero": empty_bundle["num_points"] == 0 and empty_bundle["num_voxels"] == 0,
    }
    results["empty_cloud_invariants"] = {
        "checks": empty_checks,
        "status": "PASS" if all(empty_checks.values()) else "FAIL",
    }

    overall_pass = all(v["status"] == "PASS" for v in results.values())
    results["overall_status"] = "ALL_BOUNDARY_CHECKS_PASSED" if overall_pass else "BOUNDARY_CHECKS_FAILED"
    return results


if __name__ == "__main__":
    out_dir = REPO_ROOT / "reports/phase20"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = run_boundary_audit()
    with open(out_dir / "boundary_audit.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"Boundary Audit Complete: {res['overall_status']}")
