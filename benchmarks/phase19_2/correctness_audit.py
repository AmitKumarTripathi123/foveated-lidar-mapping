import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np

from src.core.native_grid import NativeGridMapRasterizer
from src.core.foveated_grid import HierarchicalFoveatedGridEngine


def compare_grid_maps(grid_ref, grid_native, atol: float = 1e-5) -> Dict[str, Any]:
    """Compare two GridMap25D instances for exact and numerical equivalence."""
    # 1. Occupied cell set equality
    occ_ref = (grid_ref.point_count_layer > 0)
    occ_native = (grid_native.point_count_layer > 0)
    cell_set_match = bool(np.array_equal(occ_ref, occ_native))

    # 2. Point count exact equality
    count_match = bool(np.array_equal(grid_ref.point_count_layer, grid_native.point_count_layer))

    # 3. Elevation Mean Match
    valid_elev_ref = np.nan_to_num(grid_ref.elevation_mean, nan=0.0)
    valid_elev_native = np.nan_to_num(grid_native.elevation_mean, nan=0.0)
    max_elev_mean_err = float(np.max(np.abs(valid_elev_ref[occ_ref] - valid_elev_native[occ_ref]))) if np.any(occ_ref) else 0.0
    elev_mean_match = bool(max_elev_mean_err <= atol)

    # 4. Elevation Min & Max Match
    valid_min_ref = np.nan_to_num(grid_ref.elevation_min, nan=0.0)
    valid_min_native = np.nan_to_num(grid_native.elevation_min, nan=0.0)
    max_min_err = float(np.max(np.abs(valid_min_ref[occ_ref] - valid_min_native[occ_ref]))) if np.any(occ_ref) else 0.0
    elev_min_match = bool(max_min_err <= atol)

    valid_max_ref = np.nan_to_num(grid_ref.elevation_max, nan=0.0)
    valid_max_native = np.nan_to_num(grid_native.elevation_max, nan=0.0)
    max_max_err = float(np.max(np.abs(valid_max_ref[occ_ref] - valid_max_native[occ_ref]))) if np.any(occ_ref) else 0.0
    elev_max_match = bool(max_max_err <= atol)

    # 5. Semantic Layer Exact Equality
    semantic_match = bool(np.array_equal(grid_ref.semantic_layer, grid_native.semantic_layer))

    # 6. Confidence Layer Match
    max_conf_err = float(np.max(np.abs(grid_ref.confidence_layer[occ_ref] - grid_native.confidence_layer[occ_ref]))) if np.any(occ_ref) else 0.0
    conf_match = bool(max_conf_err <= atol)

    # 7. Traversability Layer Match
    trav_match = bool(np.array_equal(grid_ref.traversability_layer, grid_native.traversability_layer))

    all_passed = (
        cell_set_match and count_match and elev_mean_match and
        elev_min_match and elev_max_match and semantic_match and
        conf_match and trav_match
    )

    return {
        "passed": all_passed,
        "cell_set_match": cell_set_match,
        "point_count_match": count_match,
        "elevation_mean_match": elev_mean_match,
        "max_elevation_mean_err": max_elev_mean_err,
        "elevation_min_match": elev_min_match,
        "elevation_max_match": elev_max_match,
        "semantic_layer_match": semantic_match,
        "confidence_layer_match": conf_match,
        "max_confidence_err": max_conf_err,
        "traversability_layer_match": trav_match,
        "occupied_cells": int(np.sum(occ_ref)),
    }


def run_correctness_suite(out_json: Path) -> Dict[str, Any]:
    """Execute complete randomized and edge-case correctness test suite."""
    engine = HierarchicalFoveatedGridEngine()
    seeds = [0, 1, 2, 42, 100]
    test_results = {}

    # A. Randomized Seed Tests (50,000 points each)
    for seed in seeds:
        np.random.seed(seed)
        xyz = np.random.uniform(-45.0, 45.0, (50000, 3)).astype(np.float32)
        classes = np.random.randint(0, 4, 50000).astype(np.int64)
        confidences = np.random.uniform(0.5, 1.0, 50000).astype(np.float32)

        grid_ref = engine.build_25d_grid_reference_python(xyz, classes, confidences)
        grid_nat = engine.build_25d_grid(xyz, classes, confidences, use_native=True)

        res = compare_grid_maps(grid_ref, grid_nat)
        test_results[f"random_seed_{seed}"] = res

    # B. Edge Case 1: Empty point cloud
    grid_ref = engine.build_25d_grid_reference_python(np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32))
    grid_nat = engine.build_25d_grid(np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32), use_native=True)
    test_results["empty_input"] = compare_grid_maps(grid_ref, grid_nat)

    # C. Edge Case 2: Single point
    xyz_1 = np.array([[2.5, 3.5, 1.25]], dtype=np.float32)
    c_1 = np.array([0], dtype=np.int64)
    conf_1 = np.array([0.95], dtype=np.float32)
    grid_ref = engine.build_25d_grid_reference_python(xyz_1, c_1, conf_1)
    grid_nat = engine.build_25d_grid(xyz_1, c_1, conf_1, use_native=True)
    test_results["single_point"] = compare_grid_maps(grid_ref, grid_nat)

    # D. Edge Case 3: Exact Elevation Verification ([1.0, 2.0, 3.0] -> mean 2.0, min 1.0, max 3.0)
    xyz_elev = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]], dtype=np.float32)
    c_elev = np.array([0, 0, 0], dtype=np.int64)
    conf_elev = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    grid_ref = engine.build_25d_grid_reference_python(xyz_elev, c_elev, conf_elev)
    grid_nat = engine.build_25d_grid(xyz_elev, c_elev, conf_elev, use_native=True)
    test_results["elevation_mean_min_max"] = compare_grid_maps(grid_ref, grid_nat)

    all_passed = all(t["passed"] for t in test_results.values())
    summary = {
        "status": "CORRECTNESS_AUDIT_PASSED" if all_passed else "CORRECTNESS_AUDIT_FAILED",
        "all_tests_passed": all_passed,
        "total_test_cases": len(test_results),
        "test_results": test_results,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    out_p = Path("reports/phase19_2/correctness_audit.json")
    res = run_correctness_suite(out_p)
    print(f"Correctness Audit Status: {res['status']} ({res['total_test_cases']} cases evaluated)")
