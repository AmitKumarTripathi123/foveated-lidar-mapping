"""
Phase 19.3 Correctness & Invariant Auditor (SIH PS 26130).
Performs exhaustive bitwise, zone-boundary, and floating-point tolerance audits comparing
the native accelerated 3-zone foveation engine against the reference Python implementation.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.amit_adapter import FoveatedVoxelSampler
from src.core.native_foveation import NativeFoveationAccelerator


def compare_foveation_results(
    pts_ref: np.ndarray,
    rep_ref,
    pts_nat: np.ndarray,
    rep_nat,
    lbl_ref: np.ndarray = None,
    lbl_nat: np.ndarray = None,
) -> Dict[str, Any]:
    """Compare reference and native foveated sampling results."""
    # 1. Output shape & count exact equality
    count_match = bool(pts_ref.shape[0] == pts_nat.shape[0])

    # 2. Exact bitwise point coordinate equality
    points_match = bool(np.array_equal(pts_ref, pts_nat))
    max_point_err = float(np.max(np.abs(pts_ref - pts_nat))) if count_match and pts_ref.shape[0] > 0 else 0.0

    # 3. Label match
    labels_match = True
    if lbl_ref is not None and lbl_nat is not None:
        labels_match = bool(np.array_equal(lbl_ref, lbl_nat))

    # 4. Zone stats match
    zone_match = True
    if len(rep_ref.zone_stats) == len(rep_nat.zone_stats):
        for z_r, z_n in zip(rep_ref.zone_stats, rep_nat.zone_stats):
            if z_r.input_count != z_n.input_count or z_r.output_count != z_n.output_count:
                zone_match = False
                break
    else:
        zone_match = False

    # 5. Filtered out count match
    filtered_match = bool(rep_ref.filtered_out_count == rep_nat.filtered_out_count)

    all_passed = count_match and points_match and labels_match and zone_match and filtered_match

    return {
        "passed": all_passed,
        "count_match": count_match,
        "points_match": points_match,
        "max_point_err": max_point_err,
        "labels_match": labels_match,
        "zone_match": zone_match,
        "filtered_match": filtered_match,
        "retained_points": int(pts_nat.shape[0]),
        "original_points": int(rep_nat.original_count),
    }


def run_correctness_suite(out_json: Path) -> Dict[str, Any]:
    """Execute complete randomized and edge-case correctness test suite."""
    sampler_ref = FoveatedVoxelSampler()
    sampler_nat = NativeFoveationAccelerator()

    test_results: Dict[str, Any] = {}

    # A. Randomized Seed Tests (50,000 points each)
    seeds = [0, 1, 2, 42, 100]
    for s in seeds:
        np.random.seed(s)
        pts = np.random.uniform(-45.0, 45.0, (50000, 4)).astype(np.float32)
        lbls = np.random.randint(0, 4, 50000).astype(np.int64)

        pts_r, lbl_r, rep_r = sampler_ref.sample_reference_python(pts, lbls)
        pts_n, lbl_n, rep_n = sampler_nat.sample(pts, lbls)

        res = compare_foveation_results(pts_r, rep_r, pts_n, rep_n, lbl_r, lbl_n)
        test_results[f"random_seed_{s}"] = res

    # B. Boundary Points Audit
    boundary_points = []
    # 0.5m boundary
    boundary_points.extend([[0.4999, 0.0, 0.0, 0.5], [0.5000, 0.0, 0.0, 0.5], [0.5001, 0.0, 0.0, 0.5]])
    # 10.0m boundary
    boundary_points.extend([[9.9999, 0.0, 0.0, 0.5], [10.0000, 0.0, 0.0, 0.5], [10.0001, 0.0, 0.0, 0.5]])
    # 40.0m boundary
    boundary_points.extend([[39.9999, 0.0, 0.0, 0.5], [40.0000, 0.0, 0.0, 0.5], [40.0001, 0.0, 0.0, 0.5]])
    # 100.0m boundary
    boundary_points.extend([[99.9999, 0.0, 0.0, 0.5], [100.0000, 0.0, 0.0, 0.5], [100.0001, 0.0, 0.0, 0.5]])
    b_pts = np.array(boundary_points, dtype=np.float32)

    pts_r, _, rep_r = sampler_ref.sample_reference_python(b_pts)
    pts_n, _, rep_n = sampler_nat.sample(b_pts)
    test_results["boundary_points_audit"] = compare_foveation_results(pts_r, rep_r, pts_n, rep_n)

    # C. Special Negative Coordinates Test
    neg_pts = np.array([
        [-10.0, -10.0, 0.0, 0.5],
        [-5.0, 0.0, 0.0, 0.5],
        [0.0, -5.0, 0.0, 0.5],
        [-0.1, -0.1, 0.0, 0.5],
        [-0.05, -0.05, 0.0, 0.5],
        [-35.5, -25.5, 1.2, 0.8],
    ], dtype=np.float32)
    pts_r, _, rep_r = sampler_ref.sample_reference_python(neg_pts)
    pts_n, _, rep_n = sampler_nat.sample(neg_pts)
    test_results["negative_coordinates_test"] = compare_foveation_results(pts_r, rep_r, pts_n, rep_n)

    # D. Dense Near-Field Test (10,000 points within 5m)
    np.random.seed(99)
    dense_near = np.random.uniform(-4.0, 4.0, (10000, 4)).astype(np.float32)
    pts_r, _, rep_r = sampler_ref.sample_reference_python(dense_near)
    pts_n, _, rep_n = sampler_nat.sample(dense_near)
    test_results["dense_near_field"] = compare_foveation_results(pts_r, rep_r, pts_n, rep_n)

    # E. Sparse Far-Field Test (5,000 points between 50m and 90m)
    radii = np.random.uniform(50.0, 90.0, 5000)
    thetas = np.random.uniform(0, 2 * np.pi, 5000)
    sparse_far = np.zeros((5000, 4), dtype=np.float32)
    sparse_far[:, 0] = radii * np.cos(thetas)
    sparse_far[:, 1] = radii * np.sin(thetas)
    sparse_far[:, 2] = np.random.uniform(-2.0, 2.0, 5000)
    sparse_far[:, 3] = 0.5
    pts_r, _, rep_r = sampler_ref.sample_reference_python(sparse_far)
    pts_n, _, rep_n = sampler_nat.sample(sparse_far)
    test_results["sparse_far_field"] = compare_foveation_results(pts_r, rep_r, pts_n, rep_n)

    # F. Edge Cases: Empty & Single Point
    empty_pts = np.zeros((0, 4), dtype=np.float32)
    pts_r, _, rep_r = sampler_ref.sample_reference_python(empty_pts)
    pts_n, _, rep_n = sampler_nat.sample(empty_pts)
    test_results["empty_input"] = compare_foveation_results(pts_r, rep_r, pts_n, rep_n)

    single_pt = np.array([[2.5, 3.5, 1.2, 0.9]], dtype=np.float32)
    pts_r, _, rep_r = sampler_ref.sample_reference_python(single_pt)
    pts_n, _, rep_n = sampler_nat.sample(single_pt)
    test_results["single_point"] = compare_foveation_results(pts_r, rep_r, pts_n, rep_n)

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
    out_p = Path("reports/phase19_3/correctness_audit.json")
    res = run_correctness_suite(out_p)
    print(f"Phase 19.3 Correctness Audit Status: {res['status']} ({res['total_test_cases']} cases evaluated)")
