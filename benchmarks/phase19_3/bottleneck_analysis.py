"""
Phase 19.3 Foveation Baseline Substage Profiler (SIH PS 26130).
Empirically breaks down the reference foveation stage into individual substages
across 100 evaluation frames to measure exact CPU execution time.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.core.lidar_loader import load_lidar_points
from src.core.range_filter import RangeFilter
from ml.data.amit_adapter import voxel_grid_downsample


def profile_foveation_substages(
    dataset_dir: str = "dataset/sequences/02/velodyne",
    num_frames: int = 100,
    warmup_frames: int = 10,
    out_json: Path = Path("reports/phase19_3/foveation_baseline_profile.json"),
) -> Dict[str, Any]:
    """Profile fine-grained substages of reference Python foveation."""
    range_filter = RangeFilter(min_range=0.5, max_range=100.0)
    bin_files = sorted(list(Path(dataset_dir).glob("*.bin")))[:num_frames + warmup_frames]

    # Preload filtered points
    preloaded = []
    for f in bin_files:
        raw_pts = load_lidar_points(f)
        pts_f, _ = range_filter.filter(raw_pts)
        preloaded.append(pts_f)

    substage_times: Dict[str, List[float]] = {
        "distance_calculation_ms": [],
        "zone_classification_ms": [],
        "voxelization_near_ms": [],
        "voxelization_mid_ms": [],
        "voxelization_far_ms": [],
        "deduplication_ms": [],
        "output_construction_ms": [],
        "total_foveation_ms": [],
    }

    n_d2 = 10.0 * 10.0
    m_d2 = 40.0 * 40.0
    f_d2 = 100.0 * 100.0

    print(f"Profiling foveation baseline substages across {num_frames} frames...")
    for idx, points in enumerate(preloaded):
        if idx < warmup_frames:
            continue

        t_total_0 = time.perf_counter()

        # Substage 1: Distance calculation
        t0 = time.perf_counter()
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        d2 = x * x + y * y + z * z
        t_dist = (time.perf_counter() - t0) * 1000.0

        # Substage 2: Zone classification & boolean masking
        t0 = time.perf_counter()
        near_mask = (d2 >= 0.0) & (d2 < n_d2)
        mid_mask = (d2 >= n_d2) & (d2 < m_d2)
        far_mask = (d2 >= m_d2) & (d2 <= f_d2)
        near_pts = points[near_mask]
        mid_pts = points[mid_mask]
        far_pts = points[far_mask]
        t_zone = (time.perf_counter() - t0) * 1000.0

        # Substage 3: Voxelization per zone
        t0 = time.perf_counter()
        near_down, _ = voxel_grid_downsample(near_pts, None, 0.05)
        t_v_near = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        mid_down, _ = voxel_grid_downsample(mid_pts, None, 0.15)
        t_v_mid = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        far_down, _ = voxel_grid_downsample(far_pts, None, 0.50)
        t_v_far = (time.perf_counter() - t0) * 1000.0

        # Substage 4: Output concatenation
        t0 = time.perf_counter()
        foveated_points = np.vstack([near_down, mid_down, far_down])
        t_out = (time.perf_counter() - t0) * 1000.0

        t_total = (time.perf_counter() - t_total_0) * 1000.0

        substage_times["distance_calculation_ms"].append(t_dist)
        substage_times["zone_classification_ms"].append(t_zone)
        substage_times["voxelization_near_ms"].append(t_v_near)
        substage_times["voxelization_mid_ms"].append(t_v_mid)
        substage_times["voxelization_far_ms"].append(t_v_far)
        substage_times["deduplication_ms"].append(t_v_near + t_v_mid + t_v_far)
        substage_times["output_construction_ms"].append(t_out)
        substage_times["total_foveation_ms"].append(t_total)

    profile_summary = {
        "frames_evaluated": num_frames,
        "mean_substage_latencies_ms": {k: round(float(np.mean(v)), 3) for k, v in substage_times.items()},
        "p95_substage_latencies_ms": {k: round(float(np.percentile(v, 95)), 3) for k, v in substage_times.items()},
        "substage_percentage": {
            k: round(float(np.mean(v)) / max(float(np.mean(substage_times["total_foveation_ms"])), 1e-4) * 100.0, 2)
            for k, v in substage_times.items() if k != "total_foveation_ms"
        }
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(profile_summary, f, indent=2)

    return profile_summary


if __name__ == "__main__":
    out_p = Path("reports/phase19_3/foveation_baseline_profile.json")
    res = profile_foveation_substages(out_json=out_p)
    print(f"Foveation Baseline Profile Summary:\nTotal: {res['mean_substage_latencies_ms']['total_foveation_ms']:.2f} ms\nVoxelization & Deduplication: {res['mean_substage_latencies_ms']['deduplication_ms']:.2f} ms ({res['substage_percentage']['deduplication_ms']}%)\nZone Masking: {res['mean_substage_latencies_ms']['zone_classification_ms']:.2f} ms")
