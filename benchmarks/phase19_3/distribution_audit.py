"""
Phase 19.3 Distance Zone Distribution & Point Retention Auditor (SIH PS 26130).
Measures Near, Mid, Far, and Filtered point counts and retention percentages across 100 evaluation frames.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.core.lidar_loader import load_lidar_points
from src.core.range_filter import RangeFilter
from src.core.native_foveation import NativeFoveationAccelerator


def run_zone_distribution_audit(
    dataset_dir: str = "dataset/sequences/02/velodyne",
    num_frames: int = 100,
    warmup_frames: int = 10,
    out_json: Path = Path("reports/phase19_3/zone_distribution.json"),
) -> Dict[str, Any]:
    """Audit point distributions and retention statistics across zones."""
    range_filter = RangeFilter(min_range=0.5, max_range=100.0)
    sampler = NativeFoveationAccelerator()
    bin_files = sorted(list(Path(dataset_dir).glob("*.bin")))[:num_frames + warmup_frames]

    totals = {
        "raw_points": 0,
        "range_filtered_points": 0,
        "near_input": 0,
        "near_output": 0,
        "mid_input": 0,
        "mid_output": 0,
        "far_input": 0,
        "far_output": 0,
        "foveated_total": 0,
        "out_of_bounds": 0,
    }

    frame_records = []

    print(f"Auditing zone distribution across {num_frames} frames...")
    for idx, f in enumerate(bin_files):
        if idx < warmup_frames:
            continue

        raw_pts = load_lidar_points(f)
        pts_f, _ = range_filter.filter(raw_pts)
        fov_pts, _, rep = sampler.sample(pts_f)

        near_s = rep.zone_stats[0]
        mid_s = rep.zone_stats[1]
        far_s = rep.zone_stats[2]

        totals["raw_points"] += raw_pts.shape[0]
        totals["range_filtered_points"] += pts_f.shape[0]
        totals["near_input"] += near_s.input_count
        totals["near_output"] += near_s.output_count
        totals["mid_input"] += mid_s.input_count
        totals["mid_output"] += mid_s.output_count
        totals["far_input"] += far_s.input_count
        totals["far_output"] += far_s.output_count
        totals["foveated_total"] += fov_pts.shape[0]
        totals["out_of_bounds"] += rep.filtered_out_count

        frame_records.append({
            "frame": idx - warmup_frames,
            "raw": raw_pts.shape[0],
            "filtered": pts_f.shape[0],
            "near_in": near_s.input_count,
            "near_out": near_s.output_count,
            "mid_in": mid_s.input_count,
            "mid_out": mid_s.output_count,
            "far_in": far_s.input_count,
            "far_out": far_s.output_count,
            "total_out": fov_pts.shape[0],
        })

    def calc_reduc(in_c, out_c):
        return round(((in_c - out_c) / in_c) * 100.0, 2) if in_c > 0 else 0.0

    def calc_retention(in_c, out_c):
        return round((out_c / in_c) * 100.0, 2) if in_c > 0 else 0.0

    n_frames = len(frame_records)
    summary_data = {
        "frames_evaluated": n_frames,
        "mean_per_frame": {
            "raw_points": round(totals["raw_points"] / n_frames, 1),
            "range_filtered_points": round(totals["range_filtered_points"] / n_frames, 1),
            "foveated_retained_points": round(totals["foveated_total"] / n_frames, 1),
            "overall_reduction_pct": calc_reduc(totals["range_filtered_points"], totals["foveated_total"]),
        },
        "zone_breakdown": {
            "near_field_0_10m": {
                "mean_input_points": round(totals["near_input"] / n_frames, 1),
                "mean_output_points": round(totals["near_output"] / n_frames, 1),
                "input_share_pct": round((totals["near_input"] / max(totals["range_filtered_points"], 1)) * 100.0, 2),
                "retention_pct": calc_retention(totals["near_input"], totals["near_output"]),
                "reduction_pct": calc_reduc(totals["near_input"], totals["near_output"]),
                "voxel_size_m": 0.05,
            },
            "mid_field_10_40m": {
                "mean_input_points": round(totals["mid_input"] / n_frames, 1),
                "mean_output_points": round(totals["mid_output"] / n_frames, 1),
                "input_share_pct": round((totals["mid_input"] / max(totals["range_filtered_points"], 1)) * 100.0, 2),
                "retention_pct": calc_retention(totals["mid_input"], totals["mid_output"]),
                "reduction_pct": calc_reduc(totals["mid_input"], totals["mid_output"]),
                "voxel_size_m": 0.15,
            },
            "far_field_40_100m": {
                "mean_input_points": round(totals["far_input"] / n_frames, 1),
                "mean_output_points": round(totals["far_output"] / n_frames, 1),
                "input_share_pct": round((totals["far_input"] / max(totals["range_filtered_points"], 1)) * 100.0, 2),
                "retention_pct": calc_retention(totals["far_input"], totals["far_output"]),
                "reduction_pct": calc_reduc(totals["far_input"], totals["far_output"]),
                "voxel_size_m": 0.50,
            },
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    return summary_data


if __name__ == "__main__":
    out_p = Path("reports/phase19_3/zone_distribution.json")
    res = run_zone_distribution_audit(out_json=out_p)
    print(f"Zone Distribution Summary:\nTotal Input: {res['mean_per_frame']['range_filtered_points']} pts -> Foveated Output: {res['mean_per_frame']['foveated_retained_points']} pts ({res['mean_per_frame']['overall_reduction_pct']}% reduction)")
