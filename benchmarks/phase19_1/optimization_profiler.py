"""
Phase 19.1 Optimization Profiler CLI Entry Point.
"""

import argparse
from pathlib import Path
from benchmarks.phase19_1.latency_profiler import CanonicalLatencyProfiler, compute_stage_statistics
from benchmarks.phase19_1.telemetry import TelemetryCollector
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Phase 19.1 Latency & Optimization Profiler.")
    parser.add_argument("--config", type=str, default="configs/system_config.yaml")
    parser.add_argument("--dataset", type=str, default="dataset/sequences/02/velodyne")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=str, default="reports/phase19_1")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    profiler = CanonicalLatencyProfiler(args.config)
    telemetry = TelemetryCollector(profiler.device)

    bin_files = sorted(list(Path(args.dataset).glob("*.bin")))[:args.frames + args.warmup]
    if not bin_files:
        raise FileNotFoundError(f"No .bin files in {args.dataset}")

    # Warmup
    for i in range(min(args.warmup, len(bin_files))):
        _ = profiler.profile_frame(bin_files[i])

    # Profiling
    stage_records = {k: [] for k in ["io", "range_filter", "foveation", "ml_preprocess", "spvcnn", "postprocess", "grid", "visualization"]}
    perception_lats = []
    replay_lats = []

    for i in range(args.warmup, len(bin_files)):
        res = profiler.profile_frame(bin_files[i])
        for k, v in res["stage_latencies_ms"].items():
            stage_records[k].append(v)
        perception_lats.append(res["perception_latency_ms"])
        replay_lats.append(res["replay_latency_ms"])

    stage_stats = compute_stage_statistics(stage_records)
    tele_snap = telemetry.capture_snapshot(fps=1000.0/float(np.mean(perception_lats)))

    profile_payload = {
        "frames": len(perception_lats),
        "perception_latency_mean_ms": round(float(np.mean(perception_lats)), 2),
        "perception_latency_p95_ms": round(float(np.percentile(perception_lats, 95)), 2),
        "replay_latency_mean_ms": round(float(np.mean(replay_lats)), 2),
        "replay_latency_p95_ms": round(float(np.percentile(replay_lats, 95)), 2),
        "stages": stage_stats,
        "telemetry": telemetry.to_dict(tele_snap),
    }

    out_json = out_dir / "optimization_profile.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(profile_payload, f, indent=2)

    print(f"Optimization profile saved to: {out_json}")


if __name__ == "__main__":
    main()
