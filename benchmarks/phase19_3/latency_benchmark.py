"""
Phase 19.3 Isolated Foveation Latency Benchmark (SIH PS 26130).
Measures isolated foveated voxel downsampling throughput across 100 evaluation frames,
comparing Reference Python NumPy and Native C++/LLVM Foveation Accelerator.
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
from ml.data.amit_adapter import FoveatedVoxelSampler
from src.core.native_foveation import NativeFoveationAccelerator


def run_foveation_latency_benchmark(
    dataset_dir: str = "dataset/sequences/02/velodyne",
    num_frames: int = 100,
    warmup_frames: int = 10,
    out_json: Path = Path("reports/phase19_3/foveation_benchmark.json"),
) -> Dict[str, Any]:
    """Execute isolated foveation latency benchmark."""
    range_filter = RangeFilter(min_range=0.5, max_range=100.0)
    sampler_ref = FoveatedVoxelSampler()
    sampler_nat = NativeFoveationAccelerator()

    bin_files = sorted(list(Path(dataset_dir).glob("*.bin")))[:num_frames + warmup_frames]
    if len(bin_files) < num_frames + warmup_frames:
        raise ValueError(f"Insufficient frames in {dataset_dir}")

    # 1. Preload points so disk I/O and range filtering are 100% excluded
    print(f"Preloading point clouds for {len(bin_files)} frames...")
    preloaded_points = []
    for f in bin_files:
        raw_pts = load_lidar_points(f)
        pts_f, _ = range_filter.filter(raw_pts)
        preloaded_points.append(pts_f)

    # 2. Warmup
    print("Warming up foveation engines...")
    for i in range(warmup_frames):
        pts = preloaded_points[i]
        _ = sampler_ref.sample_reference_python(pts)
        _ = sampler_nat.sample(pts)

    # 3. Benchmark Python Reference
    print("Benchmarking Reference Python Foveator (100 frames)...")
    py_latencies = []
    for i in range(warmup_frames, len(preloaded_points)):
        pts = preloaded_points[i]
        t0 = time.perf_counter()
        _ = sampler_ref.sample_reference_python(pts)
        py_latencies.append((time.perf_counter() - t0) * 1000.0)

    # 4. Benchmark Native C++/LLVM Foveator
    print("Benchmarking Native C++/LLVM Foveation Accelerator (100 frames)...")
    native_latencies = []
    for i in range(warmup_frames, len(preloaded_points)):
        pts = preloaded_points[i]
        t0 = time.perf_counter()
        _ = sampler_nat.sample(pts)
        native_latencies.append((time.perf_counter() - t0) * 1000.0)

    def stats(arr):
        a = np.array(arr)
        return {
            "mean_ms": round(float(np.mean(a)), 2),
            "median_ms": round(float(np.median(a)), 2),
            "p95_ms": round(float(np.percentile(a, 95)), 2),
            "p99_ms": round(float(np.percentile(a, 99)), 2),
            "min_ms": round(float(np.min(a)), 2),
            "max_ms": round(float(np.max(a)), 2),
            "std_ms": round(float(np.std(a)), 2),
        }

    py_stats = stats(py_latencies)
    nat_stats = stats(native_latencies)
    speedup = round(py_stats["mean_ms"] / max(nat_stats["mean_ms"], 1e-4), 2)

    payload = {
        "frames_evaluated": num_frames,
        "reference_python": py_stats,
        "native_cpp_llvm": nat_stats,
        "speedup_multiplier": speedup,
        "target_met": {
            "mandatory_under_8ms": nat_stats["mean_ms"] < 8.0,
            "strong_under_5ms": nat_stats["mean_ms"] < 5.0,
            "stretch_under_2_5ms": nat_stats["mean_ms"] < 2.5,
        }
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload


if __name__ == "__main__":
    out_p = Path("reports/phase19_3/foveation_benchmark.json")
    res = run_foveation_latency_benchmark(out_json=out_p)
    print(f"Isolated Foveation Benchmark Summary:\nPython Reference: {res['reference_python']['mean_ms']} ms (P95: {res['reference_python']['p95_ms']} ms)\nNative C++/LLVM:  {res['native_cpp_llvm']['mean_ms']} ms (P95: {res['native_cpp_llvm']['p95_ms']} ms)\nSpeedup: {res['speedup_multiplier']}x")
