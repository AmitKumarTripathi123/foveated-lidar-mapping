import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import torch

from src.core.foveated_grid import HierarchicalFoveatedGridEngine
from src.core.lidar_loader import load_lidar_points
from src.inference.predictor import CanonicalPredictor
from ml.data.amit_adapter import FoveatedVoxelSampler
from src.core.range_filter import RangeFilter


def run_grid_latency_benchmark(
    config_path: str = "configs/system_config.yaml",
    dataset_dir: str = "dataset/sequences/02/velodyne",
    num_frames: int = 100,
    warmup_frames: int = 10,
    out_json: Path = Path("reports/phase19_2/native_grid_benchmark.json"),
) -> Dict[str, Any]:
    """Execute isolated rasterization latency benchmark."""
    engine = HierarchicalFoveatedGridEngine(config_path)
    predictor = CanonicalPredictor(config_path)
    range_filter = RangeFilter(min_range=0.5, max_range=100.0)
    foveated_sampler = FoveatedVoxelSampler()

    bin_files = sorted(list(Path(dataset_dir).glob("*.bin")))[:num_frames + warmup_frames]
    if len(bin_files) < num_frames + warmup_frames:
        raise ValueError(f"Insufficient frames in {dataset_dir}")

    # 1. Pre-generate and cache predictions so we isolate GRID RASTERIZATION ONLY
    print(f"Pre-caching perception outputs for {len(bin_files)} frames...")
    cached_data = []
    for f in bin_files:
        pts = load_lidar_points(f)
        pts_f, _ = range_filter.filter(pts)
        fov_pts, _, _ = foveated_sampler.sample(pts_f)
        preds, confs = predictor.predict(fov_pts)
        cached_data.append((fov_pts[:, :3], preds, confs))

    # 2. Warmup
    print("Warming up rasterizers...")
    for i in range(warmup_frames):
        xyz, c, conf = cached_data[i]
        _ = engine.build_25d_grid_reference_python(xyz, c, conf)
        _ = engine.build_25d_grid(xyz, c, conf, use_native=True)

    # 3. Benchmark Python Reference
    print("Benchmarking Reference Python Rasterizer (100 frames)...")
    py_latencies = []
    for i in range(warmup_frames, len(cached_data)):
        xyz, c, conf = cached_data[i]
        t0 = time.perf_counter()
        _ = engine.build_25d_grid_reference_python(xyz, c, conf)
        py_latencies.append((time.perf_counter() - t0) * 1000.0)

    # 4. Benchmark Native C++/LLVM Rasterizer
    print("Benchmarking Native C++/LLVM Rasterizer (100 frames)...")
    native_latencies = []
    for i in range(warmup_frames, len(cached_data)):
        xyz, c, conf = cached_data[i]
        t0 = time.perf_counter()
        _ = engine.build_25d_grid(xyz, c, conf, use_native=True)
        native_latencies.append((time.perf_counter() - t0) * 1000.0)

    # 5. Benchmark CUDA Parallel Tensor Rasterizer
    cuda_latencies = []
    if torch.cuda.is_available():
        print("Benchmarking CUDA Parallel Tensor Rasterizer (100 frames)...")
        cuda_cached = []
        for i in range(len(cached_data)):
            xyz, c, conf = cached_data[i]
            t_xyz = torch.from_numpy(xyz).cuda().float()
            t_c = torch.from_numpy(c).cuda().long()
            t_conf = torch.from_numpy(conf).cuda().float()
            cuda_cached.append((t_xyz, t_c, t_conf))

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)

        for i in range(warmup_frames, len(cuda_cached)):
            t_xyz, t_c, t_conf = cuda_cached[i]
            start_ev.record()
            _ = engine.native_rasterizer.rasterize(t_xyz, t_c, t_conf, mode="cuda")
            end_ev.record()
            torch.cuda.synchronize()
            cuda_latencies.append(float(start_ev.elapsed_time(end_ev)))

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
    native_stats = stats(native_latencies)
    cuda_stats = stats(cuda_latencies) if cuda_latencies else None

    speedup_native = round(py_stats["mean_ms"] / max(native_stats["mean_ms"], 1e-4), 2)
    speedup_cuda = round(py_stats["mean_ms"] / max(cuda_stats["mean_ms"], 1e-4), 2) if cuda_stats else None

    payload = {
        "frames": num_frames,
        "reference_python": py_stats,
        "native_cpp_llvm": native_stats,
        "cuda_parallel_tensor": cuda_stats,
        "speedup_native_cpu": speedup_native,
        "speedup_cuda_tensor": speedup_cuda,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload


if __name__ == "__main__":
    out_file = Path("reports/phase19_2/native_grid_benchmark.json")
    res = run_grid_latency_benchmark(out_json=out_file)
    print(f"Isolated Grid Benchmark Summary:\nPython Reference: {res['reference_python']['mean_ms']} ms\nNative CPU: {res['native_cpp_llvm']['mean_ms']} ms ({res['speedup_native_cpu']}x speedup)\nCUDA Parallel: {res['cuda_parallel_tensor']['mean_ms']} ms ({res['speedup_cuda_tensor']}x speedup)")
