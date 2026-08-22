#!/usr/bin/env python3
"""Phase 8 Pipeline Latency & Throughput Benchmark.

Measures stage-by-stage runtime for:
  1. Raw LiDAR Loading
  2. Amit Foveated Voxelization
  3. Point Count Normalization
  4. PointNet++ Forward Inference
  5. Prediction Contract Construction
  6. Phase 6 2.5D Mapping Grid Accumulation
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.dataset import load_point_cloud, load_labels
from ml.data.preprocessing import filter_invalid_points
from ml.data.amit_adapter import FoveatedVoxelSampler
from ml.data.foveated_dataset import normalize_point_count
from ml.models.pointnet2 import build_model
from ml.models.predictor import PointNet2Predictor
from ml.models.mapping_adapter import MLToMappingAdapter


def benchmark_pipeline(
    bin_path: Path,
    lbl_path: Path,
    num_points: int = 1024,
    iterations: int = 10,
    device_str: str = "cpu",
) -> dict:
    """Run stage-by-stage latency benchmark."""
    device = torch.device(device_str)
    model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4).to(device)
    model.eval()
    predictor = PointNet2Predictor(model=model, device=device_str)
    adapter = MLToMappingAdapter()
    sampler = FoveatedVoxelSampler()

    times = {
        "load": [],
        "foveated": [],
        "normalize": [],
        "inference": [],
        "mapping": [],
        "total": [],
    }

    # Warmup
    for _ in range(2):
        raw_pts = load_point_cloud(bin_path)
        v_pts, _, _ = filter_invalid_points(raw_pts)
        fov_pts, _, _ = sampler.sample(v_pts)
        norm_pts, _ = normalize_point_count(fov_pts, None, target_num_points=num_points)
        pred = predictor.predict(norm_pts)
        _ = adapter.build_25d_grid(pred)

    for _ in range(iterations):
        t0 = time.perf_counter()

        # 1. Load
        t_start = time.perf_counter()
        raw_pts = load_point_cloud(bin_path)
        raw_lbl = load_labels(lbl_path) if lbl_path.is_file() else None
        v_pts, v_lbl, _ = filter_invalid_points(raw_pts, raw_lbl)
        t_load = time.perf_counter() - t_start

        # 2. Foveated Downsample
        t_start = time.perf_counter()
        fov_pts, fov_lbl, _ = sampler.sample(v_pts, v_lbl)
        t_fov = time.perf_counter() - t_start

        # 3. Normalization
        t_start = time.perf_counter()
        norm_pts, _ = normalize_point_count(fov_pts, None, target_num_points=num_points)
        t_norm = time.perf_counter() - t_start

        # 4. PointNet++ Inference & Predictor
        t_start = time.perf_counter()
        pred = predictor.predict(norm_pts)
        t_inf = time.perf_counter() - t_start

        # 5. 2.5D Mapping Grid
        t_start = time.perf_counter()
        grid = adapter.build_25d_grid(pred)
        t_map = time.perf_counter() - t_start

        t_total = time.perf_counter() - t0

        times["load"].append(t_load * 1000.0)
        times["foveated"].append(t_fov * 1000.0)
        times["normalize"].append(t_norm * 1000.0)
        times["inference"].append(t_inf * 1000.0)
        times["mapping"].append(t_map * 1000.0)
        times["total"].append(t_total * 1000.0)

    summary = {
        "device": device_str,
        "input_points": raw_pts.shape[0],
        "foveated_points": fov_pts.shape[0],
        "normalized_points": num_points,
        "load_ms": float(np.mean(times["load"])),
        "foveated_ms": float(np.mean(times["foveated"])),
        "normalize_ms": float(np.mean(times["normalize"])),
        "inference_ms": float(np.mean(times["inference"])),
        "mapping_ms": float(np.mean(times["mapping"])),
        "total_ms": float(np.mean(times["total"])),
        "throughput_fps": float(1000.0 / np.mean(times["total"])),
    }
    return summary


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Benchmark Phase 8 Pipeline Latency.")
    parser.add_argument("--bin", type=str, default="dataset/sequences/00/velodyne/000000.bin")
    parser.add_argument("--label", type=str, default="dataset/sequences/00/labels/000000.label")
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    bin_path = Path(args.bin)
    lbl_path = Path(args.label)

    if not bin_path.is_file():
        print(f"Error: {bin_path} not found")
        return 1

    print(f"\nBenchmarking Pipeline Latency on {args.device.upper()} (N={args.num_points}, {args.iterations} iterations)...")
    res = benchmark_pipeline(bin_path, lbl_path, num_points=args.num_points, iterations=args.iterations, device_str=args.device)

    print("\n" + "=" * 60)
    print(f"       PHASE 8 END-TO-END PIPELINE LATENCY BENCHMARK ({args.device.upper()})")
    print("=" * 60)
    print(f"Raw Input Points       : {res['input_points']:,}")
    print(f"Foveated Points        : {res['foveated_points']:,}")
    print(f"Normalized Points (N)  : {res['normalized_points']:,}")
    print("-" * 60)
    print(f"1. Raw Data Loading    : {res['load_ms']:>8.2f} ms")
    print(f"2. Foveated Voxelizer  : {res['foveated_ms']:>8.2f} ms")
    print(f"3. Point Normalization : {res['normalize_ms']:>8.2f} ms")
    print(f"4. PointNet++ Inference: {res['inference_ms']:>8.2f} ms")
    print(f"5. 2.5D Mapping Grid   : {res['mapping_ms']:>8.2f} ms")
    print("-" * 60)
    print(f"TOTAL FRAME LATENCY    : {res['total_ms']:>8.2f} ms ({res['throughput_fps']:.2f} FPS)")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
