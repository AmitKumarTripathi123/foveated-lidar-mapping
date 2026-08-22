"""
Standalone Point Cloud Scaling Benchmark (Phase 3).
Evaluates pipeline latency and RAM across target scales: 10K, 100K, 500K, 1M, 5M points.
"""

import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd

# Add repository root to sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.types import PointCloudFrame
from src.range_filter import RangeFilter
from phase2.inference.predictor import Phase2Predictor, SemanticPrediction
from phase2.adapter import MLToMappingAdapter
from benchmarks.phase3.system_monitor import SystemResourceMonitor


def run_scaling_experiment(
    target_counts: List[int],
    model_type: str = "spvcnn",
    checkpoint_path: str = "checkpoints/best_spvcnn.pt",
    device: str = "cpu"
) -> List[Dict[str, Any]]:
    """
    Measures per-stage runtime and memory across point cloud scales.
    """
    monitor = SystemResourceMonitor()
    range_filter = RangeFilter(max_range=100.0, min_range=0.5)
    predictor = Phase2Predictor(model_type=model_type, model_path=checkpoint_path, device=device)
    adapter = MLToMappingAdapter()

    scaling_results = []

    for n_pts in target_counts:
        print(f"  -> Benchmarking Point Scale: N = {n_pts:,} points...")
        np.random.seed(42)
        angles = np.random.uniform(-np.pi, np.pi, n_pts)
        radii = np.random.uniform(0.5, 95.0, n_pts)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        z = np.random.uniform(-1.8, 3.5, n_pts)
        intensity = np.random.uniform(0.1, 0.9, n_pts)
        pts = np.column_stack([x, y, z, intensity]).astype(np.float32)
        lbls = np.random.choice([0, 1, 2, 3, 255], size=n_pts).astype(np.uint32)

        # 1. Load (in-memory)
        t_l0 = time.perf_counter()
        raw_frame = PointCloudFrame(points=pts, labels=lbls, frame_id="scaling_test")
        t_l1 = time.perf_counter()
        load_ms = (t_l1 - t_l0) * 1000.0

        # 2. Preprocess
        t_p0 = time.perf_counter()
        filt_frame, _ = range_filter.filter_frame(raw_frame)
        t_p1 = time.perf_counter()
        preprocess_ms = (t_p1 - t_p0) * 1000.0

        # 3. ML Inference (Chunked for multi-million points to avoid OOM on CPU)
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_m0 = time.perf_counter()
        if n_pts > 200000:
            chunk_size = 100000
            preds_list, probs_list, confs_list = [], [], []
            for c_start in range(0, len(filt_frame.points), chunk_size):
                sub_pts = filt_frame.points[c_start : c_start + chunk_size]
                sub_frame = PointCloudFrame(points=sub_pts, labels=np.zeros(len(sub_pts), dtype=np.uint32))
                sub_pred = predictor.predict_frame(sub_frame)
                preds_list.append(sub_pred.predicted_class)
                probs_list.append(sub_pred.class_probabilities)
                confs_list.append(sub_pred.confidence)
            prediction = SemanticPrediction(
                points=filt_frame.points,
                predicted_class=np.concatenate(preds_list),
                class_probabilities=np.concatenate(probs_list),
                confidence=np.concatenate(confs_list)
            )
        else:
            prediction = predictor.predict_frame(filt_frame)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_m1 = time.perf_counter()
        ml_inference_ms = (t_m1 - t_m0) * 1000.0

        # 4. Grid generation
        t_g0 = time.perf_counter()
        grid_map = adapter.prediction_to_grid(prediction)
        t_g1 = time.perf_counter()
        grid_generation_ms = (t_g1 - t_g0) * 1000.0

        # 5. Visualization prep
        t_v0 = time.perf_counter()
        _ = grid_map.num_occupied_cells
        t_v1 = time.perf_counter()
        vis_ms = (t_v1 - t_v0) * 1000.0

        total_ms = load_ms + preprocess_ms + ml_inference_ms + grid_generation_ms + vis_ms
        fps = 1000.0 / max(total_ms, 1e-4)
        snap = monitor.snapshot()

        scaling_results.append({
            "points": n_pts,
            "grid_cells": grid_map.num_occupied_cells,
            "load_ms": round(load_ms, 2),
            "preprocess_ms": round(preprocess_ms, 2),
            "ml_inference_ms": round(ml_inference_ms, 2),
            "grid_generation_ms": round(grid_generation_ms, 2),
            "visualization_prep_ms": round(vis_ms, 2),
            "total_ms": round(total_ms, 2),
            "fps": round(fps, 2),
            "memory_mb": snap["ram_mb"],
            "cpu_percent": snap["cpu_percent"]
        })

    return scaling_results


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Point Cloud Scaling Benchmark")
    parser.add_argument("--points", nargs="+", type=int, default=[10000, 100000, 500000, 1000000, 5000000])
    parser.add_argument("--model-type", type=str, default="spvcnn")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_spvcnn.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="benchmark_results/phase3/tables/scaling.csv")
    args = parser.parse_args()

    print("================================================================================")
    print("  PHASE 3 POINT CLOUD SCALING BENCHMARK")
    print("================================================================================")
    res = run_scaling_experiment(args.points, args.model_type, args.checkpoint, args.device)
    df = pd.DataFrame(res)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved scaling benchmark table to {args.output}")


if __name__ == "__main__":
    main()
