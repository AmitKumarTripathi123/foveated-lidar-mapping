"""
Phase 3 — End-to-End Performance Benchmark & Profiling Suite.
Measures the baseline performance of the complete foveated 2.5D LiDAR mapping system:
  1. LiDAR Loading
  2. Preprocessing & Range Filtering
  3. ML Semantic Inference (SPVCNN or FoveatedPointSegNet)
  4. 2.5D Foveated Grid Generation
  5. Visualization Data Preparation
  6. Total End-to-End Latency & FPS
"""

import os
import sys
import time
import json
import platform
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import psutil
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

# Add workspace root to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.types import SuperClass, PointCloudFrame
from src.range_filter import RangeFilter
from src.foveated_grid import FoveatedGrid25D, GridMap25D, DEFAULT_FROZEN_BANDS
from phase2.dataset import remap_poss_labels
from phase2.inference.predictor import Phase2Predictor, SemanticPrediction
from phase2.adapter import MLToMappingAdapter


def collect_environment_metadata(model_name: str = "SPVCNN", param_count: int = 136979) -> Dict[str, Any]:
    """Captures the system, OS, hardware, and dependency versions for reproducibility."""
    import yaml
    import scipy

    return {
        "os": platform.platform(),
        "system": platform.system(),
        "processor": platform.processor(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "pandas_version": pd.__version__,
        "matplotlib_version": matplotlib.__version__,
        "pyyaml_version": yaml.__version__,
        "model_name": model_name,
        "model_parameters": param_count,
        "grid_specification": "4-Band (0-10m:5cm, 10-30m:10cm, 30-60m:25cm, 60-100m:50cm)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }


def compute_stage_stats(values: List[float]) -> Dict[str, float]:
    """Calculates summary statistics: mean, median, min, max, std, P95."""
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "p95": 0.0}
    return {
        "mean": round(float(np.mean(arr)), 3),
        "median": round(float(np.median(arr)), 3),
        "min": round(float(np.min(arr)), 3),
        "max": round(float(np.max(arr)), 3),
        "std": round(float(np.std(arr)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3)
    }


def run_pipeline_single_frame(
    bin_path: Path,
    lbl_path: Path,
    range_filter: RangeFilter,
    predictor: Phase2Predictor,
    adapter: MLToMappingAdapter,
    process: psutil.Process
) -> Dict[str, Any]:
    """
    Executes and times the 5 distinct pipeline stages on a single real frame.
    """
    f_id = bin_path.stem

    # --------------------------------------------------------------------------
    # STAGE 1: LiDAR Loading
    # --------------------------------------------------------------------------
    t0 = time.perf_counter()
    raw_pts = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)
    raw_lbls = np.fromfile(str(lbl_path), dtype=np.uint32)
    n_p = min(len(raw_pts), len(raw_lbls))
    raw_pts = raw_pts[:n_p]
    raw_lbls = raw_lbls[:n_p]
    t1 = time.perf_counter()
    load_ms = (t1 - t0) * 1000.0

    # --------------------------------------------------------------------------
    # STAGE 2: Preprocessing (Label Remapping & 100m Range Filtering)
    # --------------------------------------------------------------------------
    t2 = time.perf_counter()
    mapped_lbls = remap_poss_labels(raw_lbls)
    raw_frame = PointCloudFrame(points=raw_pts, labels=mapped_lbls.astype(np.uint32), frame_id=f_id)
    filtered_frame, _ = range_filter.filter_frame(raw_frame)
    t3 = time.perf_counter()
    preprocess_ms = (t3 - t2) * 1000.0

    # --------------------------------------------------------------------------
    # STAGE 3: ML Inference
    # --------------------------------------------------------------------------
    t4 = time.perf_counter()
    prediction = predictor.predict_frame(filtered_frame)
    t5 = time.perf_counter()
    ml_inference_ms = (t5 - t4) * 1000.0

    # --------------------------------------------------------------------------
    # STAGE 4: Grid Generation (2.5D Foveated XY Grid Map)
    # --------------------------------------------------------------------------
    t6 = time.perf_counter()
    grid_map = adapter.prediction_to_grid(prediction)
    t7 = time.perf_counter()
    grid_generation_ms = (t7 - t6) * 1000.0

    # --------------------------------------------------------------------------
    # STAGE 5: Visualization Preparation
    # --------------------------------------------------------------------------
    t8 = time.perf_counter()
    df_cells = grid_map.to_dataframe()
    vis_data = {
        "frame_id": f_id,
        "cell_count": len(df_cells),
        "min_elev": float(df_cells["elevation_min"].min()) if len(df_cells) > 0 else 0.0,
        "max_elev": float(df_cells["elevation_max"].max()) if len(df_cells) > 0 else 0.0,
        "classes_present": df_cells["semantic_class"].unique().tolist() if len(df_cells) > 0 else []
    }
    t9 = time.perf_counter()
    visualization_prep_ms = (t9 - t8) * 1000.0

    # End-to-end total
    total_ms = load_ms + preprocess_ms + ml_inference_ms + grid_generation_ms + visualization_prep_ms
    fps = 1000.0 / max(total_ms, 1e-4)

    # Resource metrics
    cpu_pct = process.cpu_percent()
    mem_mb = process.memory_info().rss / (1024.0 * 1024.0)

    return {
        "frame_id": f_id,
        "input_points": len(raw_pts),
        "filtered_points": len(filtered_frame.points),
        "grid_cells": grid_map.num_occupied_cells,
        "load_ms": round(load_ms, 3),
        "preprocess_ms": round(preprocess_ms, 3),
        "ml_inference_ms": round(ml_inference_ms, 3),
        "grid_generation_ms": round(grid_generation_ms, 3),
        "visualization_prep_ms": round(visualization_prep_ms, 3),
        "total_ms": round(total_ms, 3),
        "fps": round(fps, 2),
        "cpu_percent": round(cpu_pct, 1),
        "memory_mb": round(mem_mb, 2)
    }


def run_scaling_benchmark(
    predictor: Phase2Predictor,
    adapter: MLToMappingAdapter,
    range_filter: RangeFilter,
    process: psutil.Process,
    target_counts: List[int]
) -> List[Dict[str, Any]]:
    """
    Benchmarks pipeline runtime and memory across various point-cloud scales:
    10K, 100K, 500K, 1M, 5M points.
    """
    scaling_results = []

    for n_pts in target_counts:
        print(f"  -> Testing Scaling for N = {n_pts:,} points...")
        np.random.seed(42)
        angles = np.random.uniform(-np.pi, np.pi, n_pts)
        radii = np.random.uniform(0.5, 95.0, n_pts)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        z = np.random.uniform(-1.8, 3.5, n_pts)
        i = np.random.uniform(0.1, 0.9, n_pts)
        pts = np.column_stack([x, y, z, i]).astype(np.float32)
        lbls = np.random.choice([0, 1, 2, 3, 255], size=n_pts).astype(np.uint32)

        t0 = time.perf_counter()

        # 1. Load (in-memory)
        t_l0 = time.perf_counter()
        raw_frame = PointCloudFrame(points=pts, labels=lbls, frame_id="scaling_test")
        t_l1 = time.perf_counter()

        # 2. Preprocess
        filt_frame, _ = range_filter.filter_frame(raw_frame)
        t_p1 = time.perf_counter()

        # 3. ML Inference (Chunked for multi-million points to avoid OOM)
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
        t_m1 = time.perf_counter()

        # 4. Grid generation
        t_g0 = time.perf_counter()
        grid_map = adapter.prediction_to_grid(prediction)
        t_g1 = time.perf_counter()

        # 5. Vis prep
        t_v0 = time.perf_counter()
        _ = len(grid_map.cells)
        t_v1 = time.perf_counter()

        total_time_ms = (time.perf_counter() - t0) * 1000.0
        fps = 1000.0 / max(total_time_ms, 1e-4)
        mem_mb = process.memory_info().rss / (1024.0 * 1024.0)

        scaling_results.append({
            "points": n_pts,
            "total_runtime_ms": round(total_time_ms, 2),
            "load_ms": round((t_l1 - t_l0) * 1000.0, 2),
            "preprocess_ms": round((t_p1 - t_l1) * 1000.0, 2),
            "ml_inference_ms": round((t_m1 - t_m0) * 1000.0, 2),
            "grid_generation_ms": round((t_g1 - t_g0) * 1000.0, 2),
            "visualization_prep_ms": round((t_v1 - t_v0) * 1000.0, 2),
            "fps": round(fps, 2),
            "memory_mb": round(mem_mb, 2),
            "occupied_cells": grid_map.num_occupied_cells,
            "data_type": "Controlled Real/Synthetic Mixture"
        })

    return scaling_results


def generate_benchmark_plots(
    df_frames: pd.DataFrame,
    df_scaling: pd.DataFrame,
    output_dir: Path,
    model_name: str = "SPVCNN"
):
    """Generates all 6 diagnostic benchmark plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Latency Breakdown per Frame (Stacked Bar)
    fig, ax = plt.subplots(figsize=(10, 6))
    frames = df_frames["frame_id"].astype(str)
    x_pos = np.arange(len(frames))
    width = 0.55

    b_load = df_frames["load_ms"]
    b_prep = df_frames["preprocess_ms"]
    b_ml = df_frames["ml_inference_ms"]
    b_grid = df_frames["grid_generation_ms"]
    b_vis = df_frames["visualization_prep_ms"]

    ax.bar(x_pos, b_load, width, label="1. Load", color="#4285F4")
    ax.bar(x_pos, b_prep, width, bottom=b_load, label="2. Preprocess", color="#34A853")
    ax.bar(x_pos, b_ml, width, bottom=b_load + b_prep, label="3. ML Inference", color="#EA4335")
    ax.bar(x_pos, b_grid, width, bottom=b_load + b_prep + b_ml, label="4. Grid Gen", color="#FBBC05")
    ax.bar(x_pos, b_vis, width, bottom=b_load + b_prep + b_ml + b_grid, label="5. Vis Prep", color="#9C27B0")

    ax.set_title(f"Phase 3 Performance — Latency Breakdown ({model_name})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Frame ID")
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(frames)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_dir / "latency_breakdown.png", dpi=200)
    plt.close()

    # 2. FPS per Frame
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(frames, df_frames["fps"], marker="o", linewidth=2.5, color="#1E88E5", label="Frame FPS")
    ax.axhline(df_frames["fps"].mean(), color="crimson", linestyle="--", linewidth=1.8, label=f"Mean: {df_frames['fps'].mean():.2f} FPS")
    ax.set_title(f"Phase 3 Performance — End-to-End Throughput ({model_name})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Frame ID")
    ax.set_ylabel("Throughput (FPS = 1000 / Total ms)")
    ax.set_ylim(0, max(df_frames["fps"].max() * 1.3, 10))
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_dir / "fps.png", dpi=200)
    plt.close()

    # 3. Memory Usage (MB)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(frames, df_frames["memory_mb"], marker="s", linewidth=2.2, color="#00897B", label="Resident Memory (RSS)")
    ax.set_title(f"Phase 3 Performance — Process Memory Footprint ({model_name})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Frame ID")
    ax.set_ylabel("Memory (MB)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_dir / "memory.png", dpi=200)
    plt.close()

    # 4. CPU Usage (%)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(frames, df_frames["cpu_percent"], marker="^", linewidth=2.2, color="#FB8C00", label="CPU Utilization (%)")
    ax.set_title(f"Phase 3 Performance — CPU Utilization Across Frames ({model_name})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Frame ID")
    ax.set_ylabel("CPU Utilization (%)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_dir / "cpu.png", dpi=200)
    plt.close()

    # 5. Scaling: Points -> Runtime
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(df_scaling["points"], df_scaling["total_runtime_ms"], marker="o", linewidth=2.5, color="#D81B60", label="Total Pipeline Runtime")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"Phase 3 Scaling Benchmark — Input Points vs Runtime ({model_name})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Input Points (Log Scale)")
    ax.set_ylabel("Total Runtime ms (Log Scale)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "scaling_runtime.png", dpi=200)
    plt.close()

    # 6. Scaling: Points -> Memory
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(df_scaling["points"], df_scaling["memory_mb"], marker="D", linewidth=2.5, color="#5E35B1", label="Process RAM Footprint")
    ax.set_xscale("log")
    ax.set_title(f"Phase 3 Scaling Benchmark — Input Points vs Memory ({model_name})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Input Points (Log Scale)")
    ax.set_ylabel("Memory RSS (MB)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "scaling_memory.png", dpi=200)
    plt.close()


def generate_benchmark_report(
    env_meta: Dict[str, Any],
    stage_stats: Dict[str, Dict[str, float]],
    df_frames: pd.DataFrame,
    df_scaling: pd.DataFrame,
    output_dir: Path,
    model_name: str = "SPVCNN"
):
    """Generates human-readable Markdown report."""

    stage_table = [
        ["1. LiDAR Loading", stage_stats["load"]["mean"], stage_stats["load"]["median"], stage_stats["load"]["p95"], stage_stats["load"]["min"], stage_stats["load"]["max"], stage_stats["load"]["std"]],
        ["2. Preprocessing", stage_stats["preprocess"]["mean"], stage_stats["preprocess"]["median"], stage_stats["preprocess"]["p95"], stage_stats["preprocess"]["min"], stage_stats["preprocess"]["max"], stage_stats["preprocess"]["std"]],
        ["3. ML Inference", stage_stats["ml_inference"]["mean"], stage_stats["ml_inference"]["median"], stage_stats["ml_inference"]["p95"], stage_stats["ml_inference"]["min"], stage_stats["ml_inference"]["max"], stage_stats["ml_inference"]["std"]],
        ["4. Grid Generation", stage_stats["grid_generation"]["mean"], stage_stats["grid_generation"]["median"], stage_stats["grid_generation"]["p95"], stage_stats["grid_generation"]["min"], stage_stats["grid_generation"]["max"], stage_stats["grid_generation"]["std"]],
        ["5. Vis Preparation", stage_stats["visualization_prep"]["mean"], stage_stats["visualization_prep"]["median"], stage_stats["visualization_prep"]["p95"], stage_stats["visualization_prep"]["min"], stage_stats["visualization_prep"]["max"], stage_stats["visualization_prep"]["std"]],
        ["TOTAL END-TO-END", stage_stats["total"]["mean"], stage_stats["total"]["median"], stage_stats["total"]["p95"], stage_stats["total"]["min"], stage_stats["total"]["max"], stage_stats["total"]["std"]]
    ]

    scaling_table = [
        [f"{r['points']:,}", f"{r['total_runtime_ms']:.2f} ms", f"{r['load_ms']:.2f} ms", f"{r['preprocess_ms']:.2f} ms", f"{r['ml_inference_ms']:.2f} ms", f"{r['grid_generation_ms']:.2f} ms", f"{r['fps']:.2f}", f"{r['memory_mb']:.1f} MB"]
        for _, r in df_scaling.iterrows()
    ]

    report_md = f"""# Phase 3 — Performance Benchmark Report ({model_name})

**Objective**: Empirical performance profiling of the 5-stage perception pipeline with {model_name}.  
**Benchmark Date**: {env_meta['timestamp']}  
**Hardware & Environment**: {env_meta['os']} | CPU: {env_meta['processor']} ({env_meta['cpu_count_logical']} threads) | RAM: {env_meta['total_ram_gb']} GB  
**PyTorch Version**: {env_meta['torch_version']} | Python: {env_meta['python_version']}  
**Model Architecture**: {env_meta['model_name']} ({env_meta['model_parameters']:,} parameters)  

---

## 1. Stage-by-Stage Latency Profile (Milliseconds)

{tabulate(stage_table, headers=["Pipeline Stage", "Mean (ms)", "Median (ms)", "P95 (ms)", "Min (ms)", "Max (ms)", "Std Dev"], tablefmt="github")}

---

## 2. End-to-End System Summary Metrics

| System Metric | Measured Value |
| :--- | :--- |
| **Mean Input Points / Frame** | **{df_frames['input_points'].mean():,.0f} points** |
| **Mean 2.5D Cells / Frame** | **{df_frames['grid_cells'].mean():,.0f} cells** |
| **Mean End-to-End Latency** | **{stage_stats['total']['mean']:.2f} ms** |
| **Median End-to-End Latency** | **{stage_stats['total']['median']:.2f} ms** |
| **95th Percentile Latency (P95)**| **{stage_stats['total']['p95']:.2f} ms** |
| **End-to-End Throughput (FPS)** | **{df_frames['fps'].mean():.2f} FPS** |
| **Mean Process RAM (RSS)** | **{df_frames['memory_mb'].mean():.2f} MB** |
| **Mean CPU Utilization** | **{df_frames['cpu_percent'].mean():.1f}%** |

---

## 3. Scaling Benchmark Across Point Counts

{tabulate(scaling_table, headers=["Points (N)", "Total (ms)", "Load (ms)", "Prep (ms)", "ML (ms)", "Grid (ms)", "FPS", "RAM (MB)"], tablefmt="github")}
"""
    with open(output_dir / f"benchmark_report_{model_name.lower()}.md", "w") as f:
        f.write(report_md)


def main():
    parser = argparse.ArgumentParser(description="Phase 3 End-to-End Performance Benchmark")
    parser.add_argument("--input", type=str, default="data/semanticposs_sequence/sequences/01", help="Dataset directory")
    parser.add_argument("--frames", type=int, default=5, help="Number of frames to benchmark")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations")
    parser.add_argument("--output", type=str, default="results/spvcnn_benchmark", help="Output directory")
    parser.add_argument("--model-type", type=str, default="spvcnn", choices=["spvcnn", "foveated_pointnet"], help="Model architecture")
    parser.add_argument("--point-counts", type=str, default="10000,100000,500000,1000000,5000000", help="Comma-separated point counts")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    process = psutil.Process(os.getpid())

    # 1. Instantiate predictor
    if args.model_type == "spvcnn":
        predictor = Phase2Predictor(model_type="spvcnn", model_path="checkpoints/spvcnn_pretrained.pt", device="cpu")
        model_name = "SPVCNN"
        param_count = 136979
    else:
        predictor = Phase2Predictor(model_type="foveated_pointnet", model_path="checkpoints/best_model.pth", device="cpu")
        model_name = "FoveatedPointSegNet"
        param_count = 451460

    print("=" * 80)
    print(f"  PHASE 3: PERFORMANCE BENCHMARK WITH {model_name.upper()}")
    print("=" * 80)

    # Environment metadata
    env_meta = collect_environment_metadata(model_name=model_name, param_count=param_count)
    print(f"Environment: {env_meta['os']} | CPU: {env_meta['processor']} | RAM: {env_meta['total_ram_gb']} GB")
    print(f"Active Model: {model_name} ({param_count:,} params) on {env_meta['torch_version']}")

    # 2. Discover dataset files
    in_dir = Path(args.input)
    bin_files = sorted(in_dir.glob("velodyne/*.bin"))
    lbl_files = sorted(in_dir.glob("labels/*.label"))
    assert len(bin_files) > 0, f"No .bin files found in {in_dir}/velodyne"
    assert len(lbl_files) > 0, f"No .label files found in {in_dir}/labels"

    num_frames = min(args.frames, len(bin_files))
    bin_files = bin_files[:num_frames]
    lbl_files = lbl_files[:num_frames]

    range_filter = RangeFilter(min_range=0.0, max_range=100.0)
    adapter = MLToMappingAdapter(bands=DEFAULT_FROZEN_BANDS, max_range=100.0)

    # 3. Warm-up Phase
    print(f"\n[1/4] Warming up pipeline ({args.warmup} iterations)...")
    for w in range(args.warmup):
        _ = run_pipeline_single_frame(bin_files[0], lbl_files[0], range_filter, predictor, adapter, process)
    print("  -> Warm-up completed successfully.")

    # 4. Measure Multi-Frame Execution
    print(f"\n[2/4] Benchmarking {num_frames} real LiDAR frames...")
    frame_results = []
    for i in range(num_frames):
        rec = run_pipeline_single_frame(bin_files[i], lbl_files[i], range_filter, predictor, adapter, process)
        frame_results.append(rec)
        print(f"  Frame {rec['frame_id']}: {rec['input_points']:,} pts -> {rec['grid_cells']:,} cells | ML: {rec['ml_inference_ms']:.2f} ms | Total: {rec['total_ms']:.2f} ms ({rec['fps']:.2f} FPS)")

    df_frames = pd.DataFrame(frame_results)
    df_frames.to_csv(out_dir / "raw_results.csv", index=False)

    # 5. Compute Summary Statistics
    stage_stats = {
        "load": compute_stage_stats(df_frames["load_ms"].tolist()),
        "preprocess": compute_stage_stats(df_frames["preprocess_ms"].tolist()),
        "ml_inference": compute_stage_stats(df_frames["ml_inference_ms"].tolist()),
        "grid_generation": compute_stage_stats(df_frames["grid_generation_ms"].tolist()),
        "visualization_prep": compute_stage_stats(df_frames["visualization_prep_ms"].tolist()),
        "total": compute_stage_stats(df_frames["total_ms"].tolist())
    }

    # 6. Scaling Benchmark
    print("\n[3/4] Running Point Cloud Scaling Benchmark...")
    target_counts = [int(x.strip()) for x in args.point_counts.split(",") if x.strip()]
    scaling_records = run_scaling_benchmark(predictor, adapter, range_filter, process, target_counts)
    df_scaling = pd.DataFrame(scaling_records)

    # 7. Export JSON and Visualizations
    print("\n[4/4] Generating JSON Summary, Plots, and Markdown Report...")
    summary_data = {
        "environment": env_meta,
        "model_type": model_name,
        "stage_statistics": stage_stats,
        "frames_summary": {
            "mean_points": round(float(df_frames["input_points"].mean()), 1),
            "mean_cells": round(float(df_frames["grid_cells"].mean()), 1),
            "mean_total_ms": stage_stats["total"]["mean"],
            "median_total_ms": stage_stats["total"]["median"],
            "p95_total_ms": stage_stats["total"]["p95"],
            "mean_fps": round(float(df_frames["fps"].mean()), 2),
            "mean_memory_mb": round(float(df_frames["memory_mb"].mean()), 2),
            "mean_cpu_percent": round(float(df_frames["cpu_percent"].mean()), 1)
        },
        "scaling_results": scaling_records
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    generate_benchmark_plots(df_frames, df_scaling, out_dir, model_name=model_name)
    generate_benchmark_report(env_meta, stage_stats, df_frames, df_scaling, out_dir, model_name=model_name)

    # 8. Print Concise Baseline Tables
    print("\n" + "=" * 80)
    print(f"  PHASE 3 PERFORMANCE PROFILE ({model_name.upper()})")
    print("=" * 80)

    summary_table = [
        ["Load", f"{stage_stats['load']['mean']:.2f} ms", f"{stage_stats['load']['median']:.2f} ms", f"{stage_stats['load']['p95']:.2f} ms"],
        ["Preprocess", f"{stage_stats['preprocess']['mean']:.2f} ms", f"{stage_stats['preprocess']['median']:.2f} ms", f"{stage_stats['preprocess']['p95']:.2f} ms"],
        ["ML Inference", f"{stage_stats['ml_inference']['mean']:.2f} ms", f"{stage_stats['ml_inference']['median']:.2f} ms", f"{stage_stats['ml_inference']['p95']:.2f} ms"],
        ["Grid Generation", f"{stage_stats['grid_generation']['mean']:.2f} ms", f"{stage_stats['grid_generation']['median']:.2f} ms", f"{stage_stats['grid_generation']['p95']:.2f} ms"],
        ["Visualization", f"{stage_stats['visualization_prep']['mean']:.2f} ms", f"{stage_stats['visualization_prep']['median']:.2f} ms", f"{stage_stats['visualization_prep']['p95']:.2f} ms"],
        ["TOTAL", f"{stage_stats['total']['mean']:.2f} ms", f"{stage_stats['total']['median']:.2f} ms", f"{stage_stats['total']['p95']:.2f} ms"]
    ]
    print(tabulate(summary_table, headers=["Stage", "Mean ms", "Median ms", "P95 ms"], tablefmt="github"))

    print("\n" + "-" * 80)
    print(f"Points/frame:  {df_frames['input_points'].mean():,.0f}")
    print(f"Cells/frame:   {df_frames['grid_cells'].mean():,.0f}")
    print(f"Total latency: {stage_stats['total']['mean']:.2f} ms (Median: {stage_stats['total']['median']:.2f} ms, P95: {stage_stats['total']['p95']:.2f} ms)")
    print(f"FPS:           {df_frames['fps'].mean():.2f} FPS")
    print(f"RAM:           {df_frames['memory_mb'].mean():.2f} MB")
    print(f"CPU:           {df_frames['cpu_percent'].mean():.1f}%")
    print("-" * 80)


if __name__ == "__main__":
    main()
