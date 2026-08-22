"""
Phase 3 Master End-to-End Pipeline Performance Benchmark Harness.
Measures real LiDAR processing latency, resource consumption, scaling, and bottleneck profile.
"""

import os
import sys
import time
import argparse
import subprocess
import platform
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import psutil
import numpy as np
import pandas as pd
import torch

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.types import PointCloudFrame
from src.range_filter import RangeFilter
from phase2.dataset import remap_poss_labels
from phase2.inference.predictor import Phase2Predictor, SemanticPrediction
from phase2.adapter import MLToMappingAdapter
from benchmarks.phase3.system_monitor import SystemResourceMonitor
from benchmarks.phase3.metrics import compute_stage_statistics, analyze_pipeline_bottlenecks
from benchmarks.phase3.plotting import generate_all_plots
from benchmarks.phase3.report import generate_markdown_report, export_tables_and_metadata
from benchmarks.phase3.benchmark_scaling import run_scaling_experiment


def get_git_commit_hash() -> str:
    """Retrieves current Git commit hash."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


def collect_full_environment_metadata(model_type: str, checkpoint_path: str) -> Dict[str, Any]:
    """Captures complete environment specification for strict reproducibility."""
    import yaml
    import scipy
    import matplotlib

    cuda_avail = torch.cuda.is_available()
    gpu_status = f"AVAILABLE ({torch.cuda.get_device_name(0)})" if cuda_avail else "UNAVAILABLE (Apple Silicon / CPU)"

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": get_git_commit_hash(),
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
        "gpu_status": gpu_status,
        "model_name": "SPVCNN" if model_type == "spvcnn" else "FoveatedPointSegNet",
        "model_parameters": 136979 if model_type == "spvcnn" else 451460,
        "checkpoint_path": checkpoint_path
    }


def find_dataset_frames(dataset_root: Path, max_frames: int) -> List[Tuple[Path, Path]]:
    """Discovers real .bin and .label pairs using filename stem matching."""
    pairs = []
    seq_root = dataset_root / "sequences" if (dataset_root / "sequences").exists() else dataset_root
    for seq_dir in sorted(seq_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        v_dir = seq_dir / "velodyne"
        l_dir = seq_dir / "labels"
        if not (v_dir.exists() and l_dir.exists()):
            continue
        bin_files = sorted(v_dir.glob("*.bin"))
        for bp in bin_files:
            lp = l_dir / f"{bp.stem}.label"
            if lp.exists():
                pairs.append((bp, lp))
                if len(pairs) >= max_frames:
                    return pairs
    return pairs


def benchmark_single_frame(
    bin_path: Path,
    lbl_path: Path,
    range_filter: RangeFilter,
    predictor: Phase2Predictor,
    adapter: MLToMappingAdapter,
    monitor: SystemResourceMonitor
) -> Dict[str, Any]:
    """
    Executes and times the 5 distinct pipeline stages on a single real frame.
    """
    f_id = bin_path.stem

    # 1. LiDAR Loading
    t0 = time.perf_counter()
    raw_pts = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)
    raw_lbls = np.fromfile(str(lbl_path), dtype=np.uint32)
    n_p = min(len(raw_pts), len(raw_lbls))
    raw_pts = raw_pts[:n_p]
    raw_lbls = raw_lbls[:n_p]
    t1 = time.perf_counter()
    load_ms = (t1 - t0) * 1000.0

    # 2. Preprocessing & Range Filtering
    t2 = time.perf_counter()
    mapped_lbls = remap_poss_labels(raw_lbls)
    raw_frame = PointCloudFrame(points=raw_pts, labels=mapped_lbls.astype(np.uint32), frame_id=f_id)
    filtered_frame, _ = range_filter.filter_frame(raw_frame)
    t3 = time.perf_counter()
    preprocess_ms = (t3 - t2) * 1000.0

    # 3. ML Inference
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t4 = time.perf_counter()
    prediction = predictor.predict_frame(filtered_frame)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t5 = time.perf_counter()
    ml_inference_ms = (t5 - t4) * 1000.0

    # 4. Grid Generation
    t6 = time.perf_counter()
    grid_map = adapter.prediction_to_grid(prediction)
    t7 = time.perf_counter()
    grid_generation_ms = (t7 - t6) * 1000.0

    # 5. Visualization Preparation
    t8 = time.perf_counter()
    _ = grid_map.num_occupied_cells
    t9 = time.perf_counter()
    vis_ms = (t9 - t8) * 1000.0

    total_ms = load_ms + preprocess_ms + ml_inference_ms + grid_generation_ms + vis_ms
    fps = 1000.0 / max(total_ms, 1e-4)
    snap = monitor.snapshot()

    return {
        "frame_id": f_id,
        "input_points": len(raw_pts),
        "filtered_points": len(filtered_frame.points),
        "grid_cells": grid_map.num_occupied_cells,
        "load_ms": round(load_ms, 3),
        "preprocess_ms": round(preprocess_ms, 3),
        "ml_inference_ms": round(ml_inference_ms, 3),
        "grid_generation_ms": round(grid_generation_ms, 3),
        "visualization_prep_ms": round(vis_ms, 3),
        "total_ms": round(total_ms, 3),
        "fps": round(fps, 2),
        "memory_mb": snap["ram_mb"],
        "cpu_percent": snap["cpu_percent"]
    }


def run_phase3_benchmark(
    dataset_root: str = "dataset",
    warmup_frames: int = 10,
    measured_frames: int = 100,
    model_type: str = "spvcnn",
    checkpoint: Optional[str] = None,
    device: str = "cpu",
    output_dir: str = "benchmark_results/phase3"
) -> Dict[str, Any]:
    """
    Master Phase 3 Benchmark Orchestrator.
    """
    out_path = Path(output_dir)
    raw_dir = out_path / "raw"
    tables_dir = out_path / "tables"
    plots_dir = out_path / "plots"
    report_dir = out_path / "report"

    for p in [raw_dir, tables_dir, plots_dir, report_dir]:
        p.mkdir(parents=True, exist_ok=True)

    if checkpoint is None:
        checkpoint = "checkpoints/best_spvcnn.pt" if model_type == "spvcnn" else "checkpoints/best_model.pth"

    env_metadata = collect_full_environment_metadata(model_type, checkpoint)
    env_metadata["warmup_frames"] = warmup_frames
    env_metadata["measured_frames"] = measured_frames

    print("=" * 80)
    print(f"  PHASE 3: COMPLETE BASELINE PERFORMANCE BENCHMARK ({model_type.upper()})")
    print("=" * 80)
    print(f"Environment: {env_metadata['os']} | CPU: {env_metadata['processor']} | RAM: {env_metadata['total_ram_gb']} GB")
    print(f"Active Model: {env_metadata['model_name']} ({env_metadata['model_parameters']:,} params)")
    print(f"Device: {device} | Checkpoint: {checkpoint}")

    # Discover Dataset
    total_needed = warmup_frames + measured_frames
    frame_pairs = find_dataset_frames(Path(dataset_root), total_needed)
    if len(frame_pairs) == 0:
        raise FileNotFoundError(f"No valid .bin/.label pairs found in {dataset_root}!")

    print(f"\n[1/4] Discovered {len(frame_pairs)} real LiDAR frames.")

    # Initialize Components
    monitor = SystemResourceMonitor()
    range_filter = RangeFilter(max_range=100.0, min_range=0.5)
    predictor = Phase2Predictor(model_type=model_type, model_path=checkpoint, device=device)
    adapter = MLToMappingAdapter()

    # 1. Warm-up
    warmup_count = min(warmup_frames, len(frame_pairs))
    print(f"[2/4] Warming up pipeline ({warmup_count} iterations)...")
    for i in range(warmup_count):
        bp, lp = frame_pairs[i % len(frame_pairs)]
        _ = benchmark_single_frame(bp, lp, range_filter, predictor, adapter, monitor)
    print("  -> Warm-up completed successfully.")

    # 2. Measurement Run
    measure_count = min(measured_frames, len(frame_pairs))
    print(f"\n[3/4] Collecting real measurements across {measure_count} frames...")
    records = []
    for i in range(measure_count):
        bp, lp = frame_pairs[i % len(frame_pairs)]
        rec = benchmark_single_frame(bp, lp, range_filter, predictor, adapter, monitor)
        records.append(rec)
        if (i + 1) % 10 == 0 or (i + 1) == measure_count:
            print(f"  Frame [{i+1:3d}/{measure_count:3d}] ({100*(i+1)/measure_count:5.1f}%) | Latency: {rec['total_ms']:.2f} ms | FPS: {rec['fps']:.2f}")

    per_frame_df = pd.DataFrame(records)

    # 3. Statistical Calculations
    keys = ["load_ms", "preprocess_ms", "ml_inference_ms", "grid_generation_ms", "visualization_prep_ms", "total_ms", "memory_mb", "cpu_percent", "input_points", "grid_cells"]
    stats = {}
    for k in keys:
        stats[k] = compute_stage_statistics(per_frame_df[k].tolist())

    mean_stages = {k: stats[k]["mean"] for k in ["load_ms", "preprocess_ms", "ml_inference_ms", "grid_generation_ms", "visualization_prep_ms", "total_ms"]}
    bottlenecks = analyze_pipeline_bottlenecks(mean_stages)

    # 4. Scaling Benchmark
    print("\n[4/4] Executing Point Cloud Scaling Experiment (10K, 100K, 500K, 1M, 5M)...")
    scaling_data = run_scaling_experiment(
        target_counts=[10000, 100000, 500000, 1000000, 5000000],
        model_type=model_type,
        checkpoint_path=checkpoint,
        device=device
    )
    scaling_df = pd.DataFrame(scaling_data)

    # 5. Generate Plots & Reports
    print("\nGenerating Phase 3 Plots, Tables, and Markdown Report...")
    generate_all_plots(stats, scaling_data, plots_dir)
    generate_markdown_report(env_metadata, stats, bottlenecks, scaling_data, report_dir / "phase3_baseline_report.md")
    export_tables_and_metadata(env_metadata, stats, per_frame_df, scaling_df, tables_dir, raw_dir)

    print("=" * 80)
    print("  PHASE 3 BENCHMARK EXECUTION COMPLETED")
    print("=" * 80)
    print(f"Total Pipeline Mean Latency: {stats['total_ms']['mean']:.2f} ms (P95: {stats['total_ms']['p95']:.2f} ms, P99: {stats['total_ms']['p99']:.2f} ms)")
    print(f"Effective Throughput:        {1000.0/max(stats['total_ms']['mean'], 1e-4):.2f} FPS")
    print(f"Resident Memory (RSS):       {stats['memory_mb']['mean']:.2f} MB (Peak: {stats['memory_mb']['max']:.2f} MB)")
    print(f"CPU Utilization:             {stats['cpu_percent']['mean']:.1f}%")
    print(f"PRIMARY BOTTLENECK:          {bottlenecks['primary_bottleneck']['stage']} ({bottlenecks['primary_bottleneck']['percentage']:.2f}%)")
    print(f"SECONDARY BOTTLENECK:        {bottlenecks['secondary_bottleneck']['stage']} ({bottlenecks['secondary_bottleneck']['percentage']:.2f}%)")
    print("=" * 80)

    return {
        "stats": stats,
        "bottlenecks": bottlenecks,
        "scaling_data": scaling_data,
        "report_path": str(report_dir / "phase3_baseline_report.md")
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Baseline Performance Benchmark Harness")
    parser.add_argument("--dataset", type=str, default="dataset")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--model-type", type=str, default="spvcnn", choices=["spvcnn", "foveated_pointnet"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="benchmark_results/phase3")
    args = parser.parse_args()

    run_phase3_benchmark(
        dataset_root=args.dataset,
        warmup_frames=args.warmup,
        measured_frames=args.frames,
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        device=args.device,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
