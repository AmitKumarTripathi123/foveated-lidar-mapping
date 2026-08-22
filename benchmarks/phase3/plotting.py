"""
High-Resolution Plotting Engine for Phase 3 Benchmark.
Generates 5 standalone, non-interactive PNG charts:
  1. points_vs_runtime.png
  2. points_vs_memory.png
  3. pipeline_latency.png
  4. points_vs_ml.png
  5. points_vs_grid.png
"""

from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_all_plots(
    baseline_stats: Dict[str, Dict[str, float]],
    scaling_data: List[Dict[str, Any]],
    output_dir: Path
):
    """Generates all 5 required Phase 3 benchmark plots and saves them as PNGs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # --------------------------------------------------------------------------
    # GRAPH 1: Points -> Total Runtime
    # --------------------------------------------------------------------------
    if scaling_data:
        pts = [d["points"] for d in scaling_data]
        total_ms = [d["total_ms"] for d in scaling_data]

        plt.figure(figsize=(9, 5), dpi=300)
        plt.plot(pts, total_ms, marker="o", linewidth=2.5, color="#1f77b4", label="Total Pipeline Latency")
        plt.xscale("log")
        plt.xlabel("Point Cloud Size (Points, Log Scale)", fontsize=11, fontweight="bold")
        plt.ylabel("Total Latency (ms)", fontsize=11, fontweight="bold")
        plt.title("Graph 1: Point Cloud Scaling vs Total Pipeline Latency", fontsize=13, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_dir / "points_vs_runtime.png")
        plt.close()

    # --------------------------------------------------------------------------
    # GRAPH 2: Points -> Memory Usage
    # --------------------------------------------------------------------------
    if scaling_data:
        pts = [d["points"] for d in scaling_data]
        mem_mb = [d["memory_mb"] for d in scaling_data]

        plt.figure(figsize=(9, 5), dpi=300)
        plt.plot(pts, mem_mb, marker="s", linewidth=2.5, color="#2ca02c", label="Resident Memory (RSS)")
        plt.xscale("log")
        plt.xlabel("Point Cloud Size (Points, Log Scale)", fontsize=11, fontweight="bold")
        plt.ylabel("Memory (MB)", fontsize=11, fontweight="bold")
        plt.title("Graph 2: Point Cloud Scaling vs Resident Memory Usage", fontsize=13, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_dir / "points_vs_memory.png")
        plt.close()

    # --------------------------------------------------------------------------
    # GRAPH 3: Pipeline Component -> Latency Breakdown
    # --------------------------------------------------------------------------
    stages = ["LiDAR Loading", "Preprocessing", "ML Inference", "Grid Generation", "Visualization Prep"]
    keys = ["load_ms", "preprocess_ms", "ml_inference_ms", "grid_generation_ms", "visualization_prep_ms"]
    means = [baseline_stats[k]["mean"] for k in keys]
    p95s = [baseline_stats[k]["p95"] for k in keys]
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]

    plt.figure(figsize=(10, 5), dpi=300)
    x = np.arange(len(stages))
    width = 0.35

    plt.bar(x - width/2, means, width, label="Mean Latency (ms)", color="#3b6978")
    plt.bar(x + width/2, p95s, width, label="P95 Latency (ms)", color="#ff6b6b")

    for i in range(len(stages)):
        plt.text(i - width/2, means[i] + 1.5, f"{means[i]:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        plt.text(i + width/2, p95s[i] + 1.5, f"{p95s[i]:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.xticks(x, stages, fontsize=10, fontweight="bold")
    plt.ylabel("Latency (ms)", fontsize=11, fontweight="bold")
    plt.title("Graph 3: Pipeline Component Latency Breakdown (Mean vs P95)", fontsize=13, fontweight="bold")
    plt.legend(frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "pipeline_latency.png")
    plt.close()

    # --------------------------------------------------------------------------
    # GRAPH 4: Points -> ML Runtime
    # --------------------------------------------------------------------------
    if scaling_data:
        pts = [d["points"] for d in scaling_data]
        ml_ms = [d["ml_inference_ms"] for d in scaling_data]

        plt.figure(figsize=(9, 5), dpi=300)
        plt.plot(pts, ml_ms, marker="^", linewidth=2.5, color="#e15759", label="SPVCNN Inference")
        plt.xscale("log")
        plt.xlabel("Point Cloud Size (Points, Log Scale)", fontsize=11, fontweight="bold")
        plt.ylabel("ML Latency (ms)", fontsize=11, fontweight="bold")
        plt.title("Graph 4: Point Cloud Scaling vs ML Inference Runtime", fontsize=13, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_dir / "points_vs_ml.png")
        plt.close()

    # --------------------------------------------------------------------------
    # GRAPH 5: Points -> Grid Runtime
    # --------------------------------------------------------------------------
    if scaling_data:
        pts = [d["points"] for d in scaling_data]
        grid_ms = [d["grid_generation_ms"] for d in scaling_data]

        plt.figure(figsize=(9, 5), dpi=300)
        plt.plot(pts, grid_ms, marker="D", linewidth=2.5, color="#76b7b2", label="Grid Generation")
        plt.xscale("log")
        plt.xlabel("Point Cloud Size (Points, Log Scale)", fontsize=11, fontweight="bold")
        plt.ylabel("Grid Generation Latency (ms)", fontsize=11, fontweight="bold")
        plt.title("Graph 5: Point Cloud Scaling vs 2.5D Grid Generation Runtime", fontsize=13, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_dir / "points_vs_grid.png")
        plt.close()
