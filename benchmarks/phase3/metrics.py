"""
Statistical Metrics & Bottleneck Analysis Engine for Phase 3 Benchmark.
Calculates Mean, Median, P95, P99, Min, Max, StdDev, and automated bottleneck ranking.
"""

from typing import Dict, List, Any, Tuple
import numpy as np


def compute_stage_statistics(series: List[float]) -> Dict[str, float]:
    """
    Computes complete statistical profile for a timing series in milliseconds.
    """
    arr = np.array(series, dtype=np.float64)
    if len(arr) == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0
        }
    return {
        "mean": round(float(np.mean(arr)), 3),
        "median": round(float(np.median(arr)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
        "p99": round(float(np.percentile(arr, 99)), 3),
        "min": round(float(np.min(arr)), 3),
        "max": round(float(np.max(arr)), 3),
        "std": round(float(np.std(arr)), 3)
    }


def analyze_pipeline_bottlenecks(stage_means: Dict[str, float]) -> Dict[str, Any]:
    """
    Computes percentage contributions for each stage and automatically identifies
    the primary and secondary bottlenecks based on empirical latency.
    """
    total_ms = stage_means.get("total_ms", sum(v for k, v in stage_means.items() if k != "total_ms"))
    total_ms = max(total_ms, 1e-4)

    stages = {
        "LiDAR Loading": stage_means.get("load_ms", 0.0),
        "Preprocessing": stage_means.get("preprocess_ms", 0.0),
        "ML Inference": stage_means.get("ml_inference_ms", 0.0),
        "Grid Generation": stage_means.get("grid_generation_ms", 0.0),
        "Visualization Prep": stage_means.get("visualization_prep_ms", 0.0)
    }

    percentages = {}
    for name, val in stages.items():
        percentages[name] = round((val / total_ms) * 100.0, 2)

    # Sort stages by latency descending
    sorted_stages = sorted(stages.items(), key=lambda x: x[1], reverse=True)
    primary = sorted_stages[0]
    secondary = sorted_stages[1] if len(sorted_stages) > 1 else ("None", 0.0)

    return {
        "percentages": percentages,
        "primary_bottleneck": {
            "stage": primary[0],
            "latency_ms": round(primary[1], 2),
            "percentage": percentages[primary[0]]
        },
        "secondary_bottleneck": {
            "stage": secondary[0],
            "latency_ms": round(secondary[1], 2),
            "percentage": percentages[secondary[0]]
        }
    }
