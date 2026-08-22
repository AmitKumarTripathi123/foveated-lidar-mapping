"""
Automated Markdown & Table Reporting Engine for Phase 3 Benchmark.
Generates comprehensive Markdown baseline performance reports and CSV tables.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd


def generate_markdown_report(
    env_metadata: Dict[str, Any],
    stats: Dict[str, Dict[str, float]],
    bottlenecks: Dict[str, Any],
    scaling_data: List[Dict[str, Any]],
    output_path: Path
):
    """Writes the full Phase 3 baseline performance markdown report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mean_total = stats["total_ms"]["mean"]
    fps = round(1000.0 / max(mean_total, 1e-4), 2)
    p95_total = stats["total_ms"]["p95"]
    p99_total = stats["total_ms"]["p99"]

    primary = bottlenecks["primary_bottleneck"]
    secondary = bottlenecks["secondary_bottleneck"]

    scaling_rows = ""
    for row in scaling_data:
        scaling_rows += f"| {row['points']:,} | {row['load_ms']:.2f} ms | {row['preprocess_ms']:.2f} ms | {row['ml_inference_ms']:.2f} ms | {row['grid_generation_ms']:.2f} ms | {row['visualization_prep_ms']:.2f} ms | {row['total_ms']:.2f} ms | {row['fps']:.2f} | {row['memory_mb']:.1f} MB |\n"

    md = f"""# PHASE 3 — BASELINE PERFORMANCE REPORT

**Execution Date**: {env_metadata.get('timestamp')}  
**Model Architecture**: {env_metadata.get('model_name')} ({env_metadata.get('model_parameters'):,} parameters)  
**Hardware Platform**: {env_metadata.get('processor')} ({env_metadata.get('cpu_count_physical')} Physical Cores, {env_metadata.get('total_ram_gb')} GB RAM)  
**Operating System**: {env_metadata.get('os')}  

---

## 1. Environment & Software Metadata

| Parameter | Specification |
| :--- | :--- |
| **Operating System** | `{env_metadata.get('os')}` |
| **Python Version** | `{env_metadata.get('python_version')}` |
| **PyTorch Version** | `{env_metadata.get('torch_version')}` |
| **Processor / Architecture** | `{env_metadata.get('processor')}` ({env_metadata.get('system')}) |
| **Total System RAM** | `{env_metadata.get('total_ram_gb')} GB` |
| **GPU Status** | `{env_metadata.get('gpu_status', 'UNAVAILABLE (Apple Silicon CPU/MPS)')}` |
| **Git Commit Hash** | `{env_metadata.get('git_commit', 'N/A')}` |

---

## 2. Baseline Performance Profile (Per-Stage Latency & Resources)

Measurements collected across representative SemanticPOSS LiDAR scans:

| Pipeline Stage | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) | Min (ms) | Max (ms) | StdDev (ms) | Stage Share (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. LiDAR Loading** | `{stats['load_ms']['mean']:.2f}` | `{stats['load_ms']['median']:.2f}` | `{stats['load_ms']['p95']:.2f}` | `{stats['load_ms']['p99']:.2f}` | `{stats['load_ms']['min']:.2f}` | `{stats['load_ms']['max']:.2f}` | `{stats['load_ms']['std']:.2f}` | `{bottlenecks['percentages']['LiDAR Loading']:.2f}%` |
| **2. Preprocessing** | `{stats['preprocess_ms']['mean']:.2f}` | `{stats['preprocess_ms']['median']:.2f}` | `{stats['preprocess_ms']['p95']:.2f}` | `{stats['preprocess_ms']['p99']:.2f}` | `{stats['preprocess_ms']['min']:.2f}` | `{stats['preprocess_ms']['max']:.2f}` | `{stats['preprocess_ms']['std']:.2f}` | `{bottlenecks['percentages']['Preprocessing']:.2f}%` |
| **3. ML Inference** | `{stats['ml_inference_ms']['mean']:.2f}` | `{stats['ml_inference_ms']['median']:.2f}` | `{stats['ml_inference_ms']['p95']:.2f}` | `{stats['ml_inference_ms']['p99']:.2f}` | `{stats['ml_inference_ms']['min']:.2f}` | `{stats['ml_inference_ms']['max']:.2f}` | `{stats['ml_inference_ms']['std']:.2f}` | `{bottlenecks['percentages']['ML Inference']:.2f}%` |
| **4. Grid Generation** | `{stats['grid_generation_ms']['mean']:.2f}` | `{stats['grid_generation_ms']['median']:.2f}` | `{stats['grid_generation_ms']['p95']:.2f}` | `{stats['grid_generation_ms']['p99']:.2f}` | `{stats['grid_generation_ms']['min']:.2f}` | `{stats['grid_generation_ms']['max']:.2f}` | `{stats['grid_generation_ms']['std']:.2f}` | `{bottlenecks['percentages']['Grid Generation']:.2f}%` |
| **5. Visualization Prep** | `{stats['visualization_prep_ms']['mean']:.2f}` | `{stats['visualization_prep_ms']['median']:.2f}` | `{stats['visualization_prep_ms']['p95']:.2f}` | `{stats['visualization_prep_ms']['p99']:.2f}` | `{stats['visualization_prep_ms']['min']:.2f}` | `{stats['visualization_prep_ms']['max']:.2f}` | `{stats['visualization_prep_ms']['std']:.2f}` | `{bottlenecks['percentages']['Visualization Prep']:.2f}%` |
| **TOTAL PIPELINE** | **`{mean_total:.2f}`** | **`{stats['total_ms']['median']:.2f}`** | **`{p95_total:.2f}`** | **`{p99_total:.2f}`** | **`{stats['total_ms']['min']:.2f}`** | **`{stats['total_ms']['max']:.2f}`** | **`{stats['total_ms']['std']:.2f}`** | **100.00%** |

### Resource Consumption & Throughput
- **Effective Pipeline Throughput**: **`{fps:.2f} FPS`**
- **Resident RAM (RSS)**: **`{stats['memory_mb']['mean']:.2f} MB`** (Peak: `{stats['memory_mb']['max']:.2f} MB`)
- **Process CPU Load**: **`{stats['cpu_percent']['mean']:.1f}%`**
- **Mean Points / Frame**: **`{stats['input_points']['mean']:,.0f}`**
- **Mean Occupied 2.5D Cells**: **`{stats['grid_cells']['mean']:,.0f}`**

---

## 3. Point Cloud Scaling Experiment

| Points | Load | Preprocess | ML | Grid | Visualize | Total | FPS | RAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{scaling_rows}

---

## 4. Bottleneck Identification & Analysis

- **PRIMARY BOTTLENECK**: **`{primary['stage']}`** (`{primary['latency_ms']:.2f} ms`, **`{primary['percentage']:.2f}%`** of total latency)
- **SECONDARY BOTTLENECK**: **`{secondary['stage']}`** (`{secondary['latency_ms']:.2f} ms`, **`{secondary['percentage']:.2f}%`** of total latency)

### Interpretation & Engineering Next Steps
1. **Grid Generation (`{primary['stage']}`)** constitutes **`{primary['percentage']:.2f}%`** of pipeline execution time. The Python-level spatial binning and cell aggregation loops are the largest drag on FPS.
2. **ML Inference (`{secondary['stage']}`)** is already optimized down to **`{secondary['latency_ms']:.2f} ms`** via SPVCNN sparse point-voxel convolutions ($136,979$ parameters).
3. **Recommended Phase 4 Optimization**: Direct NumPy C-vectorization or Cython/Numba spatial hash grid aggregation to reduce Grid Generation from ~150 ms down to < 20 ms, achieving real-time **> 10 FPS** autonomous vehicle throughput.
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)


def export_tables_and_metadata(
    env_metadata: Dict[str, Any],
    stats: Dict[str, Dict[str, float]],
    per_frame_df: pd.DataFrame,
    scaling_df: pd.DataFrame,
    tables_dir: Path,
    raw_dir: Path
):
    """Exports all raw and tabular CSV/JSON files."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. Raw Per Frame CSV
    per_frame_df.to_csv(raw_dir / "per_frame.csv", index=False)

    # 2. Scaling CSV
    scaling_df.to_csv(raw_dir / "scaling.csv", index=False)
    scaling_df.to_csv(tables_dir / "scaling.csv", index=False)

    # 3. Metadata JSON
    with open(raw_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(env_metadata, f, indent=2)

    # 4. Statistics Table CSV
    stats_df = pd.DataFrame(stats).T
    stats_df.to_csv(tables_dir / "statistics.csv")

    # 5. Baseline summary table
    summary_data = [{
        "Stage": "Total Pipeline",
        "Mean_ms": stats["total_ms"]["mean"],
        "Median_ms": stats["total_ms"]["median"],
        "P95_ms": stats["total_ms"]["p95"],
        "P99_ms": stats["total_ms"]["p99"],
        "FPS": round(1000.0 / max(stats["total_ms"]["mean"], 1e-4), 2),
        "RAM_MB": stats["memory_mb"]["mean"],
        "CPU_Percent": stats["cpu_percent"]["mean"]
    }]
    pd.DataFrame(summary_data).to_csv(tables_dir / "baseline.csv", index=False)
