# Phase 3 — End-to-End Performance Benchmark Suite

This directory contains the reproducible benchmarking suite for the **Foveated 2.5D LiDAR Mapping System for Autonomous Navigation** (Smart India Hackathon).

---

## 1. Overview

The benchmark establishes the unoptimized baseline performance of the complete 5-stage Python perception pipeline:
1. **LiDAR Loading**: Binary deserialization of raw point clouds (`float32[N, 4]`) and semantic labels (`uint32[N]`).
2. **Preprocessing**: 100m radial filtering ($r = \sqrt{x^2+y^2} \le 100.0\text{ m}$) and authoritative super-class label remapping.
3. **ML Semantic Inference**: `FoveatedPointSegNet` neural inference producing point-wise class probabilities and confidence scores.
4. **2.5D Foveated Grid Generation**: `MLToMappingAdapter` converting point predictions to multi-layer `GridMap25D` cells across the 4 distance bands.
5. **Visualization Preparation**: 2.5D layer tensor extractions, bounding box projections, and color mapping.

---

## 2. Usage & CLI Options

Run the benchmark with default settings (5 SemanticPOSS frames, 3 warm-up runs):
```bash
python benchmarks/run_benchmark.py
```

### Custom Options:
```bash
python benchmarks/run_benchmark.py \
    --input data/semanticposs_sequence/sequences/01 \
    --frames 5 \
    --warmup 3 \
    --output results/baseline \
    --point-counts 10000,100000,500000,1000000,5000000
```

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--input` | `str` | `data/semanticposs_sequence/sequences/01` | Path to dataset directory |
| `--frames` | `int` | `5` | Number of sequential frames to measure |
| `--warmup` | `int` | `3` | Number of warm-up runs before measurement |
| `--output` | `str` | `results/baseline` | Destination directory for output artifacts |
| `--point-counts` | `str` | `10000,100000,500000,1000000,5000000` | Comma-separated point scales for scaling test |

---

## 3. Output Artifacts

Benchmark results are automatically saved to `results/baseline/`:
- `raw_results.csv`: Frame-by-frame measurement data.
- `summary.json`: Machine-readable summary statistics and environment metadata.
- `benchmark_report.md`: Human-readable markdown report with summary tables and bottleneck analysis.
- `latency_breakdown.png`: Stacked bar chart showing per-stage latency breakdown.
- `fps.png`: End-to-end frame rate per second.
- `memory.png`: Process memory footprint (Resident Set Size).
- `cpu.png`: Process CPU utilization percentage.
- `scaling_runtime.png`: Computational scaling curve from 10K to 5M points.
- `scaling_memory.png`: Memory allocation scaling curve.
