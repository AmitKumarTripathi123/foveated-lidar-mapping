# Phase 3: Complete Baseline Performance Benchmark

Comprehensive, reproducible baseline performance profiling harness for the **Foveated 2.5D LiDAR Mapping System**.

---

## 1. Pipeline Stages & Timing Boundaries

Monitored using high-resolution monotonic clocks (`time.perf_counter()`):

```text
LiDAR Input
    ↓ (T0)
1. LiDAR Loading (Load = T1 - T0)
    ↓ (T1)
2. Preprocessing & Range Filtering (Preprocess = T2 - T1)
    ↓ (T2)
3. ML / SPVCNN Inference (ML = T3 - T2)
    ↓ (T3)
4. 2.5D Grid Generation (Grid = T4 - T3)
    ↓ (T4)
5. Visualization Preparation (Visualize = T5 - T4)
    ↓ (T5)
TOTAL LATENCY = T5 - T0 ≈ Load + Preprocess + ML + Grid + Visualize
THROUGHPUT = 1000 / Total_Latency_ms
```

---

## 2. Directory Structure

```text
benchmarks/phase3/
├── __init__.py
├── system_monitor.py      # Process-level CPU, RAM, and GPU memory tracking
├── metrics.py             # Mean, Median, P95, P99, StdDev, and bottleneck ranking
├── plotting.py            # High-resolution PNG generator (5 required charts)
├── report.py              # Automated Markdown & CSV reporting engine
├── benchmark_scaling.py   # Standalone point-cloud scaling benchmark (10K to 5M)
├── benchmark_pipeline.py  # Master pipeline benchmark harness
└── README.md              # Documentation & reproduction guide
```

---

## 3. How to Run

### Complete Benchmark (Warm-up + Measured Scans + Scaling + Reports + Plots)
```bash
python benchmarks/phase3/benchmark_pipeline.py --warmup 10 --frames 100 --model-type spvcnn
```

### Compare with Baseline Model
```bash
python benchmarks/phase3/benchmark_pipeline.py --warmup 10 --frames 100 --model-type foveated_pointnet --output benchmark_results/phase3_baseline
```

### Standalone Scaling Benchmark (10K, 100K, 500K, 1M, 5M)
```bash
python benchmarks/phase3/benchmark_scaling.py --points 10000 100000 500000 1000000 5000000
```

---

## 4. Output Artifacts

All outputs are saved to `benchmark_results/phase3/`:
- `raw/per_frame.csv`: Frame-by-frame latency, points, cells, RAM, and CPU.
- `raw/scaling.csv`: Point cloud scaling experiment data.
- `raw/metadata.json`: Exact hardware, OS, Git commit, and package versions.
- `tables/baseline.csv`: Summary baseline metrics.
- `tables/statistics.csv`: Mean, median, P95, P99, min, max, std across all stages.
- `plots/points_vs_runtime.png`: Point cloud size vs total latency.
- `plots/points_vs_memory.png`: Point cloud size vs memory usage.
- `plots/pipeline_latency.png`: Stage breakdown bar chart.
- `plots/points_vs_ml.png`: Point cloud size vs ML latency.
- `plots/points_vs_grid.png`: Point cloud size vs Grid generation latency.
- `report/phase3_baseline_report.md`: Full scientific Markdown report.
