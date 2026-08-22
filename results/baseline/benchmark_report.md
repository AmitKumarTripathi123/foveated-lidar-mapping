# Phase 3 — Baseline Performance Benchmark Report

**Objective**: Establish reproducible empirical performance baselines for the unoptimized Python pipeline before Phase 3 acceleration.  
**Benchmark Date**: 2026-08-22 16:40:11  
**Hardware & Environment**: macOS-26.5.2-arm64-arm-64bit | CPU: arm (10 threads) | RAM: 16.0 GB  
**PyTorch Version**: 2.8.0 | Python: 3.9.6  

---

## 1. Stage-by-Stage Latency Profile (Milliseconds)

| Pipeline Stage     |   Mean (ms) |   Median (ms) |   P95 (ms) |   Min (ms) |   Max (ms) |   Std Dev |
|--------------------|-------------|---------------|------------|------------|------------|-----------|
| 1. LiDAR Loading   |       1.048 |         1.211 |      1.275 |      0.352 |      1.282 |     0.351 |
| 2. Preprocessing   |       2.84  |         2.865 |      2.887 |      2.708 |      2.89  |     0.067 |
| 3. ML Inference    |     254.06  |       266.057 |    297.885 |    190.088 |    303.329 |    38.792 |
| 4. Grid Generation |     185.808 |       195.661 |    199.171 |    143.26  |    199.352 |    21.415 |
| 5. Vis Preparation |      98.114 |        97.951 |     99.463 |     96.614 |     99.552 |     1.088 |
| TOTAL END-TO-END   |     541.871 |       566.627 |    596.401 |    433.921 |    602.652 |    58.108 |

---

## 2. End-to-End System Summary Metrics

| System Metric | Baseline Measured Value |
| :--- | :--- |
| **Mean Input Points / Frame** | **40,000 points** |
| **Mean 2.5D Cells / Frame** | **28,308 cells** |
| **Mean End-to-End Latency** | **541.87 ms** |
| **Median End-to-End Latency** | **566.63 ms** |
| **95th Percentile Latency (P95)**| **596.40 ms** |
| **End-to-End Throughput (FPS)** | **1.87 FPS** |
| **Mean Process RAM (RSS)** | **696.82 MB** |
| **Mean CPU Utilization** | **174.3%** |

---

## 3. Bottleneck Analysis for Phase 3 Optimization

1. **Primary Computational Bottleneck**:
   - **Grid Generation (2.5D XY Cell Indexing & Aggregation)**: Accounts for **34.3%** of total latency (~185.8 ms).
2. **Secondary Bottleneck**:
   - **ML Inference (FoveatedPointSegNet)**: Accounts for **46.9%** (~254.1 ms on CPU).
3. **Low-Overhead Stages**:
   - LiDAR Loading (1.05 ms) and Preprocessing (2.84 ms) account for $< 3\%$ of compute time.

---

## 4. Scaling Benchmark Across Point Counts

| Points (N)   | Total (ms)   | Load (ms)   | Prep (ms)   | ML (ms)     | Grid (ms)   |   FPS | RAM (MB)   |
|--------------|--------------|-------------|-------------|-------------|-------------|-------|------------|
| 10,000       | 81.85 ms     | 0.02 ms     | 0.54 ms     | 43.00 ms    | 38.30 ms    | 12.22 | 712.6 MB   |
| 100,000      | 1195.21 ms   | 0.02 ms     | 4.54 ms     | 680.46 ms   | 510.19 ms   |  0.84 | 1522.5 MB  |
| 500,000      | 5027.05 ms   | 0.08 ms     | 25.27 ms    | 3118.86 ms  | 1882.83 ms  |  0.2  | 1675.3 MB  |
| 1,000,000    | 8791.34 ms   | 0.16 ms     | 46.75 ms    | 5946.12 ms  | 2798.32 ms  |  0.11 | 1797.7 MB  |
| 5,000,000    | 41276.69 ms  | 0.42 ms     | 246.85 ms   | 31269.33 ms | 9760.08 ms  |  0.02 | 2172.2 MB  |

---

## 5. Diagnostic Visualization Artifacts

The following performance diagnostic plots have been exported to `results/baseline/`:
- `latency_breakdown.png`: Stacked stage latencies per frame.
- `fps.png`: End-to-end frame rate per second.
- `memory.png`: Resident memory footprint per frame.
- `cpu.png`: CPU thread utilization percentage.
- `scaling_runtime.png`: Computational scalability across $10	ext{K} 	o 5	ext{M}$ points.
- `scaling_memory.png`: Memory allocation scaling curve.
