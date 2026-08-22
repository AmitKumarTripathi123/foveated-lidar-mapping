# Phase 3 — Performance Benchmark Report (SPVCNN)

**Objective**: Empirical performance profiling of the 5-stage perception pipeline with SPVCNN.  
**Benchmark Date**: 2026-08-22 23:05:16  
**Hardware & Environment**: macOS-26.5.2-arm64-arm-64bit | CPU: arm (10 threads) | RAM: 16.0 GB  
**PyTorch Version**: 2.8.0 | Python: 3.9.6  
**Model Architecture**: SPVCNN (136,979 parameters)  

---

## 1. Stage-by-Stage Latency Profile (Milliseconds)

| Pipeline Stage     |   Mean (ms) |   Median (ms) |   P95 (ms) |   Min (ms) |   Max (ms) |   Std Dev |
|--------------------|-------------|---------------|------------|------------|------------|-----------|
| 1. LiDAR Loading   |       1.672 |         2.13  |      2.22  |      0.359 |      2.23  |     0.714 |
| 2. Preprocessing   |       2.669 |         2.65  |      2.777 |      2.61  |      2.807 |     0.071 |
| 3. ML Inference    |      97.373 |        97.176 |    103.733 |     91.434 |    105.149 |     4.51  |
| 4. Grid Generation |     156.565 |       166.376 |    167.376 |    120.243 |    167.553 |    18.263 |
| 5. Vis Preparation |      93.227 |        93.148 |     93.997 |     92.62  |     94.124 |     0.543 |
| TOTAL END-TO-END   |     351.506 |       359.322 |    368.692 |    314.473 |    370.245 |    19.516 |

---

## 2. End-to-End System Summary Metrics

| System Metric | Measured Value |
| :--- | :--- |
| **Mean Input Points / Frame** | **40,000 points** |
| **Mean 2.5D Cells / Frame** | **28,308 cells** |
| **Mean End-to-End Latency** | **351.51 ms** |
| **Median End-to-End Latency** | **359.32 ms** |
| **95th Percentile Latency (P95)**| **368.69 ms** |
| **End-to-End Throughput (FPS)** | **2.85 FPS** |
| **Mean Process RAM (RSS)** | **552.76 MB** |
| **Mean CPU Utilization** | **140.8%** |

---

## 3. Scaling Benchmark Across Point Counts

| Points (N)   | Total (ms)   | Load (ms)   | Prep (ms)   | ML (ms)     | Grid (ms)   |   FPS | RAM (MB)   |
|--------------|--------------|-------------|-------------|-------------|-------------|-------|------------|
| 10,000       | 59.27 ms     | 0.01 ms     | 0.53 ms     | 21.42 ms    | 37.30 ms    | 16.87 | 561.8 MB   |
| 100,000      | 780.64 ms    | 0.02 ms     | 4.51 ms     | 291.57 ms   | 484.54 ms   |  1.28 | 1049.8 MB  |
| 500,000      | 3318.90 ms   | 0.06 ms     | 24.92 ms    | 1389.49 ms  | 1904.43 ms  |  0.3  | 1208.8 MB  |
| 1,000,000    | 6038.98 ms   | 0.12 ms     | 46.34 ms    | 3164.55 ms  | 2827.97 ms  |  0.17 | 1321.6 MB  |
| 5,000,000    | 24853.43 ms  | 2.18 ms     | 302.07 ms   | 14732.28 ms | 9816.89 ms  |  0.04 | 2081.8 MB  |
