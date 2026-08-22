# Phase 3 — Performance Benchmark Report (SPVCNN)

**Objective**: Empirical performance profiling of the 5-stage perception pipeline with SPVCNN.  
**Benchmark Date**: 2026-08-22 21:12:26  
**Hardware & Environment**: macOS-26.5.2-arm64-arm-64bit | CPU: arm (10 threads) | RAM: 16.0 GB  
**PyTorch Version**: 2.8.0 | Python: 3.9.6  
**Model Architecture**: SPVCNN (136,979 parameters)  

---

## 1. Stage-by-Stage Latency Profile (Milliseconds)

| Pipeline Stage     |   Mean (ms) |   Median (ms) |   P95 (ms) |   Min (ms) |   Max (ms) |   Std Dev |
|--------------------|-------------|---------------|------------|------------|------------|-----------|
| 1. LiDAR Loading   |       1.703 |         2.146 |      2.364 |      0.33  |      2.402 |     0.762 |
| 2. Preprocessing   |       2.769 |         2.802 |      2.813 |      2.629 |      2.815 |     0.07  |
| 3. ML Inference    |     100.06  |        95.572 |    108.325 |     94.341 |    108.769 |     6.256 |
| 4. Grid Generation |     157.485 |       165.137 |    168.803 |    120.891 |    169.059 |    18.372 |
| 5. Vis Preparation |      90.791 |        90.949 |     92.54  |     89.266 |     92.76  |     1.353 |
| TOTAL END-TO-END   |     352.807 |       359.496 |    364.464 |    325.38  |    365.396 |    14.272 |

---

## 2. End-to-End System Summary Metrics

| System Metric | Measured Value |
| :--- | :--- |
| **Mean Input Points / Frame** | **40,000 points** |
| **Mean 2.5D Cells / Frame** | **28,308 cells** |
| **Mean End-to-End Latency** | **352.81 ms** |
| **Median End-to-End Latency** | **359.50 ms** |
| **95th Percentile Latency (P95)**| **364.46 ms** |
| **End-to-End Throughput (FPS)** | **2.84 FPS** |
| **Mean Process RAM (RSS)** | **552.71 MB** |
| **Mean CPU Utilization** | **142.0%** |

---

## 3. Scaling Benchmark Across Point Counts

| Points (N)   | Total (ms)   | Load (ms)   | Prep (ms)   | ML (ms)     | Grid (ms)   |   FPS | RAM (MB)   |
|--------------|--------------|-------------|-------------|-------------|-------------|-------|------------|
| 10,000       | 60.34 ms     | 0.01 ms     | 0.51 ms     | 22.08 ms    | 37.73 ms    | 16.57 | 562.1 MB   |
| 100,000      | 814.39 ms    | 0.02 ms     | 4.79 ms     | 311.80 ms   | 497.77 ms   |  1.23 | 1047.9 MB  |
| 500,000      | 3309.03 ms   | 0.06 ms     | 26.95 ms    | 1404.64 ms  | 1877.39 ms  |  0.3  | 1028.8 MB  |
| 1,000,000    | 6069.47 ms   | 0.80 ms     | 56.27 ms    | 3114.36 ms  | 2898.04 ms  |  0.16 | 1228.8 MB  |
| 5,000,000    | 24935.97 ms  | 2.85 ms     | 302.37 ms   | 14903.59 ms | 9727.16 ms  |  0.04 | 2291.4 MB  |
