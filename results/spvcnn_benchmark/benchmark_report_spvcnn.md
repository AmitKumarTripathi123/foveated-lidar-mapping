# Phase 3 — Performance Benchmark Report (SPVCNN)

**Objective**: Empirical performance profiling of the 5-stage perception pipeline with SPVCNN.  
**Benchmark Date**: 2026-08-22 18:06:18  
**Hardware & Environment**: macOS-26.5.2-arm64-arm-64bit | CPU: arm (10 threads) | RAM: 16.0 GB  
**PyTorch Version**: 2.8.0 | Python: 3.9.6  
**Model Architecture**: SPVCNN (136,979 parameters)  

---

## 1. Stage-by-Stage Latency Profile (Milliseconds)

| Pipeline Stage     |   Mean (ms) |   Median (ms) |   P95 (ms) |   Min (ms) |   Max (ms) |   Std Dev |
|--------------------|-------------|---------------|------------|------------|------------|-----------|
| 1. LiDAR Loading   |       1.395 |         1.555 |      2.006 |      0.352 |      2.097 |     0.578 |
| 2. Preprocessing   |       2.803 |         2.82  |      2.924 |      2.694 |      2.946 |     0.091 |
| 3. ML Inference    |     107.604 |       106.863 |    109.36  |    106.267 |    109.427 |     1.371 |
| 4. Grid Generation |     164.522 |       174.607 |    175.648 |    125.602 |    175.781 |    19.515 |
| 5. Vis Preparation |      94.704 |        95.154 |     95.817 |     92.73  |     95.823 |     1.184 |
| TOTAL END-TO-END   |     371.028 |       379.752 |    384.785 |    330.841 |    385.148 |    20.334 |

---

## 2. End-to-End System Summary Metrics

| System Metric | Measured Value |
| :--- | :--- |
| **Mean Input Points / Frame** | **40,000 points** |
| **Mean 2.5D Cells / Frame** | **28,308 cells** |
| **Mean End-to-End Latency** | **371.03 ms** |
| **Median End-to-End Latency** | **379.75 ms** |
| **95th Percentile Latency (P95)**| **384.79 ms** |
| **End-to-End Throughput (FPS)** | **2.70 FPS** |
| **Mean Process RAM (RSS)** | **552.36 MB** |
| **Mean CPU Utilization** | **143.5%** |

---

## 3. Scaling Benchmark Across Point Counts

| Points (N)   | Total (ms)   | Load (ms)   | Prep (ms)   | ML (ms)     | Grid (ms)   |   FPS | RAM (MB)   |
|--------------|--------------|-------------|-------------|-------------|-------------|-------|------------|
| 10,000       | 61.87 ms     | 0.01 ms     | 0.54 ms     | 22.85 ms    | 38.46 ms    | 16.16 | 561.6 MB   |
| 100,000      | 820.60 ms    | 0.02 ms     | 4.76 ms     | 323.25 ms   | 492.57 ms   |  1.22 | 1021.7 MB  |
| 500,000      | 3343.18 ms   | 0.18 ms     | 24.85 ms    | 1375.69 ms  | 1942.46 ms  |  0.3  | 1177.3 MB  |
| 1,000,000    | 5819.43 ms   | 0.13 ms     | 48.78 ms    | 2961.69 ms  | 2808.83 ms  |  0.17 | 1290.2 MB  |
| 5,000,000    | 24425.38 ms  | 1.00 ms     | 287.49 ms   | 14464.45 ms | 9672.44 ms  |  0.04 | 2312.5 MB  |
