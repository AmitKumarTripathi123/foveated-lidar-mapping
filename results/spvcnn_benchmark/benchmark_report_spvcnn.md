# Phase 3 — Performance Benchmark Report (SPVCNN)

**Objective**: Empirical performance profiling of the 5-stage perception pipeline with SPVCNN.  
**Benchmark Date**: 2026-08-23 00:36:44  
**Hardware & Environment**: Windows-11-10.0.26200-SP0 | CPU: Intel64 Family 6 Model 186 Stepping 2, GenuineIntel (12 threads) | RAM: 15.7 GB  
**PyTorch Version**: 2.13.0+cpu | Python: 3.14.5  
**Model Architecture**: SPVCNN (136,979 parameters)  

---

## 1. Stage-by-Stage Latency Profile (Milliseconds)

| Pipeline Stage     |   Mean (ms) |   Median (ms) |   P95 (ms) |   Min (ms) |   Max (ms) |   Std Dev |
|--------------------|-------------|---------------|------------|------------|------------|-----------|
| 1. LiDAR Loading   |      41.78  |        47.956 |     59.7   |      0.977 |     61.374 |    21.106 |
| 2. Preprocessing   |       5.073 |         4.799 |      6.716 |      3.981 |      6.998 |     1.131 |
| 3. ML Inference    |     254.731 |       254.53  |    267.59  |    241.227 |    269.326 |     9.765 |
| 4. Grid Generation |      15.976 |        16.003 |     16.761 |     14.964 |     16.898 |     0.626 |
| 5. Vis Preparation |      13.467 |        13.366 |     14.875 |     12.35  |     15.195 |     0.966 |
| TOTAL END-TO-END   |     331.027 |       338.575 |    363.464 |    275.632 |    367.068 |    30.945 |

---

## 2. End-to-End System Summary Metrics

| System Metric | Measured Value |
| :--- | :--- |
| **Mean Input Points / Frame** | **40,000 points** |
| **Mean 2.5D Cells / Frame** | **28,308 cells** |
| **Mean End-to-End Latency** | **331.03 ms** |
| **Median End-to-End Latency** | **338.57 ms** |
| **95th Percentile Latency (P95)**| **363.46 ms** |
| **End-to-End Throughput (FPS)** | **3.05 FPS** |
| **Mean Process RAM (RSS)** | **494.24 MB** |
| **Mean CPU Utilization** | **763.7%** |

---

## 3. Scaling Benchmark Across Point Counts

|   Points (N) | Total (ms)   | Load (ms)   | Prep (ms)   | ML (ms)     | Grid (ms)   |   FPS | RAM (MB)   |
|--------------|--------------|-------------|-------------|-------------|-------------|-------|------------|
|       10,000 | 116.23 ms    | 0.03 ms     | 1.13 ms     | 69.42 ms    | 4.72 ms     |  8.6  | 388.2 MB   |
|      100,000 | 1364.90 ms   | 0.05 ms     | 10.20 ms    | 727.10 ms   | 47.36 ms    |  0.73 | 645.9 MB   |
|      500,000 | 5535.78 ms   | 1.06 ms     | 62.71 ms    | 3618.76 ms  | 269.51 ms   |  0.18 | 1009.8 MB  |
|    1,000,000 | 10123.07 ms  | 2.10 ms     | 102.38 ms   | 6854.48 ms  | 696.06 ms   |  0.1  | 987.9 MB   |
|    5,000,000 | 46984.60 ms  | 9.18 ms     | 493.33 ms   | 39081.90 ms | 4269.26 ms  |  0.02 | 1863.1 MB  |
