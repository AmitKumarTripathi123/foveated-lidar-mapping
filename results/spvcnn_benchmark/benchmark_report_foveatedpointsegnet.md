# Phase 3 — Performance Benchmark Report (FoveatedPointSegNet)

**Objective**: Empirical performance profiling of the 5-stage perception pipeline with FoveatedPointSegNet.  
**Benchmark Date**: 2026-08-22 23:05:59  
**Hardware & Environment**: macOS-26.5.2-arm64-arm-64bit | CPU: arm (10 threads) | RAM: 16.0 GB  
**PyTorch Version**: 2.8.0 | Python: 3.9.6  
**Model Architecture**: FoveatedPointSegNet (451,460 parameters)  

---

## 1. Stage-by-Stage Latency Profile (Milliseconds)

| Pipeline Stage     |   Mean (ms) |   Median (ms) |   P95 (ms) |   Min (ms) |   Max (ms) |   Std Dev |
|--------------------|-------------|---------------|------------|------------|------------|-----------|
| 1. LiDAR Loading   |       1.895 |         2.183 |      2.45  |      0.327 |      2.474 |     0.793 |
| 2. Preprocessing   |       2.799 |         2.828 |      2.883 |      2.709 |      2.896 |     0.069 |
| 3. ML Inference    |     214.35  |       176.839 |    320.208 |    157.549 |    340.783 |    69.676 |
| 4. Grid Generation |     170.515 |       173.945 |    197.615 |    120.386 |    198.612 |    27.816 |
| 5. Vis Preparation |      92.859 |        92.365 |     96.501 |     89.57  |     97.212 |     2.55  |
| TOTAL END-TO-END   |     482.418 |       449.745 |    615.537 |    374.575 |    635.963 |    92.812 |

---

## 2. End-to-End System Summary Metrics

| System Metric | Measured Value |
| :--- | :--- |
| **Mean Input Points / Frame** | **40,000 points** |
| **Mean 2.5D Cells / Frame** | **28,308 cells** |
| **Mean End-to-End Latency** | **482.42 ms** |
| **Median End-to-End Latency** | **449.75 ms** |
| **95th Percentile Latency (P95)**| **615.54 ms** |
| **End-to-End Throughput (FPS)** | **2.14 FPS** |
| **Mean Process RAM (RSS)** | **720.20 MB** |
| **Mean CPU Utilization** | **176.3%** |

---

## 3. Scaling Benchmark Across Point Counts

| Points (N)   | Total (ms)   | Load (ms)   | Prep (ms)   | ML (ms)     | Grid (ms)   |   FPS | RAM (MB)   |
|--------------|--------------|-------------|-------------|-------------|-------------|-------|------------|
| 10,000       | 80.47 ms     | 0.01 ms     | 0.54 ms     | 40.30 ms    | 39.61 ms    | 12.43 | 744.0 MB   |
| 100,000      | 1147.59 ms   | 0.02 ms     | 4.76 ms     | 641.80 ms   | 501.00 ms   |  0.87 | 1572.0 MB  |
| 500,000      | 5592.31 ms   | 0.06 ms     | 24.76 ms    | 3505.60 ms  | 2061.88 ms  |  0.18 | 1399.4 MB  |
| 1,000,000    | 9207.34 ms   | 2.82 ms     | 76.71 ms    | 6260.27 ms  | 2867.53 ms  |  0.11 | 1646.5 MB  |
| 5,000,000    | 42592.34 ms  | 1.29 ms     | 237.24 ms   | 32441.82 ms | 9911.97 ms  |  0.02 | 2212.2 MB  |
