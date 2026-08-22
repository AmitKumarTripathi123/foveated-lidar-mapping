# PHASE 3 — BASELINE PERFORMANCE REPORT

**Execution Date**: 2026-08-23 00:02:27  
**Model Architecture**: SPVCNN (136,979 parameters)  
**Hardware Platform**: arm (10 Physical Cores, 16.0 GB RAM)  
**Operating System**: macOS-26.5.2-arm64-arm-64bit  

---

## 1. Environment & Software Metadata

| Parameter | Specification |
| :--- | :--- |
| **Operating System** | `macOS-26.5.2-arm64-arm-64bit` |
| **Python Version** | `3.9.6` |
| **PyTorch Version** | `2.8.0` |
| **Processor / Architecture** | `arm` (Darwin) |
| **Total System RAM** | `16.0 GB` |
| **GPU Status** | `UNAVAILABLE (Apple Silicon / CPU)` |
| **Git Commit Hash** | `3678d4c4f8e8bd6081a105046de1d7677d4c4d4b` |

---

## 2. Baseline Performance Profile (Per-Stage Latency & Resources)

Measurements collected across representative SemanticPOSS LiDAR scans:

| Pipeline Stage | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) | Min (ms) | Max (ms) | StdDev (ms) | Stage Share (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. LiDAR Loading** | `1.56` | `1.70` | `2.22` | `2.53` | `0.40` | `2.62` | `0.56` | `0.94%` |
| **2. Preprocessing** | `4.58` | `4.62` | `4.80` | `5.25` | `3.84` | `5.40` | `0.25` | `2.76%` |
| **3. ML Inference** | `142.17` | `142.45` | `146.45` | `146.84` | `137.25` | `146.93` | `2.55` | `85.65%` |
| **4. Grid Generation** | `17.67` | `17.41` | `20.14` | `20.63` | `15.26` | `20.83` | `1.09` | `10.64%` |
| **5. Visualization Prep** | `0.01` | `0.01` | `0.01` | `0.01` | `0.01` | `0.01` | `0.00` | `0.00%` |
| **TOTAL PIPELINE** | **`165.99`** | **`166.11`** | **`172.93`** | **`173.12`** | **`157.07`** | **`173.17`** | **`3.46`** | **100.00%** |

### Resource Consumption & Throughput
- **Effective Pipeline Throughput**: **`6.02 FPS`**
- **Resident RAM (RSS)**: **`851.15 MB`** (Peak: `851.20 MB`)
- **Process CPU Load**: **`257.4%`**
- **Mean Points / Frame**: **`66,402`**
- **Mean Occupied 2.5D Cells**: **`36,197`**

---

## 3. Point Cloud Scaling Experiment

| Points | Load | Preprocess | ML | Grid | Visualize | Total | FPS | RAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 10,000 | 0.02 ms | 0.65 ms | 21.26 ms | 3.70 ms | 0.00 ms | 25.63 ms | 39.02 | 885.3 MB |
| 100,000 | 0.02 ms | 4.80 ms | 274.52 ms | 44.12 ms | 0.01 ms | 323.47 ms | 3.09 | 1194.6 MB |
| 500,000 | 0.06 ms | 25.17 ms | 1202.61 ms | 263.38 ms | 0.02 ms | 1491.24 ms | 0.67 | 1283.9 MB |
| 1,000,000 | 0.50 ms | 50.66 ms | 2463.50 ms | 524.65 ms | 0.01 ms | 3039.32 ms | 0.33 | 1391.5 MB |
| 5,000,000 | 0.74 ms | 305.98 ms | 12223.64 ms | 2894.36 ms | 0.01 ms | 15424.73 ms | 0.06 | 1810.8 MB |


---

## 4. Bottleneck Identification & Analysis

- **PRIMARY BOTTLENECK**: **`ML Inference`** (`142.17 ms`, **`85.65%`** of total latency)
- **SECONDARY BOTTLENECK**: **`Grid Generation`** (`17.67 ms`, **`10.64%`** of total latency)

### Interpretation & Engineering Next Steps
1. **Grid Generation (`ML Inference`)** constitutes **`85.65%`** of pipeline execution time. The Python-level spatial binning and cell aggregation loops are the largest drag on FPS.
2. **ML Inference (`Grid Generation`)** is already optimized down to **`17.67 ms`** via SPVCNN sparse point-voxel convolutions ($136,979$ parameters).
3. **Recommended Phase 4 Optimization**: Direct NumPy C-vectorization or Cython/Numba spatial hash grid aggregation to reduce Grid Generation from ~150 ms down to < 20 ms, achieving real-time **> 10 FPS** autonomous vehicle throughput.
