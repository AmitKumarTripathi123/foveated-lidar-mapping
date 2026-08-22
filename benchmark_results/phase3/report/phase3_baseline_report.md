# PHASE 3 — BASELINE PERFORMANCE REPORT

**Execution Date**: 2026-08-22 23:20:53  
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
| **Git Commit Hash** | `364501fde8323a7afea5c85940e6c9919ac6c4fe` |

---

## 2. Baseline Performance Profile (Per-Stage Latency & Resources)

Measurements collected across representative SemanticPOSS LiDAR scans:

| Pipeline Stage | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) | Min (ms) | Max (ms) | StdDev (ms) | Stage Share (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. LiDAR Loading** | `2.02` | `2.02` | `2.94` | `5.67` | `0.42` | `6.69` | `1.13` | `0.50%` |
| **2. Preprocessing** | `4.13` | `4.04` | `4.57` | `4.72` | `3.88` | `4.78` | `0.22` | `1.03%` |
| **3. ML Inference** | `168.76` | `167.87` | `197.91` | `231.09` | `132.57` | `241.02` | `20.78` | `42.17%` |
| **4. Grid Generation** | `219.44` | `207.93` | `275.91` | `320.54` | `161.26` | `337.31` | `38.04` | `54.83%` |
| **5. Visualization Prep** | `5.86` | `5.66` | `7.34` | `8.59` | `5.13` | `8.81` | `0.78` | `1.46%` |
| **TOTAL PIPELINE** | **`400.21`** | **`387.54`** | **`513.04`** | **`530.71`** | **`324.07`** | **`532.71`** | **`50.55`** | **100.00%** |

### Resource Consumption & Throughput
- **Effective Pipeline Throughput**: **`2.50 FPS`**
- **Resident RAM (RSS)**: **`821.22 MB`** (Peak: `823.98 MB`)
- **Process CPU Load**: **`158.4%`**
- **Mean Points / Frame**: **`66,402`**
- **Mean Occupied 2.5D Cells**: **`36,197`**

---

## 3. Point Cloud Scaling Experiment

| Points | Load | Preprocess | ML | Grid | Visualize | Total | FPS | RAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 10,000 | 0.02 ms | 0.54 ms | 22.01 ms | 74.90 ms | 1.38 ms | 98.84 ms | 10.12 | 852.9 MB |
| 100,000 | 0.02 ms | 4.72 ms | 254.55 ms | 538.29 ms | 15.97 ms | 813.55 ms | 1.23 | 1191.0 MB |
| 500,000 | 0.06 ms | 24.35 ms | 1304.25 ms | 1803.04 ms | 45.42 ms | 3177.11 ms | 0.31 | 1360.6 MB |
| 1,000,000 | 0.51 ms | 46.64 ms | 2618.26 ms | 2928.70 ms | 72.12 ms | 5666.23 ms | 0.18 | 1456.3 MB |
| 5,000,000 | 2.09 ms | 327.94 ms | 15150.77 ms | 9987.42 ms | 86.09 ms | 25554.32 ms | 0.04 | 1291.2 MB |


---

## 4. Bottleneck Identification & Analysis

- **PRIMARY BOTTLENECK**: **`Grid Generation`** (`219.44 ms`, **`54.83%`** of total latency)
- **SECONDARY BOTTLENECK**: **`ML Inference`** (`168.76 ms`, **`42.17%`** of total latency)

### Interpretation & Engineering Next Steps
1. **Grid Generation (`Grid Generation`)** constitutes **`54.83%`** of pipeline execution time. The Python-level spatial binning and cell aggregation loops are the largest drag on FPS.
2. **ML Inference (`ML Inference`)** is already optimized down to **`168.76 ms`** via SPVCNN sparse point-voxel convolutions ($136,979$ parameters).
3. **Recommended Phase 4 Optimization**: Direct NumPy C-vectorization or Cython/Numba spatial hash grid aggregation to reduce Grid Generation from ~150 ms down to < 20 ms, achieving real-time **> 10 FPS** autonomous vehicle throughput.
