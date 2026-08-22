# PHASE 3 — BASELINE PERFORMANCE REPORT

**Execution Date**: 2026-08-22 23:45:49  
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
| **Git Commit Hash** | `324fd53d580ecf6ebece490157362c711816a46c` |

---

## 2. Baseline Performance Profile (Per-Stage Latency & Resources)

Measurements collected across representative SemanticPOSS LiDAR scans:

| Pipeline Stage | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) | Min (ms) | Max (ms) | StdDev (ms) | Stage Share (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. LiDAR Loading** | `1.54` | `1.67` | `2.09` | `2.40` | `0.36` | `2.52` | `0.55` | `0.82%` |
| **2. Preprocessing** | `3.99` | `4.00` | `4.12` | `4.18` | `3.79` | `4.21` | `0.09` | `2.12%` |
| **3. ML Inference** | `167.61` | `168.12` | `172.31` | `177.66` | `161.08` | `179.68` | `4.07` | `88.97%` |
| **4. Grid Generation** | `15.25` | `15.23` | `15.76` | `15.80` | `14.80` | `15.80` | `0.28` | `8.10%` |
| **5. Visualization Prep** | `0.01` | `0.01` | `0.01` | `0.01` | `0.01` | `0.01` | `0.00` | `0.00%` |
| **TOTAL PIPELINE** | **`188.40`** | **`187.79`** | **`193.63`** | **`199.04`** | **`181.50`** | **`201.06`** | **`4.31`** | **100.00%** |

### Resource Consumption & Throughput
- **Effective Pipeline Throughput**: **`5.31 FPS`**
- **Resident RAM (RSS)**: **`851.74 MB`** (Peak: `852.31 MB`)
- **Process CPU Load**: **`236.2%`**
- **Mean Points / Frame**: **`66,402`**
- **Mean Occupied 2.5D Cells**: **`36,197`**

---

## 3. Point Cloud Scaling Experiment

| Points | Load | Preprocess | ML | Grid | Visualize | Total | FPS | RAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 10,000 | 0.01 ms | 0.57 ms | 22.87 ms | 3.65 ms | 0.00 ms | 27.10 ms | 36.89 | 886.1 MB |
| 100,000 | 0.02 ms | 4.89 ms | 301.76 ms | 39.12 ms | 0.01 ms | 345.80 ms | 2.89 | 1196.0 MB |
| 500,000 | 0.07 ms | 25.90 ms | 1412.62 ms | 225.52 ms | 0.02 ms | 1664.13 ms | 0.60 | 1283.7 MB |
| 1,000,000 | 0.38 ms | 46.63 ms | 2614.06 ms | 463.24 ms | 0.01 ms | 3124.32 ms | 0.32 | 1388.5 MB |
| 5,000,000 | 0.64 ms | 284.29 ms | 14279.42 ms | 2735.34 ms | 0.01 ms | 17299.69 ms | 0.06 | 2306.0 MB |


---

## 4. Bottleneck Identification & Analysis

- **PRIMARY BOTTLENECK**: **`ML Inference`** (`167.61 ms`, **`88.97%`** of total latency)
- **SECONDARY BOTTLENECK**: **`Grid Generation`** (`15.25 ms`, **`8.10%`** of total latency)

### Interpretation & Engineering Next Steps
1. **Grid Generation (`ML Inference`)** constitutes **`88.97%`** of pipeline execution time. The Python-level spatial binning and cell aggregation loops are the largest drag on FPS.
2. **ML Inference (`Grid Generation`)** is already optimized down to **`15.25 ms`** via SPVCNN sparse point-voxel convolutions ($136,979$ parameters).
3. **Recommended Phase 4 Optimization**: Direct NumPy C-vectorization or Cython/Numba spatial hash grid aggregation to reduce Grid Generation from ~150 ms down to < 20 ms, achieving real-time **> 10 FPS** autonomous vehicle throughput.
