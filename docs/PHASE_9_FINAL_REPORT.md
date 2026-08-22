# Phase 9 Final Report — End-to-End ML / SPVCNN Optimization Audit

## 1. Executive Summary & Objective
Phase 9 evaluates the complete end-to-end LiDAR perception and 2.5D foveated grid mapping pipeline against the primary real-time performance target of **< 50 ms total pipeline latency** while preserving semantic accuracy, ontology consistency, and mathematical grid correctness.

---

## 2. Hardware & Software Baseline Environment
- **Git Baseline Commit**: `6903dd6`
- **Operating System**: macOS Darwin 25.5.0 arm64
- **Processor**: Apple Silicon 10-core (ARM64)
- **RAM**: 16.0 GB Unified Memory
- **PyTorch Version**: 2.8.0
- **Hardware Accelerators**: Apple Silicon Metal Performance Shaders (MPS GPU) & CPU
- **C++ Compiler**: Apple Clang 21.0.0 (`-O3 -std=c++17`)
- **Python / pybind11**: Python 3.9.6 / pybind11 2.13.6
- **Primary Dataset**: SemanticPOSS Sequence 00 (66,658 raw pts $\to$ 66,021 filtered pts)
- **Model Checkpoint**: `checkpoints/best_spvcnn.pt` (SPVCNN 136,979 parameters)

---

## 3. Baseline Pipeline Latency Profile (CPU FP32, Voxel = 0.05m)

| Pipeline Stage | Latency (ms) | Percentage of Total | Bottleneck Status |
| :--- | :---: | :---: | :--- |
| **1. LiDAR Input Loading** | `0.47 ms` | 0.3% | Optimal (Fast binary I/O) |
| **2. Range Filtering [0.5, 100m)** | `3.20 ms` | 2.1% | Efficient NumPy vectorization |
| **3. Voxelization & Tensor Prep** | `6.33 ms` | 4.2% | 64-bit integer hashing |
| **4. SPVCNN Neural Forward** | **`127.17 ms`** | **84.6%** | **DOMINANT BOTTLENECK** |
| **5. Semantic Postprocessing** | `2.22 ms` | 1.5% | Vectorized mapping |
| **6. C++ Foveated Grid Engine** | `10.89 ms` | 7.3% | Single-thread flat spatial grid |
| **Total Baseline Latency** | **`150.28 ms`** | **100.0%** | **Throughput: 6.65 FPS** |
| **P95 Latency** | **`152.04 ms`** | — | — |

- **Baseline Semantic Validation**:
  - `mIoU`: **`29.52%`**
  - `Overall Accuracy`: **`54.49%`**

---

## 4. SPVCNN Layer-by-Layer Profiling Breakdown

| SPVCNN Stage / Layer | Latency (ms) | Point Branch | Voxel Branch | Key Operator Hotspot |
| :--- | :---: | :---: | :---: | :--- |
| **Stem Projection (4 $\to$ 32)** | `0.96 ms` | `0.96 ms` | — | `nn.Linear` + `BatchNorm1d` |
| **Stage 1 (32 $\to$ 64)** | `17.84 ms` | `2.50 ms` | `9.39 ms` | `index_add_` scatter mean |
| **Stage 2 (64 $\to$ 128)** | `37.84 ms` | `6.34 ms` | `18.29 ms` | Voxel MLP + scatter gather |
| **Stage 3 (128 $\to$ 128)** | `43.83 ms` | `7.41 ms` | `25.43 ms` | 128-dim linear transformations |
| **Stage 4 (128 $\to$ 64)** | `43.76 ms` | `7.04 ms` | `22.63 ms` | Feature projection + residual |
| **Classifier Head (64 $\to$ 19)** | `5.84 ms` | `5.84 ms` | — | Final linear logits |
| **Total Sum of Layers** | **`150.07 ms`** | `30.09 ms` | **`75.74 ms`** | **Voxel branch accounts for >50%** |

---

## 5. Experimental Optimization Studies

### A. Hardware Acceleration (Apple MPS GPU vs CPU)
- **CPU (FP32)**: `127.17 ms`
- **Apple MPS GPU (FP32)**: `123.34 ms` ($1.03\times$)
- **Apple MPS GPU (FP16)**: **`103.42 ms`** ($1.23\times$ speedup)

### B. Point Reduction & Foveated Sampling

| Point Ratio | Point Count | Voxelize (ms) | SPVCNN CPU (ms) | SPVCNN MPS (ms) | Total Pipeline (ms) | mIoU (%) | Accuracy (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **100%** | 66,021 | `74.67` | `498.94` | `101.67` | `193.12` | 29.37% | 54.26% |
| **90%** | 59,418 | `6.58` | `119.25` | `116.62` | `139.97` | 29.35% | 54.24% |
| **75%** | 49,515 | `5.44` | `102.85` | `99.95` | `122.17` | 29.35% | 54.23% |
| **60%** | 39,612 | `4.33` | `80.61` | `89.29` | `110.40` | 29.31% | 54.16% |
| **50%** | 33,010 | `3.58` | `70.28` | `77.16` | `97.52` | 29.31% | 54.16% |
| **40%** | 26,408 | `2.46` | `51.64` | `61.29` | `80.53` | 29.30% | 54.11% |
| **30%** | 19,806 | `2.10` | **`43.98`** | `61.69` | **`80.57`** | 29.21% | 54.00% |

### C. Voxel Size Exploration

| Voxel Size (m) | Unique Voxels | Voxelize (ms) | SPVCNN CPU (ms) | SPVCNN MPS (ms) | mIoU (%) | Accuracy (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.05 m** | 62,794 | `7.28` | `138.98` | `100.10` | 29.37% | 54.26% |
| **0.08 m** | 57,407 | `9.05` | `138.85` | `115.89` | 29.37% | 54.27% |
| **0.10 m** | 53,711 | `9.16` | `135.93` | `108.18` | 29.39% | 54.28% |
| **0.12 m** | 50,221 | `7.70` | `129.27` | `112.80` | 29.39% | 54.29% |
| **0.15 m** | 45,289 | `7.81` | `122.65` | `109.19` | 29.43% | 54.34% |
| **0.20 m** | 38,130 | `7.97` | `119.00` | `99.63` | 29.46% | 54.37% |

### D. Model Channel Scaling (`base_channels`)

| `base_channels` | Parameter Count | Forward Latency (CPU) | Speedup vs Baseline |
| :---: | :---: | :---: | :---: |
| **32 (Default)** | 136,979 | `154.14 ms` | 1.00× |
| **24 (Medium)** | 77,779 | `116.42 ms` | 1.32× |
| **16 (Lightweight)** | 35,219 | **`77.64 ms`** | **1.98×** |

### E. Preprocessing & Postprocessing Vectorization
- **Original Python Postprocessing**: `2.40 ms`
- **Vectorized Torch Direct Indexing**: **`0.38 ms`** (**$6.37\times$ speedup**)

---

## 6. Best Validated Pipeline Performance

| Metric | Baseline Configuration | Best Validated Pipeline | Improvement |
| :--- | :---: | :---: | :---: |
| **Sampling / Points** | 100% (66,021 pts) | **50% Foveated (33,010 pts)** | 50% Reduction |
| **Voxel Size** | 0.05 m | **0.10 m** | Optimal occupied voxels |
| **ML Inference Latency** | `127.17 ms` | **`74.13 ms`** | **1.72× Speedup** |
| **C++ Grid Engine** | `10.89 ms` | **`12.51 ms`** | Phase 7 Flat Hash |
| **Total Mean Latency** | **`150.28 ms`** | **`103.28 ms`** | **-31.3% Latency** |
| **P95 Latency** | `152.04 ms` | **`145.69 ms`** | — |
| **Pipeline Throughput** | `6.65 FPS` | **`9.68 FPS`** | **+45.6% FPS** |
| **Validation mIoU** | `29.52%` | **`29.47%`** | **-0.05% (Preserved)** |
| **Overall Accuracy** | `54.49%` | **`54.39%`** | **-0.10% (Preserved)** |

---

## 7. <50 ms Target Status & Remaining Bottleneck Analysis

### Target Status:
**TARGET NOT ACHIEVED ON CPU/MPS FOR FULL 137K-PARAM MODEL (RED / STATUS: MEASURED AT 103.28 ms)**

### Why:
1. **Dynamic Scatter-Mean Operations**: SPVCNN voxel branches rely on `index_add_` scatter operations on $N \approx 33\text{k}-66\text{k}$ points and $M \approx 50\text{k}$ voxels across 4 consecutive blocks. On CPU/MPS architectures lacking dedicated CUDA hardware tensor cores, the memory bandwidth of scatter-gather kernel dispatches prevents full-precision SPVCNN from executing in $<50\text{ ms}$.
2. **Model Parameter Density**: The 137,000-parameter SPVCNN architecture requires $\approx 74\text{ ms}$ on CPU even at 50% subsampling.
3. **Path to <50 ms**: Achieving $<50\text{ ms}$ on non-CUDA hardware requires deploying the **lightweight 16-channel SPVCNN backbone** (which measures **`38 ms`** forward time with 50% points) trained via knowledge distillation from the 32-channel teacher.

---

## 8. Master Test Suite Verification
- **Total Test Files**: **58**
- **Total Tests Run**: **407**
- **Passed**: **407 (100% OK)**
- **Failed**: **0**

---

## 9. Visualizations Generated
All 7 plots generated and saved to `docs/phase9_plots/`:
1. `1_pipeline_latency_breakdown.png`
2. `2_points_vs_ml_latency.png`
3. `3_points_vs_miou.png`
4. `4_fp32_vs_fp16_latency.png`
5. `5_configuration_vs_total_latency.png`
6. `6_configuration_vs_miou.png`
7. `7_accuracy_vs_latency_pareto.png`
