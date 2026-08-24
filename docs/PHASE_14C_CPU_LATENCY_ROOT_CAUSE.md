# Phase 14C — CPU Latency Root-Cause & Optimization Audit Report

## 1. Executive Summary
This report documents the forensic root-cause analysis and architectural profiling of the **Foveated 2.5D LiDAR Pipeline on CPU architectures**.

---

## 2. Component Latency Breakdown (66,658 Real Points)

| Pipeline Stage | Implementation | Latency (ms) | Fraction (%) | Bottleneck Status |
| :--- | :--- | :---: | :---: | :---: |
| **A. LiDAR Binary Parsing** | `src/data_loader.py` | 0.11 ms | 0.1% | Negligible |
| **B. Preprocessing & Range Filter** | `ml/data/preprocessing.py` | 1.26 ms | 0.8% | Negligible |
| **C. Point-Voxel Hash Voxelization** | `ml/data/spvcnn_adapter.py` | 7.99 ms | 4.8% | Low |
| **D. PyTorch Tensor Allocation** | PyTorch CPU memory | 0.00 ms | 0.0% | Zero-copy |
| **E. SPVCNN Neural Forward Pass** | `ml/models/spvcnn.py` | **141.45 ms** | **85.7%** | **PRIMARY BOTTLENECK** |
| **F. Logits -> Label/Conf Adapter** | `ml/models/spvcnn_label_adapter.py` | 2.38 ms | 1.4% | Negligible |
| **G. C++ Foveated 2.5D Grid Engine** | `cpp/src/foveated_grid.cpp` | 11.81 ms | 7.2% | Low |
| **TOTAL CPU PIPELINE** | **End-to-End** | **165.01 ms** | **100.0%** | **TARGET NOT MET (6.1 FPS)** |

---

## 3. SPVCNN Layer-by-Layer Profiling

* **Stem Layer ($4 \to 32$)**: $0.92\text{ ms}$ ($0.7\%$)
* **Stage 1 Block ($32 \to 64$)**: $15.85\text{ ms}$ ($11.2\%$)
* **Stage 2 Block ($64 \to 128$)**: **$38.88\text{ ms}$ ($27.5\%$)**
* **Stage 3 Block ($128 \to 128$)**: **$42.04\text{ ms}$ ($29.7\%$)**
* **Stage 4 Block ($128 \to 64$)**: **$33.78\text{ ms}$ ($23.9\%$)**
* **Classifier Head**: $6.48\text{ ms}$ ($4.6\%$)

**Root Cause**: Stages 2, 3, and 4 account for **$81.1\%$** of total neural compute time due to repeated point-wise linear matrix multiplications with 128-channel tensors across 66,402 points.

---

## 4. Architectural Model Reduction Strategies

| Model Variant | Base Channels | Parameters | CPU Latency | CPU Throughput | Speedup | Meets <50ms CPU? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Teacher / Full SPVCNN** | 32 | 136,979 | 144.66 ms | 6.9 FPS | 1.00x | NO |
| **Distilled Student** | 16 | 35,219 | 81.06 ms | 12.3 FPS | 1.78x | NO |
| **Lightweight Student** | 8 | 9,299 | **38.89 ms** | **25.7 FPS** | **3.72x** | **YES** |

---

## 5. Architectural Recommendation

1. **Production Deployment on Vehicle**: Use **CUDA GPU acceleration** with the verified FP16 SPVCNN Student ($\sim 23.59\text{ ms} \implies 42.4\text{ FPS}$), providing a generous $+26.41\text{ ms}$ headroom against the $<50\text{ ms}$ ceiling.
2. **CPU-Constrained Hardware**: If deployed on a CPU-only platform, adopt the **8-channel Lightweight Student** or a **Foveated Semantic Network** (downsampling mid/far-field points during ML inference while maintaining 5cm resolution in the C++ Grid Engine).
