# PHASE 19.1 — OPTIMIZATION PROFILER + DISTANCE-WISE mIoU AUDIT REPORT

**Problem Statement**: SIH Problem Statement PS 26130 — *Foveated 2.5D LiDAR Mapping for Autonomous Navigation*  
**Repository**: `https://github.com/AmitKumarTripathi123/foveated-lidar-mapping`  
**Engineer**: Senior LiDAR Perception & ML Systems Optimization Lead (Atul)  
**Mapping / Foveated Pipeline Lead**: Amit  
**Branch**: `atul/phase19.1-profiler-distance-audit`  
**Execution Date**: 2026-08-24  
**Production Checkpoint Tested**: [`experiments/phase12_full_semanticposs_spvcnn/best_checkpoint.pt`](file:///C:/Users/atuls/OneDrive/Desktop/Lidar/experiments/phase12_full_semanticposs_spvcnn/best_checkpoint.pt)  
**SHA256**: `b15c6dfb2f20d1dce4febc47be67f9d50b86a0af72f1895176c6a6ee58bca142` (`VERIFIED IMMUTABLE`)  
**Single Source of Truth Config**: [`configs/system_config.yaml`](file:///C:/Users/atuls/OneDrive/Desktop/Lidar/configs/system_config.yaml)  
**Generated Diagnostic Figures**:
* [`reports/phase19_1/figures/latency_breakdown.png`](file:///C:/Users/atuls/OneDrive/Desktop/Lidar/reports/phase19_1/figures/latency_breakdown.png)
* [`reports/phase19_1/figures/distance_miou.png`](file:///C:/Users/atuls/OneDrive/Desktop/Lidar/reports/phase19_1/figures/distance_miou.png)
* [`reports/phase19_1/figures/class_iou.png`](file:///C:/Users/atuls/OneDrive/Desktop/Lidar/reports/phase19_1/figures/class_iou.png)
* [`reports/phase19_1/figures/confusion_matrix.png`](file:///C:/Users/atuls/OneDrive/Desktop/Lidar/reports/phase19_1/figures/confusion_matrix.png)
* [`reports/phase19_1/figures/performance_summary.png`](file:///C:/Users/atuls/OneDrive/Desktop/Lidar/reports/phase19_1/figures/performance_summary.png)
* [`reports/phase19_1/figures/gpu_cpu_telemetry.png`](file:///C:/Users/atuls/OneDrive/Desktop/Lidar/reports/phase19_1/figures/gpu_cpu_telemetry.png)

---

## 1. Executive Summary

In **Phase 19.1**, a reproducible, non-destructive measurement harness was implemented to profile the canonical foveated perception and mapping pipeline across 100 evaluation frames on an NVIDIA GeForce RTX 4050 Laptop GPU.

### Core Scientific Findings:
1. **Perception Latency vs Replay Latency**:
   - **Active Perception Latency (Preloaded)**: **$94.10\text{ ms}$** (**$10.63\text{ FPS}$**), successfully operating below the $\le 100\text{ ms}$ real-time threshold.
   - **Replay Latency (Disk I/O Included)**: **$95.67\text{ ms}$**.
2. **Primary Execution Bottleneck**:
   - **`grid` (2.5D GridMap Compilation)**: **$36.57\text{ ms}$** ($38.03\%$ of total latency) is the **Primary Bottleneck**.
   - **`spvcnn` (CUDA Forward Pass)**: **$20.37\text{ ms}$** ($21.18\%$) is the **Secondary Bottleneck**.
   - **`foveation`**: **$17.76\text{ ms}$** ($18.46\%$).
   - **`ml_preprocess`**: **$13.67\text{ ms}$** ($14.22\%$).
3. **Distance-Stratified Semantic Accuracy**:
   - **Near Zone ($0\text{--}10\text{m}$ @ $5\text{cm}$)**: **$66.60\%$ mIoU** (High fidelity: $78.64\%$ Drivable, $94.44\%$ Non-Drivable).
   - **Mid Zone ($10\text{--}40\text{m}$ @ $15\text{cm}$)**: **$42.82\%$ mIoU** ($61.22\%$ Drivable, $78.33\%$ Static).
   - **Far Zone ($40\text{--}100\text{m}$ @ $50\text{cm}$)**: **$36.98\%$ mIoU** (Dominated by Static Obstacles: $87.12\%$).
4. **Accuracy Weaknesses**:
   - **Weakest Semantic Class**: `non_drivable` ($27.28\%$ IoU overall due to severe mid/far sparsity).
   - **Worst Distance Band**: `far_40_100m` ($36.98\%$ mIoU, $-29.62\%$ drop relative to near-field).

---

## 2. Stage-Wise Execution Latency Breakdown (`optimization_profile.json`)

Evaluated over 100 frames with `torch.cuda.Event` synchronization:

| Pipeline Stage | Mean Latency (ms) | Median (ms) | P95 (ms) | P99 (ms) | % Total | Role & Implementation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`io`** | 1.57 | 1.53 | 2.39 | 2.53 | 1.64% | Disk reader / cache lookup |
| **`range_filter`** | 3.68 | 3.45 | 5.43 | 5.96 | 3.82% | Radial range filtering $[0.5, 100]\text{m}$ |
| **`foveation`** | 17.76 | 17.76 | 22.61 | 31.27 | 18.46% | 3-Zone distance voxelization |
| **`ml_preprocess`** | 13.67 | 13.20 | 17.29 | 23.81 | 14.22% | Coordinate hashing & tensor packaging |
| **`spvcnn`** | 20.37 | 14.93 | 37.89 | 39.79 | 21.18% | CUDA Tensor-Core FP32 inference |
| **`postprocess`** | 2.06 | 1.98 | 3.06 | 3.82 | 2.14% | Softmax argmax & DTO validation |
| **`grid`** ⭐ | **36.57** | **36.72** | **45.70** | **49.82** | **38.03%** | **Vectorized 2.5D rasterizer (PRIMARY)** |
| **`visualization`** | 0.50 | 0.50 | 0.50 | 0.50 | 0.52% | Diagnostic telemetry HUD |
| **Active Perception Total** | **94.10** | **89.54** | **122.46** | **132.80** | **100.0%** | **$10.63\text{ FPS}$ (Target Met)** |

---

## 3. Global & Distance-Stratified Semantic Metrics

### A. Authoritative 4-Class Global Metrics (`accuracy_audit.json`)
Evaluated across $4,675,813$ ground-truth verified points (ignoring label 255):

* **Overall Semantic mIoU**: **$52.04\%$**
* **Point Accuracy**: **$80.63\%$**
* **Mean Class Accuracy (mAcc)**: **$61.82\%$**

| Class ID & Name | IoU (%) | Precision (%) | Recall (%) | F1 Score | Support (Points) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0: `drivable_terrain`** | 66.34% | 83.46% | 76.38% | 0.7977 | 1,038,983 |
| **1: `non_drivable_terrain`** | 27.28% | 87.91% | 28.35% | 0.4287 | 3,898 |
| **2: `static_obstacle`** | 76.96% | 85.44% | 88.58% | 0.8698 | 2,931,833 |
| **3: `dynamic_object`** | 37.56% | 55.28% | 53.95% | 0.5461 | 701,099 |

---

### B. Distance-Stratified Segmentation Telemetry (`distance_miou.json`)

| Distance Zone | Radius & Resolution | Total Points | mIoU (%) | Drivable IoU | Non-Drivable IoU | Static IoU | Dynamic IoU | Mean Conf | Occupied Cells | Pts / Cell |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Near Zone** | $0\text{--}10\text{m}$ @ $5\text{cm}$ | 789,038 | **66.60%** | 78.64% | 94.44% | 40.34% | 53.00% | 0.8247 | 703,254 | 1.12 |
| **Mid Zone** | $10\text{--}40\text{m}$ @ $15\text{cm}$ | 3,282,440 | **42.82%** | 61.22% | 0.00% | 78.33% | 31.72% | 0.8848 | 2,849,954 | 1.15 |
| **Far Zone** | $40\text{--}100\text{m}$ @ $50\text{cm}$ | 719,334 | **36.98%** | 40.48% | 0.00% | 87.12% | 20.32% | 0.8899 | 617,708 | 1.16 |

---

## 4. Hardware & Resource Telemetry (`telemetry.json`)

* **GPU Name**: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)
* **GPU Memory Allocated**: $39.46\text{ MB}$
* **GPU Memory Reserved**: $70.00\text{ MB}$
* **GPU Memory Peak**: $41.87\text{ MB}$ (Extremely light VRAM footprint)
* **Host CPU Utilization**: $21.9\%$
* **Process RSS Memory**: $674.31\text{ MB}$
* **Host System RAM**: $66.4\%$
* **Dropped Frames**: $0 / 100$

---

## 5. Phase 19.2 Optimization Direction (Evidence-Based)

Based strictly on the measured empirical profile:

1. **Primary Latency Optimization Target (Grid Engine)**:
   - The Python NumPy 2.5D rasterizer consumes **$36.57\text{ ms}$ ($38.03\%$ of latency)**.
   - **Action**: Bind and accelerate the grid rasterizer using the native C++/CUDA backend (`cpp/src/foveated_grid.cpp`) to reduce grid time from $36.57\text{ ms} \to < 3.0\text{ ms}$.
2. **Secondary Latency Optimization Target (Inference & Sampler Fusion)**:
   - SPVCNN CUDA execution takes **$20.37\text{ ms}$** and Foveated Sampling takes **$17.76\text{ ms}$**.
   - **Action**: TorchScript / ONNX / TensorRT / CUDA-kernel fusion will compress perception latency below $35\text{ ms}$ ($>28\text{ FPS}$).

---

## Final Scientific Verdict Block

```text
============================================================
PHASE 19.1 — OPTIMIZATION PROFILER & ACCURACY AUDIT VERDICT
============================================================

Repository:
https://github.com/AmitKumarTripathi123/foveated-lidar-mapping

Evaluation Scale:
100 Evaluation Frames (10 Warmup Frames)

Production Checkpoint:
experiments/phase12_full_semanticposs_spvcnn/best_checkpoint.pt (IMMUTABLE)

SHA256:
b15c6dfb2f20d1dce4febc47be67f9d50b86a0af72f1895176c6a6ee58bca142

Baseline vs Measured Comparison:
• Active Perception Latency: 94.10 ms (10.63 FPS) [Target <= 100 ms MET]
• Replay Latency:            95.67 ms
• Overall Semantic mIoU:     52.04% (Point Acc: 80.63%)
• Near / Mid / Far mIoU:     66.60% / 42.82% / 36.98%

Identified Bottlenecks:
• Primary:   GRID MAP RASTERIZATION (36.57 ms / 38.03%)
• Secondary: SPVCNN FORWARD INFERENCE (20.37 ms / 21.18%)
• Weak Class: NON_DRIVABLE TERRAIN (27.28% IoU)
• Worst Band: FAR ZONE 40-100m (36.98% mIoU)

Phase 19.2 Recommendation:
Accelerate 2.5D grid rasterization via native C++/CUDA kernel binding
to reduce rasterization latency from 36.57 ms to < 3.0 ms.

Unit & Boundary Tests:
7 PASS / 0 FAIL

Status:
AUDIT_COMPLETE
============================================================
```
