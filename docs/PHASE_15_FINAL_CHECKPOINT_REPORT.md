# PHASE 15 — FINAL CHECKPOINT CERTIFICATION & INDEPENDENT EVALUATION REPORT

**Repository**: `https://github.com/AmitKumarTripathi123/foveated-lidar-mapping`  
**Engineer**: Atul (Senior Perception Scientist & ML Validation Lead)  
**Mapping / Foveated Pipeline Lead**: Amit  
**Branch**: `atul/phase15-final-checkpoint-evaluation`  
**Certification Date**: 2026-08-24  
**Production Checkpoint Certified**: [`experiments/phase12_full_semanticposs_spvcnn/best_checkpoint.pt`](file:///C:/Users/atuls/OneDrive/Desktop/Lidar/experiments/phase12_full_semanticposs_spvcnn/best_checkpoint.pt)  
**Production Artifact Package**: [`artifacts/final_model/`](file:///C:/Users/atuls/OneDrive/Desktop/Lidar/artifacts/final_model/)  

---

## 1. Executive Summary & Objective

In **Phase 15**, the canonical **SPVCNN sparse point-voxel perception model** fine-tuned on the full 2,988-frame SemanticPOSS dataset was subjected to rigorous forensic certification, checksum immutability validation, deterministic reload reproducibility assertions, and partition leakage auditing.

Following comprehensive validation across **Phases 12, 13, and 14**, the Phase 12 checkpoint was definitively frozen as the production perception model:
* **Validation mIoU (Sequence 02 Held-Out)**: **53.59%**
* **Cross-Sequence Mean mIoU (All 6 Sequences)**: **51.94%** (Std: $3.17\%$)
* **Dynamic Object Mean IoU**: **43.68%**
* **End-to-End Latency (RTX 4050 GPU)**: **$96.49\text{ ms}$** (**$10.36\text{ FPS}$** real-time throughput)

---

## 2. Checkpoint Forensic Audit & Immutability Verification

* **Canonical Path**: `experiments/phase12_full_semanticposs_spvcnn/best_checkpoint.pt`
* **File Size**: $1.65\text{ MB}$ ($1,732,951\text{ bytes}$)
* **Architecture**: SPVCNN (Point-Voxel Sparse Convolution, 4-Class SIH Output)
* **Total Parameters**: **138,514**
* **Missing Keys**: **0** | **Unexpected Keys**: **0** | **Shape Mismatches**: **0**
* **Optimizer & Scheduler State**: Present & Intact
* **Recorded Validation Metrics**: Epoch 5, Val mIoU = **53.59%**, Accuracy = **77.53%**

### Checksum Immutability Assertion:
* **SHA256 (Pre-Evaluation)**: `b15c6dfb2f20d1dce4febc47be67f9d50b86a0af72f1895176c6a6ee58bca142`
* **SHA256 (Post-Evaluation)**: `b15c6dfb2f20d1dce4febc47be67f9d50b86a0af72f1895176c6a6ee58bca142`
* **SHA256 (Packaged Artifact)**: `b15c6dfb2f20d1dce4febc47be67f9d50b86a0af72f1895176c6a6ee58bca142`
* **Immutability Status**: **`PASS`** (Bitwise identical).

---

## 3. Deterministic Reload Reproducibility

* Two independent model instances were instantiated from disk and evaluated across identical validation batches.
* **Maximum Absolute Logit Difference**:
  $$\Delta_{\max} = 3.81 \times 10^{-6} < 10^{-5}$$
* **Reload Reproducibility Status**: **`PASS`**.

---

## 4. Independent Evaluation Protocol & Data Leakage Audit

* **Training Partition**: Sequences `00, 01, 03, 04, 05` (**2,488 frames**).
* **Held-Out Validation Partition**: Sequence `02` (**500 frames**).
* **Partition Disjointness**: $\text{Train} \cap \text{Val} = \emptyset$ (**0 overlapping frames**).
* **Independent External Test Set**: **`UNAVAILABLE`** (In strict compliance with scientific integrity rules, Sequence 02 is classified strictly as **`HELD-OUT VALIDATION`**, avoiding artificial claims of external test generalization).
* **Data Leakage Status**: **`PASS`**.

---

## 5. Final Six-Sequence Performance Matrix

| Sequence ID | Partition | Frame Count | Supervised Points | mIoU (%) | Overall Accuracy (%) | IoU 0 (Drivable) | IoU 1 (Non-Drivable) | IoU 2 (Static Obs) | IoU 3 (Dynamic Obj) | Dominant Class % |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **00** | Train | 488 | 22,514,756 | **55.68%** | 83.80% | 73.31% | 17.98% | 79.98% | 51.47% | 59.88% |
| **01** | Train | 500 | 21,770,305 | **51.46%** | 84.50% | 67.20% | 0.00%* | 81.64% | 57.02% | 57.77% |
| **02** | **Held-Out Val** | 500 | 24,925,499 | **53.59%** | 77.53% | 63.02% | 50.88% | 74.42% | 26.06% | 66.70% |
| **03** | Train | 500 | 24,683,611 | **45.60%** | 82.66% | 77.61% | 0.00%* | 78.82% | 25.98% | 58.12% |
| **04** | Train | 500 | 23,869,131 | **53.67%** | 83.79% | 70.90% | 7.65% | 79.52% | 56.63% | 60.10% |
| **05** | Train | 500 | 23,384,610 | **51.64%** | 81.14% | 65.12% | 19.12% | 77.40% | 44.93% | 57.51% |

*\*Note: Sequences 01 and 03 contain virtually zero true non-drivable ground points ($<0.01\%$).*

---

## 6. Final Class-Wise Analysis (SIH 4-Class Standard)

| Semantic Class | Mean IoU | Median IoU | Std IoU | Min IoU | Max IoU | Precision | Recall | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0: `drivable_terrain`** | **69.53%** | 69.05% | 4.98% | 63.02% | 77.61% | 83.96% | 71.64% | Highly stable roadway segmentation |
| **1: `non_drivable_terrain`**| **15.94%** | 12.82% | 17.38% | 0.00% | 50.88% | 86.56% | 55.24% | High precision; sparse in urban scenes |
| **2: `static_obstacle`** | **78.63%** | 79.17% | 2.27% | 74.42% | 81.64% | 88.33% | 82.52% | Consistently robust obstacle boundary detection |
| **3: `dynamic_object`** | **43.68%** | 48.20% | 13.11% | 25.98% | 57.02% | 33.13% | 54.97% | Strong detection on sequences 01, 04, 00 |

---

## 7. Distance Robustness & Spatial Degradation

* **Near-Field ($0\text{--}10\text{m}$, $0.05\text{m}$ voxel)**: **57.50% mIoU** (Drivable: $82.90\%$, Dynamic: $53.01\%$).
* **Mid-Field ($10\text{--}40\text{m}$, $0.15\text{m}$ voxel)**: **49.22% mIoU** (Static Obs: $83.71\%$, Drivable: $63.23\%$).
* **Far-Field ($40\text{--}100\text{m}$, $0.50\text{m}$ voxel)**: **39.07% mIoU** (Static Obs: $86.93\%\text{ at }40\text{--}60\text{m}$, $78.57\%\text{ at }80\text{--}100\text{m}$).
* **Spatial Degradation Assessment**: Expected beam attenuation at $>60\text{m}$ affects dynamic point density while static obstacle structures remain reliably detected.

---

## 8. End-to-End Pipeline Latency & Hardware Profile (RTX 4050 GPU)

| Subsystem Stage | Mean Latency (ms) | Median (ms) | P95 (ms) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **1. LiDAR Point Cloud Loading** | $1.59\text{ ms}$ | $1.55\text{ ms}$ | $1.85\text{ ms}$ | Binary buffer memory mapping |
| **2. 3-Zone Distance Foveation** | $16.76\text{ ms}$ | $16.20\text{ ms}$ | $18.40\text{ ms}$ | Amit 3-zone distance voxel hash |
| **3. SPVCNN GPU Inference** | **$43.11\text{ ms}$** | $42.50\text{ ms}$ | $46.80\text{ ms}$ | NVIDIA RTX 4050 Tensor Cores |
| **4. Vectorized 2.5D GridMap Generation** | **$35.02\text{ ms}$** | $34.10\text{ ms}$ | $38.75\text{ ms}$ | Vectorized NumPy reductions (`np.bincount`/`minimum.at`) |
| **TOTAL END-TO-END LATENCY** | **$96.49\text{ ms}$** | **$94.10\text{ ms}$** | **$105.80\text{ ms}$** | **Throughput: 10.36 FPS (Real-Time)** |

* **Peak Allocated VRAM**: $199.62\text{ MB}$
* **Peak Reserved VRAM**: $582.00\text{ MB}$

---

## 9. Production Checkpoint Artifact Package

The certified production model package is assembled in [`artifacts/final_model/`](file:///C:/Users/atuls/OneDrive/Desktop/Lidar/artifacts/final_model/):
1. `best_checkpoint.pt`: Certified model weights (`138,514` parameters).
2. `checkpoint_sha256.txt`: SHA256 checksum manifest (`b15c6dfb...`).
3. `model_metadata.json`: Model configuration, dataset ontology, and latency metadata.
4. `inference_config.yaml`: Runtime deployment configuration for autonomous navigation.

---

## 10. Automated Test Suite Status

```bash
py -3.12 -m unittest discover -s tests -p "test_*.py" -v
```

* **Regression Test Coverage**: **424 PASS / 0 FAIL** ($100\%$ green across all perception, foveation, mapping, robustness, and certification tests).

---

## 11. Final Scientific Verdict Block

```text
============================================================
PHASE 15 FINAL VERDICT
============================================================

Production Checkpoint:
experiments/phase12_full_semanticposs_spvcnn/best_checkpoint.pt

Checkpoint SHA256:
b15c6dfb2f20d1dce4febc47be67f9d50b86a0af72f1895176c6a6ee58bca142

Checkpoint Integrity:
PASS

Checkpoint Immutability:
PASS

Dataset:
2,988 / 2,988

Sequences:
6 / 6

Training Frames:
2,488

Held-Out Validation Frames:
500

Independent Test:
UNAVAILABLE (SemanticPOSS Sequence 02 used as HELD-OUT VALIDATION)

Validation mIoU:
53.59%

Mean Sequence mIoU:
51.94%

Worst Sequence:
03

Worst Sequence mIoU:
45.60%

Best Sequence:
00

Best Sequence mIoU:
55.68%

Dynamic Object Mean IoU:
43.68%

Distance Robustness:
PASS

Class Robustness:
PASS

Prediction Collapse:
PASS

Checkpoint Reproducibility:
PASS

Data Leakage:
PASS

ML Mapping Contract:
PASS

GridMap25D:
PASS

GPU Benchmark:
PASS (96.49 ms / 10.36 FPS on RTX 4050)

Regression Tests:
424 PASS / 0 FAIL

Production Model:
PHASE 12 SPVCNN CHECKPOINT

Scientific Verdict:
CERTIFIED_WITH_LIMITATIONS

============================================================
```
