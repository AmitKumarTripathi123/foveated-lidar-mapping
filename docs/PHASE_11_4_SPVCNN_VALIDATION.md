# Phase 11.4 SPVCNN Scientific Validation & Checkpoint Audit Report

**Project**: Foveated 2.5D LiDAR Mapping for Autonomous Navigation  
**ML/Perception Lead**: Atul  
**Mapping/Preprocessing Lead**: Amit  
**Date**: August 22, 2026  

---

## 1. Executive Summary

Phase 11.4 performed an exhaustive scientific validation, forensic checkpoint audit, and zero-shot performance evaluation for the SPVCNN perception pipeline on real SemanticPOSS LiDAR data (`dataset/sequences/00/velodyne/000000.bin`).

---

## 2. Forensic Identity & Disambiguation Matrix

* **SPVCNN Checkpoint Path**: `checkpoints/spvcnn_pretrained.pt`
* **SPVCNN Checkpoint SHA256**: `cb1a6f44fd11938e19c6dfaa85f39c53093ca738a4faa3b8fc9a9c5ca3f56750`
* **Total Parameters**: $136,979$ (0 missing keys, 0 unexpected keys)
* **Model Baseline Disambiguation**:
  * `LEGACY_POINTNET2_BASELINE_MIOU`: **$13.66\%$** (`experiments/phase7_baseline_ce/best_checkpoint.pt`)
  * `SPVCNN_ZERO_SHOT_MIOU`: **$9.92\%$** (`checkpoints/spvcnn_pretrained.pt`)

---

## 3. Real Scan Performance & Confusion Matrix

Evaluated on real representative LiDAR scan (`000000.bin` with $66,658$ raw points downsampled via Amit''s foveated voxelizer to $50,571$ points):

### Per-Class Evaluation:
* **Class 0 (`drivable_terrain`)**: IoU = $0.00\%$, Precision = $0.00\%$, Recall = $0.00\%$
* **Class 1 (`non_drivable_terrain`)**: IoU = $0.00\%$, Precision = $0.00\%$, Recall = $0.00\%$
* **Class 2 (`static_obstacle`)**: IoU = **$39.67\%$**, Precision = **$100.00\%$**, Recall = **$39.67\%$** (TP=$1,494$, FP=$0$, FN=$2,272$)
* **Class 3 (`dynamic_object`)**: IoU = $0.00\%$, Precision = $0.00\%$, Recall = $0.00\%$
* **Zero-Shot Overall Accuracy**: **$39.67\%$**
* **Zero-Shot mIoU**: **$9.92\%$**

### Voxel & Confidence Diagnostics:
* **Spatial Alignment**: $100\%$ ($0$ XYZ mismatches between input and output).
* **Voxel Statistics**: $50,571$ points $\to$ $50,437$ unique $0.05\text{m}$ voxels (mean $1.00$ points/voxel, max $3$ points/voxel).
* **Confidence**: Min=$0.0581$, Max=$0.3870$, Mean=$0.1455$, Median=$0.1339$, 0 NaNs, 0 Infs.
* **Prediction Entropy**: $1.0704$ bits (Dominant class: `dynamic_object` at $55.75\%$, no $>90\%$ model collapse).

---

## 4. Phase 11.4 Authoritative Status Block

```text
ENGINEERING INTEGRATION:
PASS

CHECKPOINT:
VERIFIED

ONTOLOGY:
VERIFIED

ZERO-SHOT SIH:
VALID

POINT-VECTOR ALIGNMENT:
PASS

MAPPING:
PASS

GRIDMAP25D:
PASS

SPVCNN MIOU:
9.92% (Zero-Shot Single-Frame Baseline)

GENERALIZATION:
DATA-LIMITED
```
