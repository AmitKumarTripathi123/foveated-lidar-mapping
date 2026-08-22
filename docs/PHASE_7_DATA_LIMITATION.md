# Phase 7 Dataset Limitation & Generalization Statement

**Project**: Foveated 2.5D LiDAR Mapping for Autonomous Navigation  
**Lead Engineer**: Atul (ML/AI Perception Lead)  
**Date**: August 22, 2026  

---

## 1. Executive Summary

This report documents the empirical and statistical limitations of the currently available local LiDAR dataset and outlines the exact requirements for true out-of-sample model generalization.

---

## 2. Available Local Data

* **Sequences Found**: 1 (`sequence 00`)
* **Scans Found**: 1 scan pair (`dataset/sequences/00/velodyne/000000.bin` and `dataset/sequences/00/labels/000000.label`)
* **Total Points**: $66,658$ points
* **Supervised Points**: $65,500$ points ($98.26\%$)
* **Ignored Points**: $1,158$ points ($1.74\%$)

---

## 3. Scientific Integrity & Generalization Rules

1. **No Data Fabrication**: The ML pipeline does not fabricate synthetic frames or pretend that multiple independent driving sequences exist.
2. **No Intrascan Leakage**: Points from the single scan `000000.bin` are **never** partitioned into artificial "train" and "validation" sets to report artificially inflated metrics.
3. **Status Classification**:
   * **Software & Deep Learning Infrastructure**: **PASS** (127 automated tests pass, foveated preprocessing, loss weighting, 3D augmentation, trainer, metrics, predictor, and 2.5D mapping adapter functional).
   * **Statistical Model Generalization**: **DATA-LIMITED** (Validation mIoU on a single frame reflects spatial memorization/collapse rather than generalizable feature learning).

---

## 4. Required Data for Phase 8 Generalization

To achieve a production-grade 3D semantic segmentation model with $>65\%$ mIoU across all 4 classes:
* **Training Sequences**: SemanticPOSS / SemanticKITTI Sequences `00`, `01`, `03`, `04`, `05` ($\sim 15,000+$ scans)
* **Validation Sequence**: Sequence `02` ($\sim 1,500$ scans)
* **Independent Test Sequence**: Sequence `08` ($\sim 4,000$ scans)
* **Hardware**: CUDA GPU with $\ge 8\text{GB}$ VRAM for training at $N=16,384$ points per batch.
