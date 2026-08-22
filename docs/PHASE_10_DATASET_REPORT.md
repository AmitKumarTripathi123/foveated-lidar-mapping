# Phase 10 Dataset Inventory & Quality Audit Report

**Project**: Foveated 2.5D LiDAR Mapping for Autonomous Navigation  
**Lead Engineer**: Atul (ML/AI Perception Lead)  
**Date**: August 22, 2026  

---

## 1. Dataset Discovery & Storage Inventory

* **Dataset Source**: Local SemanticKITTI / SemanticPOSS representative scan
* **Dataset Root**: `dataset` (supports `$DATASET_ROOT` and `--dataset-root`)
* **Sequence Count**: 1 (`sequence 00`)
* **Frame Count**: 1 (`000000.bin` / `000000.label`)
* **Total Points**: $66,658$ points
* **Supervised Points**: $65,500$ ($98.26\%$)
* **Ignored Points**: $1,158$ ($1.74\%$)
* **Missing Files**: 0
* **Corrupt Files**: 0
* **Duplicate Files**: 0

---

## 2. SIH 4-Class Distribution

| Class ID | Class Name | Supervised Count | Percentage |
| :---: | :--- | :---: | :---: |
| `0` | `drivable_terrain` | $23,000$ | $34.50\%$ |
| `1` | `non_drivable_terrain` | $8,000$ | $12.00\%$ |
| `2` | `static_obstacle` | $28,500$ | $42.76\%$ |
| `3` | `dynamic_object` | $6,000$ | $9.00\%$ |
| `255` | `ignore` | $1,158$ | $1.74\%$ |

---

## 3. Preprocessing & Foveated Voxel Alignment

* **Raw Loading**: $100\%$ finite float32 ($N=66,658$).
* **Amit 3-Zone Downsampling**: Reduced to $50,571$ points ($24.13\%$ reduction) with $100\%$ point-label alignment preservation.
* **Point Normalization**: Sampled to $N=1024$ points for PointNet++ input.
