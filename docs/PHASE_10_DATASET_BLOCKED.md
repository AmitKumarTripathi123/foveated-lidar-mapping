# Phase 10 Real Dataset Acquisition Gate & Blockage Statement

**Project**: Foveated 2.5D LiDAR Mapping for Autonomous Navigation  
**Lead Engineer**: Atul (ML/AI Perception Lead)  
**Date**: August 22, 2026  

---

## 1. Executive Summary & Hard Stop Condition

Phase 10 enforces the physical data availability gate before launching multi-frame training.

* **Exact Dataset Root Audited**: `dataset` / `$DATASET_ROOT`
* **Raw `.bin` Files Discovered**: 1 (`dataset/sequences/00/velodyne/000000.bin`)
* **Raw `.label` Files Discovered**: 1 (`dataset/sequences/00/labels/000000.label`)
* **Physical Sequence Count**: 1 (`sequence 00`)
* **Physical Frame Count**: 1 (Frame `000000`, 66,658 points)
* **Status**: **PHASE 10 = DATASET BLOCKED (ENGINEERING PASS — 210/210 TESTS PASS)**.

---

## 2. Reason Multi-Frame Generalization Cannot Be Measured

Evaluating semantic segmentation generalization requires independent, out-of-distribution sequence testing (e.g. sequence `08` evaluated on a model trained on sequences `00--05`). With only 1 physical frame present:
1. Cross-sequence generalization is mathematically undefined.
2. Generating fake frames or splitting points from the same scan would violate scientific integrity.
3. Training on a single scan produces expected majority-class collapse into `static_obstacle` (mIoU = 13.66%).

---

## 3. Dataset Acquisition Setup Instructions

To activate multi-frame generalization training:
1. Download SemanticKITTI / SemanticPOSS sequences `00` through `05` and `08`.
2. Configure path: `export DATASET_ROOT="/path/to/kitti/dataset"`.
3. Index data: `python scripts/generate_manifest.py --dataset-root $DATASET_ROOT`.
4. Train baseline: `python scripts/train_phase7.py --config configs/phase10_training.yaml`.
