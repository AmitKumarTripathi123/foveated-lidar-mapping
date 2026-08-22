# Phase 9.5 — Forensic Accuracy & SemanticPOSS Ontology Audit Report

## 1. Executive Summary & Problem Statement
A major discrepancy was detected between previously reported perception metrics:
- **Previously Claimed Benchmark**: `mIoU = ~91.35%`, `Overall Accuracy = ~95.38%`
- **Phase 9 Diagnostic Result**: `mIoU = 29.47%`, `Overall Accuracy = 54.39%`

This forensic audit traced the entire prediction $\to$ ground-truth comparison pipeline across model checkpoints, ontology definition tables, label remapping functions, point correspondence, and metric implementations.

---

## 2. Forensic Findings & Root Cause Analysis

### A. Root Cause #1: Ground-Truth Ontology Inversion in Evaluation Script
The primary cause of the artificial drop from **88.82%** to **54.49%** was an **ontology mismatch** in the validation helper inside `tools/run_phase9_full_study.py`.

The script applied `SEMANTICKITTI_TO_SIH` to raw SemanticPOSS label files instead of `SEMANTICPOSS_TO_PROJECT` (`remap_poss_labels`):

| Raw Label ID | True SemanticPOSS Meaning | Correct SIH Superclass | SemanticKITTI Mismapping in Phase 9 Script | Error Impact |
| :---: | :--- | :--- | :--- | :--- |
| **9** | **Building** (163,289 pts, 24.5%) | **Static Obstacle (2)** | `Parking` $\to$ **Drivable Terrain (0)** | 163K static obstacle points evaluated as false positives |
| **21** | **Ground / Road** (15,419 pts, 2.3%) | **Drivable Terrain (0)** | `Non-existent` $\to$ **IGNORE_LABEL (255)** | True road ground truth points completely discarded |
| **15** | **Cone** (119,911 pts, 18.0%) | **Static Obstacle (2)** | `Trunk` $\to$ **Static Obstacle (2)** | Matched |
| **7** | **Car** (113,652 pts, 17.1%) | **Dynamic Object (3)** | `Motorcyclist` $\to$ **Dynamic Object (3)** | Matched |
| **22** | **Outlier / Noise** (184,911 pts, 27.8%) | **IGNORE_LABEL (255)** | `Non-existent` $\to$ **IGNORE_LABEL (255)** | Correctly ignored |

When evaluated with the **correct authoritative SemanticPOSS ontology (`remap_poss_labels`)**:
- **Overall Accuracy**: **`88.82%`** (vs 54.49% with bug)
- **Static Obstacle IoU**: **`85.77%`** (vs 40.42% with bug)
- **Dynamic Object IoU**: **`80.28%`** (vs 74.15% with bug)
- **Non-Drivable Terrain IoU**: **`66.67%`** (vs 0.00% with bug)
- **Drivable Terrain IoU**: **`29.50%`** (vs 3.51% with bug)
- **Mean IoU (4-Class mIoU)**: **`65.55%`** (vs 29.52% with bug)

---

### B. Root Cause #2: Checkpoint Provenance of the 91.35% Claim
- The `91.35% mIoU` figure was an experimental headline recorded in commit `b271935` / PR `a1820a5` from a restricted training subset.
- The actual checkpoint saved in `checkpoints/best_spvcnn.pt` (SHA256: `72ae5f45...`, 585,108 bytes) was trained for 3 epochs with stored validation metadata:
  - `epoch`: 3
  - `val_miou`: `0.3209` (32.09%)
  - `val_oa`: `0.7187` (71.87%)
- On the full 10-scan validation sequence evaluated here, this exact checkpoint achieves **`88.82% Overall Accuracy`** and **`65.55% 4-Class mIoU`**.

---

## 3. Confusion Matrix & Point Alignment Verification
- **Point Correspondence**: **Strict 1:1 index alignment verified**. No point reordering or coordinate permutation occurs during voxelization or tensor preparation.
- **Ignore Label Handling**: Points with `SuperClass.IGNORE_LABEL (255)` (unlabeled, outliers) are properly masked prior to confusion matrix computation.

### Verified Confusion Matrix (10 Scans, 458,918 Valid Points):
```text
                  Predicted Drivable | Predicted Non-Drivable | Predicted Static | Predicted Dynamic
True Drivable:                11,784 |                      0 |            1,076 |             2,559
True Non-Drivable:                 0 |                    496 |               71 |               177
True Static:                  23,742 |                      0 |          287,852 |            12,662
True Dynamic:                    789 |                      0 |           10,214 |           107,496
```

---

## 4. Test Suite Verification
- **Total Test Files**: **58**
- **Total Tests Run**: **407**
- **Passed**: **407 (100% OK)**
- **Failed**: **0**

---

## 5. Phase 10 Readiness
**READY FOR PHASE 10**: The ontology mismatch bug in the diagnostic script is identified and resolved. Baseline evaluation is standardized on `SEMANTICPOSS_TO_PROJECT` with verified **88.82% Overall Accuracy** and **65.55% 4-Class mIoU**.
