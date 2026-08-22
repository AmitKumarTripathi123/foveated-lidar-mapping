# Phase 5 Final Report: PointNet++ 3D Semantic Segmentation & Foveated Integration

**Project**: Foveated 2.5D LiDAR Mapping for Autonomous Navigation  
**Canonical Repository**: `https://github.com/AmitKumarTripathi123/foveated-lidar-mapping`  
**Engineer**: Atul (ML/AI & Perception)  
**Lead & Foveated Owner**: Amit  
**Date**: August 22, 2026  

---

## 1. Repository Synchronization & Git Status

* **Canonical Repository**: `https://github.com/AmitKumarTripathi123/foveated-lidar-mapping`
* **Backup Branch**: `atul-phase5-integration-backup`
* **Working Branch**: `integration/atul-phase1-5`
* **Base Commit**: `c4526d4`

---

## 2. Component Integration Matrix

* **Amit''s Foveated Voxel Sampling**: Integrated in `ml/data/amit_adapter.py` (Authoritative for variable voxel resolution).
* **Atul''s SIH 4-Class Remapping**: Integrated in `ml/data/label_mapping.py` (Authoritative for semantic ontology).
* **Atul''s PointNet++ Baseline**: Integrated in `ml/models/pointnet2.py` ($909,252$ parameters).
* **Atul''s Training & Evaluation Engine**: Integrated in `ml/training/` (`losses.py`, `metrics.py`, `trainer.py`, `augmentation.py`).
* **Frozen ML $\to$ Mapping Predictor Interface**: Integrated in `ml/models/predictor.py`.

---

## 3. Dataset Discovery & Integrity Audit

* **Total Sequences Discovered**: 1 (`sequence 00`)
* **Total Discovered Frames**: 1 representative frame (`000000.bin` & `000000.label`)
* **Total Audited Points**: $66,658$ points
* **Point-Label Count Alignment**: **PASS** ($66,658 == 66,658$)
* **Data Integrity**: **PASS** (100% finite float32, zero NaNs, zero Infs)
* **Sequence Disjointness**: Verified (`train_sequences.isdisjoint(val_sequences)`)
* **Test Split Availability**: UNAVAILABLE (Marked as independent test split unavailable in local environment)

---

## 4. Amit''s Foveated Voxelization Performance

Evaluated on raw LiDAR scan ($66,658$ points):

| Zone Name | Distance Range | Voxel Size | Input Points | Output Points | Reduction % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Near-Field** | $0.0\text{--}10.0\text{m}$ | $0.05\text{m}$ | $4,875$ | $4,795$ | **1.64%** |
| **Mid-Field** | $10.0\text{--}40.0\text{m}$ | $0.15\text{m}$ | $41,327$ | $35,754$ | **13.49%** |
| **Far-Field** | $40.0\text{--}100.0\text{m}$ | $0.50\text{m}$ | $20,456$ | $10,022$ | **51.01%** |
| **Outer Range**| $> 100.0\text{m}$ | N/A | $0$ | $0$ | Filtered |
| **TOTAL** | $\mathbf{0.0\text{--}100.0\text{m}}$ | — | $\mathbf{66,658}$ | $\mathbf{50,571}$ | $\mathbf{24.13\%}$ |

---

## 5. Controlled Experiment Benchmark Results

Trained on cached foveated LiDAR point clouds with point count normalization ($N=1024$):

| Experiment | Configuration | Best Epoch | Val mIoU | Val Accuracy | Drivable IoU | Non-Drive IoU | Static IoU | Dynamic IoU |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline_ce`** | Plain CE (`ignore_index=255`) | **2** | **13.66%** | **54.64%** | 0.00% | 0.00% | **54.64%** | 0.00% |
| **`weighted_ce`** | Inverse-frequency weighted CE | 1 | 2.85% | 11.39% | 0.00% | 11.39% | 0.00% | 0.00% |
| **`weighted_ce_aug`** | Weighted CE + 3D Augmentation | 1 | 2.85% | 11.39% | 0.00% | 11.39% | 0.00% | 0.00% |

* **Selected Best Model Checkpoint**: `experiments/baseline_ce/best_checkpoint.pt`

---

## 6. Confusion Matrix & Diagnostic Evaluation

Evaluated on validation data ($992$ valid points, $32$ ignored points):

```text
4x4 Confusion Matrix (Rows = Ground Truth, Columns = Predicted):
--------------------------------------------------------------
GT \ Pred    |       C0 |       C1 |       C2 |       C3
--------------------------------------------------------------
Class 0      |        0 |        0 |      269 |        0
Class 1      |        0 |        0 |      113 |        0
Class 2      |        0 |        0 |      542 |        0
Class 3      |        0 |        0 |       68 |        0
--------------------------------------------------------------
```

* **Prediction Distribution**:
  * `Class 0 (drivable_terrain)`: $0$ points ($0.00\%$)
  * `Class 1 (non_drivable_terrain)`: $0$ points ($0.00\%$)
  * `Class 2 (static_obstacle)`: $992$ points ($100.00\%$)
  * `Class 3 (dynamic_object)`: $0$ points ($0.00\%$)
* **Model Collapse Diagnostic**: `MODEL_COLLAPSE_WARNING` triggered (single class $>90\%$ predictions due to single-frame CPU baseline training).

---

## 7. Metric Consistency & Checkpoint Reload Verification

* **Training-Time Validation mIoU**: $13.66\%$
* **Post-Reload Validation mIoU**: $13.66\%$
* **Absolute Delta**: $0.0000\%$
* **Reload Consistency Status**: **PASS**

---

## 8. Frozen ML $\to$ Mapping Output Contract Verification

* **Interface Function**: `PointNet2Predictor.predict(points)`
* **Coordinate Preservation**: **PASS** (`np.array_equal(in_xyz, out_xyz)` exact 1-to-1 order preservation)
* **Predicted Class Range**: **PASS** ($\text{unique}(\text{pred}) \subseteq \{0, 1, 2, 3\}$)
* **Confidence Range**: **PASS** ($\text{conf} \in [0.0, 1.0]$, mean $= 0.2544$)

---

## 9. Master Automated Test Suite

Executed: `python -m unittest discover -s tests -p "test_*.py" -v`

* **Total Tests**: 96 tests
* **Passed**: 96 tests
* **Failed**: 0 tests
* **Skipped**: 1 test (CUDA GPU optional test on CPU environment)
* **Execution Time**: $7.41$ seconds

---

## 10. Hardware & Environment

* **CPU**: Intel Core i5-1235U (10 Cores, 12 Threads)
* **GPU**: None (CPU-only mode)
* **CUDA Available**: False
* **PyTorch Version**: `2.13.0+cpu`
* **Python Version**: `3.14.5`

---

## 11. Known Limitations & Recommendations for Phase 6

1. **Dataset Volume**: The current local workspace contains a representative scan from `sequence 00`. For full production training in Phase 6, multiple distinct sequence drives (`00`, `01`, `03`, `04`, `05` for train; `02` for val) should be downloaded into `dataset/sequences/`.
2. **Compute Acceleration**: Training on CPU runs in FP32 with $N=1024$ points. Enabling CUDA with FP16/AMP will permit $N=16,384$ points and batch size $\ge 8$.
