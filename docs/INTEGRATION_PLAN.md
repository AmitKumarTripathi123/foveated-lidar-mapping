# Integration Plan: Merging Atul Phase 1–5 ML Pipeline into Amit Foveated LiDAR Mapping Repository

**Repository**: `https://github.com/AmitKumarTripathi123/foveated-lidar-mapping`  
**Engineer (ML Perception & Training)**: Atul  
**Data Pipeline & Foveated Preprocessing Lead**: Amit  
**Status**: APPROVED & ACTIVE  

---

## 1. Executive Summary

This document establishes the single canonical architecture for merging Atul''s PyTorch PointNet++ 3D semantic segmentation pipeline (Phases 1–5) into Amit''s foveated LiDAR mapping repository. The combined system creates an autonomous perception pipeline that transforms raw LiDAR point clouds into foveated representations and produces high-precision 4-class semantic segmentation for 2.5D elevation grid mapping.

---

## 2. Component Responsibility & Ownership Matrix

| System Responsibility | Component | Lead Owner | Authoritative Implementation | Status |
| :--- | :--- | :---: | :--- | :---: |
| **Raw LiDAR Loading** | `ml/data/dataset.py` | Atul / Amit | `load_point_cloud`, `load_labels` ($[N, 4]$ `float32`, $[N]$ `uint32`) | Merged & Frozen |
| **Data Quality Validation** | `ml/data/dataset.py` | Atul | `validate_data_integrity`, `validate_point_label_alignment` | Merged & Frozen |
| **Foveated Voxel Sampling** | `ml/data/amit_adapter.py` | Amit | 3-Zone Voxelization ($0\text{--}10\text{m} \to 0.05\text{m}$, $10\text{--}40\text{m} \to 0.15\text{m}$, $40\text{--}100\text{m} \to 0.50\text{m}$) | Merged & Frozen |
| **SIH 4-Class Remapping** | `ml/data/label_mapping.py` | Atul | `SemanticLabelRemapper` (Lookup table, $\{0,1,2,3,255\}$) | Merged & Frozen |
| **Dataset Manifest & Audit** | `ml/data/manifest.py` | Shared | `discover_dataset`, `audit_dataset` | Merged & Active |
| **Point Normalization** | `ml/data/foveated_dataset.py` | Atul | `normalize_point_count` (Configurable $N = 1024$ / $16384$) | Merged & Active |
| **PyTorch Dataset Loader** | `ml/data/foveated_dataset.py` | Atul | `FoveatedLidarDataset` with `lidar_collate_fn` | Merged & Active |
| **PointNet++ Architecture** | `ml/models/pointnet2.py` | Atul | Pure PyTorch `PointNet2SemSeg` ($909,252$ parameters) | Merged & Frozen |
| **ML $\to$ Mapping Contract**| `ml/models/predictor.py` | Shared | `PointNet2Predictor` ($[x, y, z, \text{class}, \text{conf}]$) | Merged & Frozen |
| **Loss Formulations** | `ml/training/losses.py` | Atul | Plain CE & Training-Weighted CE (`ignore_index=255`) | Merged & Active |
| **Evaluation & Metrics** | `ml/training/metrics.py` | Atul | `SemanticSegmentationMetrics` ($4 \times 4$ CM, mIoU, IoU) | Merged & Active |
| **Training & Checkpoint** | `ml/training/trainer.py` | Atul | `PointNet2Trainer` (Validation-mIoU tracking) | Merged & Active |
| **3D Data Augmentation** | `ml/training/augmentation.py`| Atul | `LidarAugmentor` (Training-only yaw, scale, jitter) | Merged & Active |

---

## 3. Canonical Repository Structure

```text
foveated-lidar-mapping/
│
├── dataset/                        # Read-only raw LiDAR sequences (.bin & .label)
│   └── sequences/
│       └── 00/
│           ├── velodyne/
│           └── labels/
│
├── configs/                        # Unified YAML configurations
│   ├── dataset_split.yaml
│   ├── label_mapping.yaml
│   ├── model.yaml
│   ├── preprocessing.yaml
│   └── training.yaml
│
├── ml/                             # Core Machine Learning Perception Module
│   ├── data/                       # Loaders, foveated voxelizer, remapper, dataset
│   │   ├── amit_adapter.py
│   │   ├── dataset.py
│   │   ├── foveated_dataset.py
│   │   ├── label_mapping.py
│   │   ├── manifest.py
│   │   └── preprocessing.py
│   ├── models/                     # Neural architectures & predictor interface
│   │   ├── pointnet2.py
│   │   └── predictor.py
│   ├── training/                   # Training loop, losses, augmentation, metrics
│   │   ├── augmentation.py
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   └── trainer.py
│   └── evaluation/                 # Model evaluation and diagnostic tools
│       └── evaluator.py
│
├── scripts/                        # Automated CLI execution tools
│   ├── compare_experiments.py
│   ├── evaluate.py
│   ├── generate_manifest.py
│   ├── inspect_lidar.py
│   ├── preprocess_foveated.py
│   ├── preprocess_lidar.py
│   ├── remap_labels.py
│   └── train.py
│
├── tests/                          # 96-test automated regression & integration test suite
│   ├── test_full_pipeline.py       # Master 26 integration tests
│   ├── test_label_mapping.py       # Phase 3 tests (14 tests)
│   ├── test_lidar_loader.py        # Phase 1 tests (11 tests)
│   ├── test_pointnet2.py           # Phase 4 tests (16 tests)
│   ├── test_preprocessing.py       # Phase 2 tests (14 tests)
│   └── test_training.py            # Phase 5 tests (15 tests)
│
├── processed/                      # Preprocessed foveated voxel caches
│   ├── train/
│   └── val/
│
├── experiments/                    # Reproducible experiment runs and checkpoints
│   ├── baseline_ce/
│   ├── weighted_ce/
│   └── weighted_ce_aug/
│
├── docs/                           # Team contracts, plans, and final reports
│   ├── INTEGRATION_PLAN.md
│   ├── TEAM_CONTRACT.md
│   └── PHASE_5_FINAL_REPORT.md
│
├── data_manifest.json              # Discovered dataset manifest
├── dataset_audit.json              # Numerical and semantic audit output
├── dataset_audit.md                # Markdown integrity audit report
└── README.md                       # Canonical repository documentation
```

---

## 4. Distance Calculation & Range Policy

* **Distance Policy**: 3D Euclidean distance $d = \sqrt{x^2 + y^2 + z^2}$.
* **Outer Range**: $d \le 100.0\text{m}$ ($d > 100.0\text{m}$ filtered out).
* **Foveated Zones**:
  * Near-Field ($0.0\text{m} \le d < 10.0\text{m}$): Voxel size $= 0.05\text{m}$
  * Mid-Field ($10.0\text{m} \le d < 40.0\text{m}$): Voxel size $= 0.15\text{m}$
  * Far-Field ($40.0\text{m} \le d \le 100.0\text{m}$): Voxel size $= 0.50\text{m}$

---

## 5. Frozen Data Contracts

1. **Phase 1 Contract**: `load_point_cloud` $\to (N, 4)$ `float32`, `load_labels` $\to (N,)$ `uint32`.
2. **Phase 3 Contract**: Remapped labels $\in \{0, 1, 2, 3, 255\}$.
3. **Phase 4 Output Contract**: `PointNet2Predictor.predict(pts)` $\to$ `Dict` with:
   * `xyz`: `(N, 3)` `float32` (exact 1-to-1 order preservation, $\text{in\_xyz}[i] == \text{out\_xyz}[i]$)
   * `predicted_class`: `(N,)` `int64` strictly $\in \{0, 1, 2, 3\}$
   * `confidence`: `(N,)` `float32` strictly $\in [0.0, 1.0]$.
