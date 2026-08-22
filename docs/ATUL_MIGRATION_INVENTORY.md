# Atul Phase 1–5 ML Migration Inventory

This inventory tracks all files migrated from Atul''s Phase 1–5 ML perception system into Amit''s canonical `foveated-lidar-mapping` repository.

---

## 1. Migration File Inventory

| Phase | Source Path | Target Canonical Path | Purpose | Action |
| :--- | :--- | :--- | :--- | :---: |
| **Phase 1** | `ml/data/dataset.py` | `ml/data/dataset.py` | Robust raw LiDAR loader (`load_point_cloud`, `load_labels`, validation) | **Migrated & Active** |
| **Phase 2** | `ml/data/preprocessing.py`| `ml/data/preprocessing.py`| Modular preprocessor, range filtering, sampling & coordinate handling | **Migrated & Active** |
| **Phase 2** | `ml/data/amit_adapter.py` | `ml/data/amit_adapter.py` | Amit''s 3-Zone Foveated Voxel Sampling adapter ($0\text{--}10\text{m}$, $10\text{--}40\text{m}$, $40\text{--}100\text{m}$) | **Migrated & Active** |
| **Phase 2** | `ml/data/foveated_dataset.py`| `ml/data/foveated_dataset.py`| PyTorch Dataset adapter loading foveated scans with point-count normalization | **Migrated & Active** |
| **Phase 3** | `ml/data/label_mapping.py`| `ml/data/label_mapping.py`| Authoritative SIH 4-class label remapping engine (`0, 1, 2, 3, 255`) | **Migrated & Active** |
| **Phase 3** | `ml/data/manifest.py` | `ml/data/manifest.py` | Automated dataset discovery, split manifest & integrity audit engine | **Migrated & Active** |
| **Phase 4** | `ml/models/pointnet2.py` | `ml/models/pointnet2.py` | PointNet++ 3D semantic segmentation neural network ($909,252$ parameters) | **Migrated & Active** |
| **Phase 4** | `ml/models/predictor.py` | `ml/models/predictor.py` | `PointNet2Predictor` enforcing Amit''s frozen ML $\to$ Mapping contract | **Migrated & Active** |
| **Phase 5** | `ml/training/losses.py` | `ml/training/losses.py` | Plain & Training-Weighted Cross-Entropy loss functions (`ignore_index=255`) | **Migrated & Active** |
| **Phase 5** | `ml/training/metrics.py` | `ml/training/metrics.py` | Point-wise evaluation engine ($4 \times 4$ confusion matrix, IoU, mIoU, precision, recall) | **Migrated & Active** |
| **Phase 5** | `ml/training/augmentation.py`| `ml/training/augmentation.py`| Training-only 3D geometric augmentation (yaw rotation, scaling, jitter) | **Migrated & Active** |
| **Phase 5** | `ml/training/trainer.py` | `ml/training/trainer.py` | `PointNet2Trainer` with validation-mIoU tracking and automated logging | **Migrated & Active** |
| **Configs** | `configs/*.yaml` | `configs/*.yaml` | Centralized YAML configurations (dataset split, label mapping, model, training) | **Migrated & Active** |
| **Scripts** | `scripts/*.py` | `scripts/*.py` | CLI execution scripts for training, evaluation, comparison, preprocessing, manifest | **Migrated & Active** |
| **Tests** | `tests/test_*.py` | `tests/test_*.py` | 96-test automated regression & full integration test suite | **Migrated & Active** |
| **Docs** | `docs/*.md` | `docs/*.md` | Integration plan, team contract, final report, and migration inventory | **Migrated & Active** |

---

## 2. Amit Existing Component Preservation

| Existing Amit File | Role | Status in Canonical Repository |
| :--- | :--- | :---: |
| `dataset.py` | Amit''s root foveated dataset & multi-band voxel downsampler | **Preserved 100% Unchanged** |
| `class_map.py` | Amit''s root label mapping & class color visualization | **Preserved 100% Unchanged** |
| `preprocess.py` | Amit''s root preprocessing cache generator | **Preserved 100% Unchanged** |
| `verify_pipeline.py`| Amit''s pipeline verification script | **Preserved 100% Unchanged** |
| `lidar.py` | Amit''s LiDAR test script | **Preserved 100% Unchanged** |
| `testnumpy.py` | Quick environment test | **Preserved 100% Unchanged** |
| `README.md` | Root documentation | **Updated with Unified Overview** |
| `.gitignore` | Git ignore rules | **Preserved & Updated for ML Checkpoints** |
