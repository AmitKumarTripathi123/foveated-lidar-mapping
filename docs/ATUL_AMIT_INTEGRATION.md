# Atul + Amit Pipeline Integration Report

**Canonical Repository**: `https://github.com/AmitKumarTripathi123/foveated-lidar-mapping`  
**Integration Branch**: `integration/atul-phase1-5`  
**Engineer**: Atul (ML/AI Perception Lead)  
**Lead Owner**: Amit (Foveated Preprocessing & 2.5D Mapping Lead)  

---

## 1. Summary of Integrated Components

### What Atul Added:
1. **Phase 1 LiDAR Loader (`ml/data/dataset.py`)**: Validated parsing of `.bin` ($[N, 4]$ `float32`) and `.label` ($[N]$ `uint32`), NaN/Inf detection, point-label alignment assertions.
2. **Phase 2 Preprocessing & Sampling (`ml/data/preprocessing.py`)**: Invalid point removal, range cropping, deterministic/random sampling, intensity normalization.
3. **Phase 2 Foveated Adapter & PyTorch Dataset (`ml/data/amit_adapter.py`, `ml/data/foveated_dataset.py`)**: Standardized wrapper consuming Amit''s 3-zone voxelization and normalizing point count to fixed $N=1024$ / $16384$ for PointNet++.
4. **Phase 3 SIH 4-Class Semantic Ontology Remapper (`ml/data/label_mapping.py`)**: Vectorized lookup table remapping SemanticPOSS / SemanticKITTI raw labels into $\{0, 1, 2, 3, 255\}$.
5. **Phase 4 PointNet++ 3D Semantic Segmentation (`ml/models/pointnet2.py`)**: Hierarchical Set Abstraction (SA) and Feature Propagation (FP) network ($909,252$ parameters).
6. **Phase 4 Frozen ML $\to$ Mapping Contract (`ml/models/predictor.py`)**: `PointNet2Predictor` returning `[x, y, z, predicted_class, confidence]` with exact point order preservation.
7. **Phase 5 Training & Validation Engine (`ml/training/`)**: Plain & Class-Weighted Cross-Entropy (`ignore_index=255`), $4 \times 4$ confusion matrix, per-class IoU, mIoU, precision, recall, validation-mIoU checkpointing, 3D training augmentation.
8. **Automated Test Suite (`tests/`)**: 96 tests across Phases 1–5 and full pipeline integration.
9. **Manifest & Audit Generator (`ml/data/manifest.py`, `scripts/generate_manifest.py`)**: Discovers sequences, asserts disjointness, produces `data_manifest.json` and `dataset_audit.md`.

### What of Amit''s Code Was Preserved:
1. **`dataset.py` (Root)**: Amit''s `FoveatedLidarDataset`, `_range_aware_downsample` (3 bands: $0\text{--}10\text{m} \to 0.05\text{m}$, $10\text{--}40\text{m} \to 0.15\text{m}$, $40\text{--}100\text{m} \to 0.50\text{m}$), `collate_fn_foveated`, `create_dataloader` — **100% Preserved**.
2. **`class_map.py` (Root)**: Amit''s `PROJECT_CLASSES`, `POSS_RAW_CLASSES`, `POSS_CLASS_REMAP`, `get_class_colors`, `compute_class_weights` — **100% Preserved**.
3. **`preprocess.py` (Root)**: Amit''s offline preprocessing and caching script — **100% Preserved**.
4. **`verify_pipeline.py` (Root)**: Amit''s pipeline verification script — **100% Preserved**.
5. **`lidar.py` (Root)**: Amit''s point cloud inspection script — **100% Preserved**.
6. **`testnumpy.py` (Root)**: Environment verification — **100% Preserved**.

---

## 2. Interface Contracts

### Input Contract to ML:
* Point Cloud: $[N, 4]$ NumPy/PyTorch array (`float32`), $[x, y, z, \text{intensity}]$
* Labels: $[N]$ array (`uint8`/`int64`), semantic values $\in \{0, 1, 2, 3, 255\}$.

### Output Contract from ML (Consumable by Mapping Module):
```python
from ml.models.predictor import PointNet2Predictor

predictor = PointNet2Predictor(model=trained_model, device="cpu")
result = predictor.predict(points)  # points: (N, 4)

# Result dictionary contains:
# result["xyz"]             : (N, 3) float32 in exact input point order
# result["predicted_class"] : (N,) int64 strictly in {0, 1, 2, 3}
# result["confidence"]      : (N,) float32 strictly in [0.0, 1.0]
```

---

## 3. How to Run Tests

```bash
# Run all 96 unit and integration tests
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## 4. How to Run the ML Pipeline

```bash
# 1. Discover dataset & generate audit report
python scripts/generate_manifest.py

# 2. Preprocess and cache foveated scans
python scripts/preprocess_foveated.py

# 3. Train PointNet++ Baseline (Experiment A)
python scripts/train.py --experiment baseline_ce --epochs 10 --num-points 1024

# 4. Train Class-Weighted PointNet++ (Experiment B)
python scripts/train.py --experiment weighted_ce --epochs 10 --num-points 1024 --weighted-loss

# 5. Train with 3D Augmentation (Experiment C)
python scripts/train.py --experiment weighted_ce_aug --epochs 10 --num-points 1024 --weighted-loss --augmentation

# 6. Compare Experiments
python scripts/compare_experiments.py --experiments baseline_ce weighted_ce weighted_ce_aug

# 7. Evaluate Checkpoint & Verify Contract
python scripts/evaluate.py --checkpoint experiments/baseline_ce/best_checkpoint.pt
```

---

## 5. Known Limitations

* **Single Representative Frame**: Local workspace currently contains 1 representative scan pair in `dataset/sequences/00/`. For full production training in Phase 6, multiple distinct sequence drives should be added.
* **CPU Execution**: Model runs on CPU in FP32 mode at $N=1024$ points; GPU acceleration with FP16/AMP will enable scaling to $N=16,384$ points.
