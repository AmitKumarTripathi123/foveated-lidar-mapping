# Foveated 2.5D LiDAR Mapping for Autonomous Navigation

[![Tests](https://img.shields.io/badge/Tests-96%20Passed-brightgreen)](tests/)
[![Architecture](https://img.shields.io/badge/Model-PointNet%2B%2B-blue)](ml/models/)
[![Ontology](https://img.shields.io/badge/Ontology-SIH%204--Class-orange)](ml/data/)

An autonomous vehicle perception system integrating **3-Zone Foveated Voxel Downsampling** with a **PointNet++ 3D Semantic Segmentation** neural network to generate high-resolution, low-latency 2.5D elevation and semantic occupancy grids.

---

## 1. Pipeline Overview

```text
RAW LiDAR (.bin) ──> Preprocessing ──> Amit's Foveated Voxelizer ──> SIH 4-Class Remapper
                                                                            │
                                                                            ▼
Amit's Frozen Contract <── PointNet2Predictor <── PointNet++ <── PyTorch Dataset
[x,y,z, class, conf]
```

### 3-Zone Foveated Voxelization:
* **Near-Field ($0\text{--}10\text{m}$)**: Voxel size $= 0.05\text{m}$ (dense spatial fidelity)
* **Mid-Field ($10\text{--}40\text{m}$)**: Voxel size $= 0.15\text{m}$ (balanced coverage)
* **Far-Field ($40\text{--}100\text{m}$)**: Voxel size $= 0.50\text{m}$ (computational efficiency)

---

## 2. Quickstart & CLI Commands

```bash
# 1. Discover dataset & generate integrity audit
python scripts/generate_manifest.py

# 2. Preprocess & cache 3-zone foveated point clouds
python scripts/preprocess_foveated.py

# 3. Train PointNet++ Baseline (Experiment A)
python scripts/train.py --experiment baseline_ce --epochs 10 --num-points 1024

# 4. Train Class-Weighted Model (Experiment B)
python scripts/train.py --experiment weighted_ce --epochs 10 --num-points 1024 --weighted-loss

# 5. Train Augmented Model (Experiment C)
python scripts/train.py --experiment weighted_ce_aug --epochs 10 --num-points 1024 --weighted-loss --augmentation

# 6. Compare Experiments
python scripts/compare_experiments.py --experiments baseline_ce weighted_ce weighted_ce_aug

# 7. Evaluate Checkpoint & Verify Amit Contract
python scripts/evaluate.py --checkpoint experiments/baseline_ce/best_checkpoint.pt

# 8. Run Full 96-Test Suite
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## 3. Project Documentation

* [`docs/INTEGRATION_PLAN.md`](docs/INTEGRATION_PLAN.md): Architecture blueprint and module ownership.
* [`docs/TEAM_CONTRACT.md`](docs/TEAM_CONTRACT.md): Amit / Atul domain boundaries and data contracts.
* [`docs/PHASE_5_FINAL_REPORT.md`](docs/PHASE_5_FINAL_REPORT.md): Full Phase 5 experimental validation and benchmark results.
