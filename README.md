# Foveated 2.5D LiDAR Mapping for Autonomous Navigation

[![Tests](https://img.shields.io/badge/Tests-96%20Passed-brightgreen)](tests/)
[![Architecture](https://img.shields.io/badge/Model-PointNet%2B%2B-blue)](ml/models/)
[![Ontology](https://img.shields.io/badge/Ontology-SIH%204--Class-orange)](ml/data/)

A complete autonomous vehicle perception system integrating **3-Zone Distance-Aware Foveated Voxel Downsampling** with a **PointNet++ 3D Semantic Segmentation** neural network to generate high-resolution, low-latency 2.5D elevation and semantic occupancy grids.

---

## 1. Overview & Architecture

```text
RAW LiDAR (.bin) ──> Preprocessing ──> Amit's Foveated Voxelizer ──> SIH 4-Class Remapper
                                                                            │
                                                                            ▼
Amit's Frozen Contract <── PointNet2Predictor <── PointNet++ <── PyTorch Dataset
[x,y,z, class, conf]
```

### 3-Zone Foveated Voxelization
Foveated processing applies **distance-adaptive multi-resolution 3D voxel downsampling**:
- **Near-Field ($0\text{m} - 10\text{m}$)**: $0.05\text{ m}$ ($5\text{ cm}$) voxel size — High resolution for immediate ego-vehicle surroundings.
- **Mid-Field ($10\text{m} - 40\text{m}$)**: $0.15\text{ m}$ ($15\text{ cm}$) voxel size — Mid-range detail tradeoff.
- **Far-Field ($40\text{m} - 100\text{m}$)**: $0.50\text{ m}$ ($50\text{ cm}$) voxel size — Sparse retention for far targets.

### Obstacle-Preserving Voxel Aggregation
Points inside each voxel cell are aggregated according to a strict priority hierarchy so obstacles are never swallowed by ground/ignored points during downsampling:
$$\text{Priority}: \quad \text{dynamic\_object (3)} > \text{static\_obstacle (2)} > \text{non\_drivable (1)} > \text{drivable (0)} > \text{IGNORE (255)}$$

---

## 2. Phase-2 Label Mapping Scheme

Raw SemanticPOSS 32-bit labels (`0` through `22`) map to **4 project super-classes**:

| ID | Class Name | Raw SemanticPOSS Sources | Description |
| :---: | :--- | :--- | :--- |
| `0` | `drivable_terrain` | Drivable (`21`) | Navigable road surfaces |
| `1` | `non_drivable_terrain` | Sidewalks (`19, 20`) | Non-drivable terrain / curbs |
| `2` | `static_obstacle` | Trunk (`8`), Plants (`9`), Signs (`10-12, 18`), Pole (`13`), Trashcan (`14`), Building (`15`), Fence (`17`) | Fixed environmental structures |
| `3` | `dynamic_object` | Pedestrians (`4, 5`), Rider (`6`), Car (`7`) | Dynamic / moving agents |
| `255` | `IGNORE_LABEL` | Unlabeled (`0, 1`), Ground (`22`) | Excluded from loss & evaluation |

---

## 3. Data Interface & Output Contract

Each frame returns:
- **`points`**: Tensor of shape `(N, 4)` of type `float32` representing `(x, y, z, intensity)`
- **`labels`**: Tensor of shape `(N,)` of type `int64` representing `class_id`

---

## 4. Codebase Structure

```
3d lidar foveated mapping/
├── class_map.py          # Label remapping, color definitions, and loss weight calculations
├── dataset.py            # PyTorch FoveatedLidarDataset, obstacle priority, & DataLoader factory
├── preprocess.py         # Multi-sequence preprocessing & .npy dataset caching tool
├── verify_pipeline.py    # Pipeline verification, priority test suite, & DataLoader tests
├── ml/                   # PointNet++ model architecture, trainer, and predictor
├── configs/              # YAML configurations (dataset, model, training, label mapping)
├── scripts/              # Training, evaluation, and experiment comparison CLI tools
├── experiments/          # Model checkpoints (best_checkpoint.pt), metrics, and training logs
├── docs/                 # Architectural documentation and team contracts
└── tests/                # Unit test suite (96 tests passed)
```

---

## 5. Quickstart & CLI Commands

```bash
# 1. Verify Foveated Pipeline & Obstacle Priority Test
OMP_NUM_THREADS=1 python3 verify_pipeline.py

# 2. Discover dataset & generate integrity audit
python scripts/generate_manifest.py

# 3. Preprocess & cache 3-zone foveated point clouds
python scripts/preprocess_foveated.py

# 4. Train PointNet++ Baseline (Experiment A)
python scripts/train.py --experiment baseline_ce --epochs 10 --num-points 1024

# 5. Train Class-Weighted Model (Experiment B)
python scripts/train.py --experiment weighted_ce --epochs 10 --num-points 1024 --weighted-loss

# 6. Compare Experiments & Evaluate Checkpoint
python scripts/evaluate.py --checkpoint experiments/baseline_ce/best_checkpoint.pt

# 7. Run Full Test Suite
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## 6. Project Documentation

* [`docs/INTEGRATION_PLAN.md`](docs/INTEGRATION_PLAN.md): Architecture blueprint and module ownership.
* [`docs/TEAM_CONTRACT.md`](docs/TEAM_CONTRACT.md): Domain boundaries and data contracts.
* [`docs/PHASE_5_FINAL_REPORT.md`](docs/PHASE_5_FINAL_REPORT.md): Full experimental validation and benchmark results.
