# 3D LiDAR Foveated Mapping Data Pipeline (Phase 2 Frozen Interface)

A PyTorch and NumPy implementation of a **distance-aware (foveated) 3D LiDAR data pre-processing and semantic segmentation pipeline** for the **SemanticPOSS** dataset (**Hesai 40-beam LiDAR**, 40 vertical channels, $1800 \times 40$ range image resolution, 10Hz).

---

## Overview

Foveated processing applies **distance-adaptive multi-resolution 3D voxel downsampling**:
- **Near-Field ($0\text{m} - 10\text{m}$)**: $0.05\text{ m}$ ($5\text{ cm}$) voxel size — High resolution for immediate ego-vehicle surroundings.
- **Mid-Field ($10\text{m} - 40\text{m}$)**: $0.15\text{ m}$ ($15\text{ cm}$) voxel size — Mid-range detail tradeoff.
- **Far-Field ($40\text{m} - 100\text{m}$)**: $0.50\text{ m}$ ($50\text{ cm}$) voxel size — Sparse retention for far targets.

### Obstacle-Preserving Voxel Aggregation
Points inside each voxel cell are aggregated according to a strict priority hierarchy so obstacles are never swallowed by ground/ignored points during downsampling:
$$\text{Priority}: \quad \text{dynamic\_object (3)} > \text{static\_obstacle (2)} > \text{non\_drivable (1)} > \text{drivable (0)} > \text{IGNORE (255)}$$

---

## Phase-2 Label Mapping Scheme

Raw SemanticPOSS 32-bit labels (`0` through `22`) map to **4 project super-classes**:

| ID | Class Name | Raw SemanticPOSS Sources | Description |
| :---: | :--- | :--- | :--- |
| `0` | `drivable_terrain` | Drivable (`21`) | Navigable road surfaces |
| `1` | `non_drivable_terrain` | Sidewalks (`19, 20`) | Non-drivable terrain / curbs |
| `2` | `static_obstacle` | Trunk (`8`), Plants (`9`), Signs (`10-12, 18`), Pole (`13`), Trashcan (`14`), Building (`15`), Fence (`17`) | Fixed environmental structures |
| `3` | `dynamic_object` | Pedestrians (`4, 5`), Rider (`6`), Car (`7`) | Dynamic / moving agents |
| `255` | `IGNORE_LABEL` | Unlabeled (`0, 1`), Ground (`22`) | Excluded from loss & evaluation |

---

## Output Contract & Interface (For Model Consumption)

Each frame returns:
- **`points`**: Tensor of shape `(N, 4)` of type `float32` representing `(x, y, z, intensity)`
- **`labels`**: Tensor of shape `(N,)` of type `int64` representing `class_id`

---

## Codebase Structure

```
3d lidar foveated mapping/
├── class_map.py          # Label remapping, color definitions, and loss weight calculations
├── dataset.py            # PyTorch FoveatedLidarDataset, obstacle priority, & DataLoader factory
├── preprocess.py         # Multi-sequence preprocessing & .npy dataset caching tool
├── verify_pipeline.py    # Pipeline verification, priority test suite, & DataLoader tests
├── lidar.py              # Single-scan inspection utility
├── testnumpy.py          # Environment check script
├── .gitignore            # Git ignore rules for data, venv, and binary artifacts
└── README.md             # Project documentation
```

---

## Quick Start

### 1. Requirements
- Python 3.9+
- PyTorch
- NumPy

### 2. Preprocess Dataset
Run the preprocessing pipeline over train and validation splits:
```bash
python3 preprocess.py
```

### 3. Verify Pipeline & DataLoader
Run automated tests including obstacle-preserving label priority verification:
```bash
python3 verify_pipeline.py
```

