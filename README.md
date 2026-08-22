# 3D LiDAR Foveated Mapping Data Pipeline

A PyTorch and NumPy implementation of a **distance-aware (foveated) 3D LiDAR data pre-processing and semantic segmentation pipeline** for the **SemanticPOSS** dataset (40-beam LiDAR).

---

## Overview

Foveated processing draws inspiration from human vision by applying **distance-adaptive multi-resolution voxel downsampling**:
- **Near-Field ($0\text{m} - 10\text{m}$)**: $0.05\text{ m}$ ($5\text{ cm}$) voxel size — High resolution for immediate ego-vehicle surroundings.
- **Mid-Field ($10\text{m} - 40\text{m}$)**: $0.15\text{ m}$ ($15\text{ cm}$) voxel size — Mid-range detail tradeoff.
- **Far-Field ($40\text{m} - 100\text{m}$)**: $0.50\text{ m}$ ($50\text{ cm}$) voxel size — Sparse retention for far targets.

This strategy preserves geometric detail near the vehicle while dramatically reducing computational and memory overhead for far-field points.

---

## Class Mapping Scheme

Raw SemanticPOSS 32-bit labels (`0` through `22`) are mapped to **4 project super-classes**:

| ID | Class Name | Description |
| :---: | :--- | :--- |
| `0` | `drivable_terrain` | Navigable road surfaces |
| `1` | `non_drivable_terrain` | Reserved (sidewalks/curbs) |
| `2` | `static_obstacle` | Buildings, poles, traffic signs, fences, vegetation |
| `3` | `dynamic_object` | Pedestrians, riders, cars, bicycles |
| `255` | `IGNORE_LABEL` | Unlabeled points / unmapped IDs (excluded from loss & eval) |

---

## Code Base Structure

```
3d lidar foveated mapping/
├── class_map.py          # Label remapping, color definitions, and loss weight calculations
├── dataset.py            # PyTorch FoveatedLidarDataset, batch collation, & DataLoader factory
├── preprocess.py         # Multi-sequence preprocessing & .npy dataset caching tool
├── verify_pipeline.py    # Pipeline verification & DataLoader test suite
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
- Open3D *(optional)*

### 2. Preprocess Dataset
Run the preprocessing pipeline over train and validation splits:
```bash
python3 preprocess.py
```

### 3. Verify Pipeline & DataLoader
Test raw frame loading, PyTorch batch collation, and metadata integrity:
```bash
python3 verify_pipeline.py
```
