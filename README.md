# Foveated 2.5D LiDAR Mapping & Semantic Segmentation for Autonomous Navigation

**Project**: Smart India Hackathon (SIH) — Problem Statement PS 26130  
**Architecture Version**: Phase 18 Canonical Frozen Architecture  
**Sensor Model**: Hesai Pandar40 (40-beam LiDAR) / Velodyne HDL-64E, 10 Hz, 100m Range  
**Single Source of Truth Configuration**: `configs/system_config.yaml`  

---

## 1. System Architecture Overview

This repository implements the canonical end-to-end foveated perception and 2.5D grid mapping architecture:

```text
Raw LiDAR (.bin / sensor_msgs/PointCloud2)
                 │
                 ▼
       RangeFilter (0.5m - 100.0m)
                 │
                 ▼
  FoveatedVoxelSampler (3-Zone Distance Tiers)
  ├── Near Zone (0–10m @ 0.05m / Level 0)
  ├── Mid Zone  (10–40m @ 0.15m / Level 1)
  └── Far Zone  (40–100m @ 0.50m / Level 2)
                 │
                 ▼
       SPVCNN CUDA Tensor-Core Inference
  (136,004 params | 4-Class SIH Ontology)
                 │
                 ▼
   HierarchicalFoveatedGridEngine (src/core/foveated_grid.py)
                 │
                 ▼
       GridMap25D (500x500 cells @ 0.20m)
  ├── Elevation (Mean, Min, Max)
  ├── Semantic Layer (Dominant Class)
  ├── Traversability (+1.0 Go, 0.0 Stop, -1.0 Off-Road)
  ├── Confidence Layer (Mean Score)
  └── Point Density Layer
                 │
        ┌────────┴────────┐
        ▼                 ▼
Live Dashboard HUD    ROS2 Publishers
(visualization/)      (ros2_ws/)
```

### Canonical Foveation Geometry (Frozen):
* **Near Zone ($0.0\text{m} \le d < 10.0\text{m}$)**: **$0.05\text{m}$ ($5\text{ cm}$)** fine voxel resolution.
* **Mid Zone ($10.0\text{m} \le d < 40.0\text{m}$)**: **$0.15\text{m}$ ($15\text{ cm}$)** balanced resolution.
* **Far Zone ($40.0\text{m} \le d \le 100.0\text{m}$)**: **$0.50\text{m}$ ($50\text{ cm}$)** coarse resolution.
* **Outer Range ($d > 100.0\text{m}$)**: Dropped defensively.

### Authoritative 4-Class SIH Semantic Ontology:
* `0 = drivable_terrain` (Asphalt, roads, driveable surfaces)
* `1 = non_drivable_terrain` (Sidewalks, curbs, off-road terrain)
* `2 = static_obstacle` (Buildings, poles, fences, vegetation, trees)
* `3 = dynamic_object` (Vehicles, pedestrians, cyclists)
* `255 = ignore` (Unlabeled, laser dropouts, noise)

---

## 2. Canonical Directory Layout

```text
foveated-lidar-mapping/
│
├── configs/
│   ├── system_config.yaml      ⭐ SINGLE SOURCE OF TRUTH
│   ├── model.yaml              # Model reference config
│   └── ros2.yaml               # ROS2 topics & queue config
│
├── src/
│   ├── core/
│   │   ├── types.py            # PointXYZL, CellKey, GridCell, ElevationCell
│   │   ├── lidar_loader.py     # Robust .bin / byte LiDAR loader
│   │   ├── range_filter.py     # Range boundary & NaN filter
│   │   ├── hierarchy.py        # Multiresolution CellKey & spatial indexing
│   │   ├── foveated_grid.py    # Hierarchical 2.5D Grid Engine
│   │   └── traversability.py   # Continuous traversability scoring
│   │
│   ├── inference/
│   │   ├── predictor.py        # SPVCNN inference with SHA256 validation
│   │   ├── postprocess.py      # O(N) validation & DTO formatting
│   │   └── pipeline.py         # Canonical FoveatedPipeline orchestrator
│   │
│   └── visualization/
│       └── dashboard.py        # Canonical multi-panel & HTML dashboard
│
├── cpp/                        # C++ accelerated 3-zone grid engine
├── ros2_ws/                    # ROS2 PointCloud2 package & replay node
├── benchmarks/                 # Standardized benchmark harness
└── reports/                    # Generated benchmark scorecards & figures
```

---

## 3. Quick Start & Execution Guide

### 1. Canonical End-to-End Pipeline
```python
from src.inference.pipeline import FoveatedPipeline

pipeline = FoveatedPipeline(config_path="configs/system_config.yaml")
result = pipeline.run("dataset/sequences/02/velodyne/000001.bin")

print(f"Total Latency: {result.total_latency_ms} ms")
print(f"Grid Map Shape: {result.grid_map.grid_shape}")
```

### 2. Run Canonical Benchmark
```bash
python benchmarks/benchmark_canonical.py --config configs/system_config.yaml --frames 100
```

### 3. Generate Live Visualization Dashboard
```bash
python visualization/dashboard.py
```

### 4. Run Automated Test Suite
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
