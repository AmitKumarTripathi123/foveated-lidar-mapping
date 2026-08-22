# Foveated 2.5D LiDAR Mapping & Semantic Segmentation for Autonomous Navigation

**Project**: Smart India Hackathon  
**Target System**: Real-time Distance-Aware (Foveated) 3D LiDAR Data Pipeline & Semantic Segmentation  
**Sensor Configuration**: Hesai Pandar40 (40-beam LiDAR), 10 Hz, 100m Range Filtering  

---

## Overview

This repository implements an end-to-end LiDAR perception pipeline for autonomous navigation, featuring:
1. **Distance-Adaptive 3D Voxel Foveation**:
   - **Near-Field (0–10m)**: 0.05m (5 cm) voxel resolution
   - **Mid-Field (10–40m)**: 0.15m (15 cm) voxel resolution
   - **Far-Field (40–100m)**: 0.50m (50 cm) voxel resolution
2. **Authoritative 4-Super-Class Semantic Segmentation**:
   - `0 = drivable_terrain` (Asphalt, road surfaces)
   - `1 = non_drivable_terrain` (Sidewalks, curbs, off-road terrain)
   - `2 = static_obstacle` (Buildings, poles, fences, vegetation, trees)
   - `3 = dynamic_object` (Pedestrians, cars, riders, bicycles, trucks)
   - `255 = IGNORE_LABEL` (Outliers, unlabeled points)
3. **`FoveatedPointSegNet` AI Model**:
   - Lightweight neural point segmentation model (~450k parameters)
   - Distance-conditioned embedding & multi-scale residual spatial feature extraction
   - Generates per-point class probabilities, predicted class, and confidence scores
4. **Interface Contract for Phase 3 2.5D Mapping**:
   - Standardized `SemanticPrediction` dataclass for seamless costmap integration.

---

## Directory Structure

```text
3D Lidar/
├── configs/
│   ├── foveation_default.yaml     # Foveation bands & voxel resolutions
│   ├── phase2.yaml                # AI model & training hyperparameters
│   └── semanticposs_mapping.yaml  # SemanticPOSS raw -> super-class mapping
├── data/
│   └── semanticposs_sequence/     # 40-beam SemanticPOSS sequence scans
├── phase2/
│   ├── dataset.py                 # PyTorch Phase2Dataset adapter (sequence-split)
│   ├── models/
│   │   └── point_seg_net.py       # FoveatedPointSegNet architecture
│   ├── training/
│   │   └── trainer.py             # Model training with class-weighted loss
│   ├── inference/
│   │   └── predictor.py           # Phase2Predictor & SemanticPrediction interface
│   └── metrics/
│       └── semantic_evaluator.py  # mIoU, confusion matrix, & calibration evaluator
├── reports/
│   ├── phase1/                    # Phase 1 verification & benchmarks
│   ├── terrain/                   # SemanticPOSS terrain validation & review tables
│   └── phase2/                    # Phase 2 audit, training, & comparison reports
├── src/                           # Phase 1 LiDAR data foundation & foveation engine
├── tests/                         # Full unit & regression test suite (52 tests)
├── checkpoints/
│   └── best_model.pth             # Trained AI model weights
├── run_phase1_pipeline.py         # Phase 1 end-to-end runner
└── run_phase2_pipeline.py         # Phase 2 end-to-end runner
```

---

## Quick Start

### 1. Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pyyaml numpy scipy pandas matplotlib tabulate
```

### 2. Run Test Suite
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

### 3. Run End-to-End Pipeline
```bash
python3 run_phase2_pipeline.py
```
