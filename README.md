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
3. **PointNet++ & SPVCNN Architectures**:
   - Standardized 3D neural point segmentation networks (138,514 parameters, 53.59% Val mIoU)
   - Distance-conditioned embedding & multi-scale residual spatial feature extraction
   - Predicts per-point classes and calibrated confidence scores
4. **Interface Contract for 2.5D Mapping**:
   - Standardized `GridMap25D` output layers: `elevation_mean`, `semantic_layer`, `traversability_layer`, and `confidence_layer`.
5. **Real-Time 3D Web Visualization & SIH Workstation (`frontend/` & `backend/`)**:
   - 100% Pure JavaScript / TypeScript web platform (Next.js 14 + Three.js + Node.js Express + WebSockets).
   - 6 Dedicated 3D View Modes + 8-Stage Research Pipeline + 6-Step Judge Presentation Mode.
   - Live quantitative benchmark comparison proving ~82.8% memory reduction and 4.6x speedup.

---

## Web Dashboard & 3D Visualization Setup

### 1. Start Backend Streaming Engine (`/backend`)
```bash
cd backend
npm install
npm run build
node dist/server.js
```
* Backend starts at `http://localhost:8000` with WebSocket stream on `ws://localhost:8000/ws/stream`.

### 2. Start Frontend 3D Workstation (`/frontend`)
```bash
cd frontend
npm install
npm run build
npx next start -p 3000
```
* Open **`http://localhost:3000`** in your browser.

---

## Dataset Setup & Activation Guide

To activate the full multi-frame SemanticPOSS dataset (6 sequences, 2,988 frame pairs):

### 1. Download & Directory Layout
Download the SemanticPOSS dataset from the official repository and extract it under `dataset/` or an external directory:
```text
dataset/
└── sequences/
    ├── 00/
    │   ├── velodyne/*.bin
    │   └── labels/*.label
    ├── 01/
    ├── 02/
    ├── 03/
    ├── 04/
    └── 05/
```

### 2. Run Dataset Forensic Audit Tool
```bash
python scripts/audit_semanticposs.py --root dataset
```

### 3. Verify Pipeline
```bash
python verify_pipeline.py --dataset-root dataset
```

### 4. Run Automated Test Suite
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## Quick Start (ML Pipeline)

### 1. Setup Python Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pyyaml numpy scipy pandas matplotlib tabulate
```

### 2. Run End-to-End Pipeline
```bash
python3 run_phase2_pipeline.py
```
