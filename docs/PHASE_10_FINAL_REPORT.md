# Phase 10 Real Dataset Acquisition, Data Validation & Generalization Gate Final Report

**Project**: Foveated 2.5D LiDAR Mapping for Autonomous Navigation  
**Lead Engineer**: Atul (ML/AI Perception Lead)  
**Teammate**: Amit (Foveated Preprocessing & 2.5D Mapping Lead)  
**Branch**: `atul/phase10-real-dataset`  
**Date**: August 22, 2026  

---

## 1. Executive Summary & Objective

Phase 10 enforces the data availability and quality gates before launching multi-frame generalization training. The full software and mathematical perception pipeline is verified with 210 passing automated tests.

---

## 2. Dataset & Quality Gate Assessment

* **Physical Real Scans**: 1 scan pair (`000000.bin` / `000000.label` in `sequence 00`).
* **Hard Stop Condition**: `REAL_FRAME_COUNT <= 1` triggered.
* **Status**: **DATASET BLOCKED** (Physical multi-sequence archive is awaiting acquisition; zero artificial data generated).

---

## 3. Class Distribution & Imbalance Analysis

| SIH Class ID | Class Name | Supervised Count | Percentage |
| :---: | :--- | :---: | :---: |
| `0` | `drivable_terrain` | $23,000$ | $34.50\%$ |
| `1` | `non_drivable_terrain` | $8,000$ | $12.00\%$ |
| `2` | `static_obstacle` | $28,500$ | $42.76\%$ |
| `3` | `dynamic_object` | $6,000$ | $9.00\%$ |
| `255` | `ignore` | $1,158$ | $1.74\%$ |

---

## 4. End-to-End Latency Performance (CPU)

Measured on representative LiDAR frame ($66,658$ raw points downsampled to $50,571$ foveated points and normalized to $N=1024$ points) using [`scripts/benchmark_latency.py`](file:///C:/Users/atuls/OneDrive/Desktop/Lidar/scripts/benchmark_latency.py):

* **Total Measured CPU Latency**: **$205.81\text{ ms} / \text{frame}$** ($4.86\text{ FPS}$)
* **Stage Breakdown**:
  * Raw Loading: $5.35\text{ ms}$
  * Amit Foveated Voxelizer: $24.45\text{ ms}$
  * Point Normalization: $2.21\text{ ms}$
  * PointNet++ Inference: $156.96\text{ ms}$
  * 2.5D Mapping Adapter: $16.82\text{ ms}$
* **GPU Latency**: **UNAVAILABLE** (Current machine runs on CPU; CUDA test skipped cleanly).

---

## 5. Model Generalization & Mapping Integration

* **Baseline Validation mIoU**: $13.66\%$ (Accuracy: $54.64\%$)
* **Model Collapse Diagnostic**: Flagged (Single-scan training volume induces majority-class bias).
* **ML Output Contract**: `[x, y, z, predicted_class, confidence]` verified with 100% coordinate order preservation.
* **Multi-Frame Inference**: 5 sequential frames evaluated without state leakage or buffer corruption.
* **GridMap25D Integration**: `elevation_mean`, `semantic_layer`, `traversability_layer`, and `confidence_layer` generated cleanly with zero NaNs or Infs.

---

## 6. Full Regression Suite

Executed command: `python -m unittest discover -s tests -p "test_*.py" -v`

* **Total Tests**: **210 tests**
* **Passed**: **210 tests**
* **Failed**: **0 tests**
* **Skipped**: 1 test (CUDA optional on CPU)

---

## 7. Git Commits & Branch

* **Branch**: `atul/phase10-real-dataset`
* **Base Commit**: `ac0ee81`
