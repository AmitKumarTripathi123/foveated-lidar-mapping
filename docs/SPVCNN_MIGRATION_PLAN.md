# SPVCNN Migration Plan & Architectural Design Document

**Project**: Foveated 2.5D LiDAR Mapping for Autonomous Navigation  
**Perception Lead**: Atul  
**Mapping & Preprocessing Lead**: Amit  
**Date**: August 22, 2026  

---

## 1. Executive Summary

This document establishes the architectural migration strategy for upgrading the primary 3D LiDAR perception model from **PointNet++** (dense set-abstraction baseline) to **SPVCNN** (Sparse Point-Voxel Convolutional Neural Network, Tang et al., ECCV 2020).

PointNet++ is retained in the codebase as a verified legacy/baseline model (`pointnet2_legacy`), ensuring backward compatibility and baseline comparisons.

---

## 2. Current Architecture & Data Flow

```text
Raw LiDAR (.bin / .label)
        │
        ▼
Amit 3-Zone Distance-Adaptive Foveated Voxelizer (0.05m, 0.15m, 0.50m)
        │
        ▼
Point Normalization (N = 1024 or 16,384)
        │
        ▼
PointNet2SemSeg (Dense Set Abstraction & Feature Propagation, 909,252 params)
        │
        ▼
PointNet2Predictor -> Output Dict {"xyz", "predicted_class", "confidence"}
        │
        ▼
MLToMappingAdapter -> GridMap25D (elevation, semantic, traversability, confidence)
```

---

## 3. SPVCNN Integration & Replacement Point

SPVCNN replaces PointNet++ in the perception inference stage by coupling high-resolution point-wise feature extraction with 3D sparse voxel convolutions:

```text
Raw LiDAR (.bin / .label)
        │
        ▼
Amit 3-Zone Distance-Adaptive Foveated Voxelizer (0.05m, 0.15m, 0.50m)
        │
        ▼
SPVCNNInputAdapter (Coordinate Quantization, Voxel-Point Index Mapping)
        │
        ▼
SPVCNN (Point-Voxel Sparse Convolution Backbone)
        │
        ▼
Point-Level Feature Recovery & Softmax Calibration
        │
        ▼
SPVCNNLabelAdapter (Native Model Classes -> SIH 4-Class Ontology)
        │
        ▼
SPVCNNPredictor (Frozen Output Contract: [x, y, z, predicted_class, confidence])
        │
        ▼
MLToMappingAdapter -> GridMap25D
```

---

## 4. Format & Contract Invariant Guarantees

| Invariant | Specification | Guarantee Mechanism |
| :--- | :--- | :--- |
| **Point Count & Ordering** | $(N, 3)$ float32 | `SPVCNNInputAdapter` preserves original XYZ coordinates and creates deterministic `point_to_voxel_index` & `voxel_to_point_index` mappings. |
| **Class Label Range** | $\text{classes} \in \{0, 1, 2, 3\}$ | `SPVCNNLabelAdapter` maps native predictions to SIH super-classes with unmapped fallback to `255` (ignore). |
| **Confidence Range** | $\text{conf} \in [0.0, 1.0]$ | $\max(\text{softmax}(\text{logits}))$ computed on valid logits. |
| **Foveation Alignment** | 3-Zone obstacle priority | Amit's foveated voxelizer is executed before the SPVCNN input adapter. |
| **Mapping Compatibility** | `GridMap25D` layers | `MLToMappingAdapter` consumes the exact dictionary format. |

---

## 5. Migration Strategy & Milestone Roadmap

1. **Milestone 1**: Pretrained SPVCNN checkpoint selection, audit, and verification (`docs/SPVCNN_CHECKPOINT_AUDIT.md`).
2. **Milestone 2**: Core SPVCNN architecture implementation in PyTorch with CPU & CUDA support (`ml/models/spvcnn.py`).
3. **Milestone 3**: SPVCNN input adapter with bidirectional point-voxel index preservation (`ml/data/spvcnn_adapter.py`).
4. **Milestone 4**: Label adapter mapping native classes to the SIH 4-class ontology (`ml/models/spvcnn_label_adapter.py`).
5. **Milestone 5**: High-level predictor implementing the frozen ML contract (`ml/models/spvcnn_predictor.py`).
6. **Milestone 6**: End-to-end CLI tool and latency benchmarking (`scripts/infer_spvcnn.py`).
7. **Milestone 7**: Comprehensive automated unit & integration test suites (`tests/test_spvcnn*.py`).
8. **Milestone 8**: Final integration report (`docs/SPVCNN_INTEGRATION_REPORT.md`).
