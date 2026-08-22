# Phase 2 — SPVCNN Integration & Comparative Evaluation Report

**Project**: Foveated 2.5D LiDAR Mapping System for Autonomous Navigation (Smart India Hackathon)  
**Lead Authors**: Atul (ML/Perception) & Amit (LiDAR/Foveation)  
**Date**: 2026-08-22  

---

## 1. Overview & Architecture Selection

Phase 2 introduces **SPVCNN (Sparse Point-Voxel Convolutional Neural Network)** (Tang et al., ECCV 2020) as the primary ML perception engine for semantic understanding. 
SPVCNN couples high-resolution point-wise feature branches with voxel coordinate convolutions via pure PyTorch scatter-add operations (`torch.index_add_`). This enables high computational throughput on CPU (macOS ARM64) and GPU (CUDA) without requiring external C++ compiler dependencies.

```text
Input Point Cloud (N, 4)
           │
           ├──────────────────────────────┬──────────────────────────────┐
           ▼                                                             ▼
   Point Branch (MLP)                                            Voxel Branch (3D Voxel)
(High-resolution spatial geometry)                          (Low-frequency regional context)
           │                                                             │
           └──────────────────────────────┬──────────────────────────────┘
                                          ▼
                                   SPVConv Fusion
                               (Point + Voxel + Res)
                                          │
                                          ▼
                                  Logits (N, 19)
                                          │
                                          ▼
                                 SPVCNNLabelAdapter
                              (19 Native -> 4 SIH Classes)
                                          │
                                          ▼
                                SemanticPrediction
                                          │
                                          ▼
                                MLToMappingAdapter
                                          │
                                          ▼
                                      GridMap25D
```

---

## 2. Model & Checkpoint Specifications

| Attribute | FoveatedPointSegNet (Fallback) | SPVCNN (Primary) |
| :--- | :--- | :--- |
| **Model Class** | `FoveatedPointSegNet` | `SPVCNN` (`PointBranch` + `VoxelSpatialBranch` + `SPVConvBlock`) |
| **Parameters** | 451,460 parameters (~1.8 MB) | **136,979 parameters (~0.58 MB)** |
| **Quantization Resolution** | N/A (continuous MLP) | **0.05m (5 cm) 3D Voxel Grid** |
| **Checkpoint** | `checkpoints/best_model.pth` | `checkpoints/spvcnn_pretrained.pt` (SHA256: `cb1a6f44...`) |
| **Source Dataset** | SemanticPOSS sequence 01 | SemanticKITTI (19 classes) $\to$ SIH 4-Class Adapter |

---

## 3. End-to-End Latency & Throughput Benchmark

Evaluated on 5 real SemanticPOSS frames ($40,000$ points/frame):

| Metric | FoveatedPointSegNet Baseline | SPVCNN Pipeline | Speedup / Reduction |
| :--- | :---: | :---: | :---: |
| **LiDAR Loading** | 1.25 ms | 1.40 ms | — |
| **Preprocessing** | 2.83 ms | 2.80 ms | — |
| **ML Semantic Inference** | **185.60 ms** | **107.60 ms** | **1.73x Speedup** |
| **2.5D Grid Generation** | 167.22 ms | 164.52 ms | 1.02x |
| **Visualization Prep** | 95.39 ms | 94.70 ms | 1.01x |
| **Total End-to-End Latency** | **452.31 ms** | **371.03 ms** | **1.22x Speedup** |
| **Throughput (FPS)** | **2.21 FPS** | **2.70 FPS** | **+22.2%** |
| **Resident RAM (RSS)** | 719.50 MB | **552.36 MB** | **-23.2% RAM** |
| **CPU Utilization** | 178.1% | **143.5%** | **-19.4% CPU** |

---

## 4. SemanticPOSS Accuracy & mIoU Evaluation

| Metric | FoveatedPointSegNet (Fine-Tuned) | SPVCNN (Zero-Shot Pretrained) | Notes |
| :--- | :---: | :---: | :--- |
| **Overall Accuracy** | **74.44%** | 21.03% | Fine-tuning head on SemanticPOSS multi-sequence data is recommended. |
| **mIoU** | **50.77%** | 8.29% | Zero-shot transfer from SemanticKITTI. |
| **Drivable Terrain (0) IoU** | 19.41% | 0.80% | Drivable precision: 100.0% |
| **Non-Drivable Terrain (1) IoU**| 54.74% | 0.00% | Under-represented prior to fine-tuning. |
| **Static Obstacle (2) IoU** | **88.35%** | **27.07%** | Precision: 47.42%, Recall: 38.67% |
| **Dynamic Object (3) IoU** | **40.58%** | **5.27%** | Dynamic Recall: **81.50%** |

---

## 5. Phase 2 Model Factory & Integration Contract

`Phase2Predictor` in `phase2/inference/predictor.py` provides runtime model switching:

```python
# SPVCNN Primary (Default)
predictor = Phase2Predictor(model_type="spvcnn", model_path="checkpoints/spvcnn_pretrained.pt")

# Fallback
predictor = Phase2Predictor(model_type="foveated_pointnet", model_path="checkpoints/best_model.pth")
```

Both models return the exact frozen `SemanticPrediction` dataclass and seamlessly project into `GridMap25D` via `MLToMappingAdapter`.
