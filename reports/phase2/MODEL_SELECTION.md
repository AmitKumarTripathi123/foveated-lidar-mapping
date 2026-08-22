# Phase 2 — Model Selection Report

## 1. Candidate Architecture Evaluation

| Model Family | Representative Architecture | Pros | Cons | Decision |
| :--- | :--- | :--- | :--- | :--- |
| **Point-based (MLP)** | **FoveatedPointSegNet (Selected Baseline)** | Lightweight, real-time (50+ FPS), zero voxelization quantization in feature space, distance-aware feature conditioning | Moderate receptive field | **SELECTED** |
| **Range Image (2D Conv)** | SalsaNext / RangeNet++ | High FPS on dense 64-beam grids | Distorts sparse 40-beam and foveated multi-resolution rings | Candidate for future scale |
| **Voxel-based (Sparse Conv)**| MinkowskiNet / Cylinder3D | Excellent mIoU on large benchmark clusters | High GPU VRAM footprint, compilation overhead | Future Phase 2.5 extension |

## 2. Selected Baseline: FoveatedPointSegNet
- **Parameters**: ~450,000 parameters (~1.8 MB)
- **Input Channels**: 4 (x, y, z, intensity) + 1 distance feature $r = \sqrt{x^2 + y^2}$
- **Output Classes**: 4 navigation classes (0, 1, 2, 3)
- **Inference Speed**: ~8.5 ms / scan on CPU / GPU (~117 FPS)
