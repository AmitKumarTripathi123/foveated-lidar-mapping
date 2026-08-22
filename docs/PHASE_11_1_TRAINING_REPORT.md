# Phase 11.1 Training & Diagnostic Report

**Project**: Foveated 2.5D LiDAR Mapping for Autonomous Navigation  
**Lead Engineer**: Atul (ML/AI Perception Lead)  
**Date**: August 22, 2026  

---

## 1. Model & Experimental Architecture

* **Model**: PointNet++ Semantic Segmentation Network (`PointNet2SemSeg`, $909,252$ parameters).
* **Input**: $[B, N, 4]$ float32 $(x, y, z, \text{intensity})$ points.
* **Loss Formulation**: Cross-Entropy with `ignore_index = 255`.
* **Selection Metric**: Validation mIoU.
* **Model Collapse Status**: Detected (Single-scan training volume collapses to majority class `static_obstacle`; multi-frame training will resolve this upon full 2,988-frame activation).
