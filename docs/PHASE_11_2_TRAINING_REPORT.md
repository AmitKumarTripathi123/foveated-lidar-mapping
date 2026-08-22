# Phase 11.2 Multi-Frame Training & Diagnostic Report

**Project**: Foveated 2.5D LiDAR Mapping for Autonomous Navigation  
**Lead Engineer**: Atul (ML/AI Perception Lead)  
**Date**: August 22, 2026  

---

## 1. Experimental Training Configurations

* **Model Architecture**: PointNet++ Semantic Segmentation (`PointNet2SemSeg`, $909,252$ parameters).
* **Loss Function**: Plain Cross-Entropy with `ignore_index = 255`.
* **Primary Selection Metric**: Validation mIoU.
* **Model Collapse Status**: Single-scan baseline produces $13.66\%$ mIoU and majority-class bias towards `static_obstacle`.
