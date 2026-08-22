# Phase 10 Multi-Frame Training Configuration & Diagnostic Report

**Project**: Foveated 2.5D LiDAR Mapping for Autonomous Navigation  
**Lead Engineer**: Atul (ML/AI Perception Lead)  
**Date**: August 22, 2026  

---

## 1. Experimental Training Configurations

* **Model Architecture**: PointNet++ Semantic Segmentation (`PointNet2SemSeg`, $909,252$ parameters).
* **Optimizer**: Adam ($lr = 0.005, \text{weight decay} = 10^{-4}$).
* **Learning Rate Scheduler**: Cosine Annealing.
* **Loss Functions**: Plain Cross-Entropy, Inverse-Frequency Weighted Cross-Entropy.
* **Ignore Index**: $255$ (Strictly excluded from supervised loss and metric computations).

---

## 2. Model Collapse Analysis

On single-scan training data, the model converges to majority-class prediction:
* **Validation mIoU**: $13.66\%$
* **Overall Accuracy**: $54.64\%$
* **Collapse Diagnosis**: `MODEL_COLLAPSE_WARNING = YES` (All predictions belong to class `2` `static_obstacle`).
* **Root Cause**: Low intra-class variance in a single LiDAR scan.
