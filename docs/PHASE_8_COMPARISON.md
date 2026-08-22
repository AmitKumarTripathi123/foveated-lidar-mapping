# Phase 8 Baseline vs Multi-Frame Infrastructure Comparison

**Project**: Foveated 2.5D LiDAR Mapping for Autonomous Navigation  
**Perception Lead**: Atul  
**Date**: August 22, 2026  

---

## 1. System Capability Progression

| Engineering Metric | Phase 7 Baseline | Phase 8 Infrastructure | Status |
| :--- | :---: | :---: | :---: |
| **Total Automated Tests** | 127 PASS | **162 PASS** | $+35$ Tests (100% Pass Rate) |
| **Dataset Discovery** | Manifest only | **Recursive Multi-Sequence + `DATASET_ROOT`** | Fully Portable & External |
| **Latency Benchmark** | Unmeasured | **Measured: 205.81ms / 4.86 FPS (CPU)** | Stage-by-Stage Breakdown |
| **Multi-Frame Mapping** | Single-scan validation | **Sequential Multi-Frame Ingestion Verified** | `GridMap25D` Active |
| **Scientific Boundary** | Honest single-scan report | **Data-Blockage Explicitly Enforced** | Zero Fake Data |

---

## 2. Experimental Benchmark Summary

| Experiment ID | Dataset Size | Validation mIoU | Overall Accuracy | Model Collapse Warning | Generalization Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Phase 7 Baseline (`phase7_baseline_ce`)** | 1 scan ($N=1024$) | $13.66\%$ | $54.64\%$ | YES (`static_obstacle`) | Single-Scan Overfit |
| **Phase 8 Baseline (`phase8_baseline_ce`)** | 1 scan ($N=1024$) | $13.66\%$ | $54.64\%$ | YES (`static_obstacle`) | Single-Scan Overfit |
| **Phase 8 Production Target** | $\sim 20,000$ scans | $\ge 65.0\%$ (Target) | $\ge 90.0\%$ (Target) | NO | Multi-Sequence Generalization |
