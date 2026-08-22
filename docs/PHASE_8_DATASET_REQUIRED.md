# Phase 8 Real Dataset Acquisition & Blockage Statement

**Project**: Foveated 2.5D LiDAR Mapping for Autonomous Navigation  
**Lead Engineer**: Atul (ML/AI Perception Lead)  
**Date**: August 22, 2026  

---

## 1. Executive Summary

Phase 8 implements full multi-frame dataset discovery, leakage-free split allocation, latency benchmarking, and multi-frame mapping integration. However, the physical local workspace currently contains only $1$ labeled representative scan (`000000.bin` in `sequence 00`).

In strict accordance with scientific integrity guidelines:
* **Zero Data Fabrication**: No synthetic frames, cloned scans, or artificial intrascan splits have been created.
* **Status**: **PHASE 8 = DATASET BLOCKED (ENGINEERING PASS — 162/162 TESTS PASS)**.

---

## 2. Required Multi-Sequence Dataset Specifications

| Split Category | Recommended Sequences | Scans Needed | Purpose |
| :--- | :---: | :---: | :--- |
| **TRAIN** | `00`, `01`, `03`, `04`, `05` | $\sim 15,000+$ scans | Representation learning across diverse roadways and obstacles |
| **VALIDATION** | `02` | $\sim 1,500$ scans | Model selection and hyperparameter evaluation |
| **TEST** | `08` | $\sim 4,000$ scans | Completely independent out-of-sample test benchmark |

---

## 3. How to Connect Real Data Once Acquired

1. Download sequence archives (`sequences 00--05`) from SemanticPOSS or SemanticKITTI.
2. Set environment variable: `DATASET_ROOT=/path/to/dataset`.
3. Execute `python scripts/generate_manifest.py` to index the dataset.
4. Execute `python scripts/train_phase7.py --config configs/phase8_training.yaml` to begin multi-frame training.
