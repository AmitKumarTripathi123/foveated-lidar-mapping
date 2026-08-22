# Phase 11.1 Dataset Identity & Ontology Audit

**Project**: Foveated 2.5D LiDAR Mapping for Autonomous Navigation  
**Lead Engineer**: Atul (ML/AI Perception Lead)  
**Date**: August 22, 2026  

---

## 1. Dataset Identification

* **Dataset Identity**: **SemanticPOSS / SemanticKITTI Hybrid Support**
* **Confidence**: **HIGH**
* **Evidence**:
  * The canonical codebase originates from Amit''s SemanticPOSS pipeline (Pandora 40-beam LiDAR).
  * The physical local sample scan (`dataset/sequences/00/velodyne/000000.bin`) contains SemanticKITTI-compatible raw IDs (`40=road, 48=sidewalk, 50=building, 70=vegetation, 10=car`).
  * The perception system is equipped with dedicated remappers for both SemanticPOSS ([`ml/data/semanticposs_label_mapping.py`](file:///C:/Users/atuls/OneDrive/Desktop/Lidar/ml/data/semanticposs_label_mapping.py)) and SemanticKITTI ([`ml/data/authoritative_label_mapping.py`](file:///C:/Users/atuls/OneDrive/Desktop/Lidar/ml/data/authoritative_label_mapping.py)).
