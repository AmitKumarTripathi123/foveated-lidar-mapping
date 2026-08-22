# Phase 11.1 SemanticPOSS Split Policy

**Project**: Foveated 2.5D LiDAR Mapping for Autonomous Navigation  
**Lead Engineer**: Atul (ML/AI Perception Lead)  
**Date**: August 22, 2026  

---

## 1. Disjoint Sequence Partitioning Scheme

* **Train Set**: Sequences `00`, `01`, `03`, `04`, `05` ($2,488$ scans planned)
* **Validation Set**: Sequence `02` ($500$ scans planned)
* **Test Set**: Independent held-out sequence (marked **UNAVAILABLE** until external sequence acquired)
* **Leakage Gate**: Mutually disjoint sequences prevent temporal and point-level cross-split leakage ($\text{Train} \cap \text{Val} = \emptyset$).
