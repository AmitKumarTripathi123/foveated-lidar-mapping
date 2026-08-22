# Phase 11.2 Sequence-Level Split Policy

**Project**: Foveated 2.5D LiDAR Mapping for Autonomous Navigation  
**Lead Engineer**: Atul (ML/AI Perception Lead)  
**Date**: August 22, 2026  

---

## 1. Disjoint Sequence Partitioning Scheme

* **Train Set**: Sequences `00`, `01`, `03`, `04`, `05` ($2,488$ scans planned)
* **Validation Set**: Sequence `02` ($500$ scans planned)
* **Test Set**: Independent held-out sequence (**UNAVAILABLE** in workspace)
* **Zero Leakage**: Enforced at the sequence level ($\text{Train} \cap \text{Val} = \emptyset, \text{Train} \cap \text{Test} = \emptyset$).
