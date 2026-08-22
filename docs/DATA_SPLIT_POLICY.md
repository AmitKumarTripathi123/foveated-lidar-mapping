# Authoritative Dataset Partition & Leakage Prevention Policy

**Project**: Foveated 2.5D LiDAR Mapping for Autonomous Navigation  
**Perception Lead**: Atul  
**Date**: August 22, 2026  

---

## 1. Disjoint Sequence-Level Partitioning Policy

To eliminate cross-frame and cross-point contamination:
* **Train Split**: Disjoint sequences (e.g. `00`, `01`, `03`, `04`, `05`).
* **Validation Split**: Disjoint sequence `02` (used strictly for model selection).
* **Test Split**: Disjoint sequence `08` (held out completely until final checkpoint evaluation).
* **Guarantees**: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
* **Zero Point Synthesis**: Random point slicing from the same scan into train/test is strictly prohibited.
