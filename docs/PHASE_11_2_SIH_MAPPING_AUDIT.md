# Phase 11.2 SIH 4-Class Mapping Audit

**Project**: Foveated 2.5D LiDAR Mapping for Autonomous Navigation  
**Lead Engineer**: Atul (ML/AI Perception Lead)  
**Date**: August 22, 2026  

---

## 1. Frozen SIH 4-Class Semantic Ontology

* `0 = drivable_terrain` (Roads, parking, navigable surfaces)
* `1 = non_drivable_terrain` (Sidewalks, curbs, rough terrain)
* `2 = static_obstacle` (Buildings, vegetation, trunks, poles, fences)
* `3 = dynamic_object` (Vehicles, pedestrians, cyclists)
* `255 = ignore` (Unlabeled noise, outliers, unsupported points)

## 2. Invariant Verification

* **Output Label Dtype**: `uint8`
* **Output Class Range**: $\text{unique}(\text{SIH\_labels}) \subseteq \{0, 1, 2, 3, 255\}$
* **Neural Network Output**: Exactly 4 classes ($[B, N, 4]$) with `ignore_index = 255`.
* **Point-Label Correspondence**: $100\%$ aligned without shuffling.
