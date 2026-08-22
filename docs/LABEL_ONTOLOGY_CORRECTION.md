# Label Ontology Forensic Audit & Resolution Report

**Project**: Foveated 2.5D LiDAR Mapping for Autonomous Navigation  
**Perception Lead**: Atul  
**Date**: August 22, 2026  

---

## 1. Forensic Discrepancy Analysis

* **Amit''s Initial Documentation (`class_map.py`)**: Documented SemanticPOSS IDs (e.g. 21=bike, 22=ground, 15=building, 4=people).
* **Physical Local Dataset Scan (`000000.label`)**: Contains actual SemanticKITTI IDs (`40=road`, `48=sidewalk`, `50=building`, `70=vegetation`, `10=car`).
* **Resolution in Phase 11**:
  * [`ml/data/authoritative_label_mapping.py`](file:///C:/Users/atuls/OneDrive/Desktop/Lidar/ml/data/authoritative_label_mapping.py) and [`configs/authoritative_label_mapping.yaml`](file:///C:/Users/atuls/OneDrive/Desktop/Lidar/configs/authoritative_label_mapping.yaml) provide authoritative support for **both SemanticKITTI and SemanticPOSS**.
  * The local physical scan is mapped through the verified SemanticKITTI specification into the SIH 4-Class ontology.

---

## 2. Authoritative SIH 4-Class Mapping Summary

| Raw SemanticKITTI ID | Class Name | Supervised Count | SIH Class ID | SIH Super-Class |
| :---: | :--- | :---: | :---: | :--- |
| `40` | `road` | $23,000$ | `0` | `drivable_terrain` |
| `48` | `sidewalk` | $8,000$ | `1` | `non_drivable_terrain` |
| `50` | `building` | $10,000$ | `2` | `static_obstacle` |
| `51` | `fence` | $2,000$ | `2` | `static_obstacle` |
| `70` | `vegetation` | $13,000$ | `2` | `static_obstacle` |
| `71` | `trunk` | $2,000$ | `2` | `static_obstacle` |
| `80` | `pole` | $1,500$ | `2` | `static_obstacle` |
| `10` | `car` | $6,000$ | `3` | `dynamic_object` |
| `0` | `unlabeled` | $1,158$ | `255` | `ignore` |
