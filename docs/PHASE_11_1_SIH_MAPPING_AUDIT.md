# Phase 11.1 SIH 4-Class Mapping Audit

**Project**: Foveated 2.5D LiDAR Mapping for Autonomous Navigation  
**Lead Engineer**: Atul (ML/AI Perception Lead)  
**Date**: August 22, 2026  

---

## 1. Frozen SIH 4-Class Mapping Specifications

### SemanticPOSS Mapping
| Raw ID | Raw Semantics | SIH ID | SIH Super-Class |
| :---: | :--- | :---: | :--- |
| `22` | ground | `0` | `drivable_terrain` |
| `19, 20` | other static, ground | `1` | `non_drivable_terrain` |
| `8, 9, 10-18` | trunk, plants, signs, poles, buildings, fence | `2` | `static_obstacle` |
| `4, 5, 6, 7, 21` | people, rider, car, bike | `3` | `dynamic_object` |
| `0, 1` | unlabeled, outlier | `255` | `ignore` |

### SemanticKITTI Mapping (Local Representative Scan)
| Raw ID | Raw Semantics | Count | Pct | SIH ID | SIH Super-Class |
| :---: | :--- | :---: | :---: | :---: | :--- |
| `40` | road | $23,000$ | $34.50\%$ | `0` | `drivable_terrain` |
| `48` | sidewalk | $8,000$ | $12.00\%$ | `1` | `non_drivable_terrain` |
| `50, 51, 70, 71, 80` | building, fence, veg, trunk, pole | $28,500$ | $42.76\%$ | `2` | `static_obstacle` |
| `10` | car | $6,000$ | $9.00\%$ | `3` | `dynamic_object` |
| `0` | unlabeled | $1,158$ | $1.74\%$ | `255` | `ignore` |
