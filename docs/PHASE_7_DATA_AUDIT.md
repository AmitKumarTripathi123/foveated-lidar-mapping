# Phase 7 Real LiDAR Dataset & Class Distribution Audit

**Project**: Foveated 2.5D LiDAR Mapping for Autonomous Navigation  
**Lead Engineer**: Atul (ML/AI Perception Lead)  
**Date**: August 22, 2026  

---

## 1. Discovered Dataset Inventory

* **Discovered Sequences**: 1 (`sequence 00`)
* **Discovered Scans**: 1 scan pair (`000000.bin` and `000000.label`)
* **Total Points**: $66,658$ points
* **Point Cloud Format**: `float32`, shape $(66658, 4)$ `[x, y, z, intensity]`
* **Label Format**: `uint32`, shape $(66658,)$
* **Finite / NaN / Inf Check**: **PASS** ($0$ NaNs, $0$ Infs, $100\%$ finite values)
* **Point-Label Alignment**: **PASS** ($N_{\text{points}} == N_{\text{labels}} == 66,658$)

---

## 2. Raw Semantic Breakdown & SIH Remapping

| Raw ID | SemanticKITTI Name | Raw Count | Raw Pct | SIH Super-Class ID | SIH Super-Class Name |
| :---: | :--- | :---: | :---: | :---: | :--- |
| `0` | `unlabeled` | $1,158$ | $1.74\%$ | `255` | `ignore` |
| `10` | `car` | $6,000$ | $9.00\%$ | `3` | `dynamic_object` |
| `40` | `road` | $23,000$ | $34.50\%$ | `0` | `drivable_terrain` |
| `48` | `sidewalk` | $8,000$ | $12.00\%$ | `1` | `non_drivable_terrain` |
| `50` | `building` | $15,000$ | $22.50\%$ | `2` | `static_obstacle` |
| `51` | `fence` | $3,500$ | $5.25\%$ | `2` | `static_obstacle` |
| `70` | `vegetation` | $7,000$ | $10.50\%$ | `2` | `static_obstacle` |
| `71` | `trunk` | $1,500$ | $2.25\%$ | `2` | `static_obstacle` |
| `80` | `pole` | $1,500$ | $2.25\%$ | `2` | `static_obstacle` |

---

## 3. SIH 4-Class Distribution & Imbalance Analysis

| SIH Class ID | Class Name | Supervised Points | Percentage | Imbalance Ratio (vs Min) |
| :---: | :--- | :---: | :---: | :---: |
| `0` | `drivable_terrain` | $23,000$ | $34.50\%$ | $3.83 : 1$ |
| `1` | `non_drivable_terrain` | $8,000$ | $12.00\%$ | $1.33 : 1$ |
| `2` | `static_obstacle` | $28,500$ | $42.76\%$ | $\mathbf{4.75 : 1}$ (Majority) |
| `3` | `dynamic_object` | $6,000$ | $9.00\%$ | $\mathbf{1.00 : 1}$ (Minority) |
| `255` | `ignore` | $1,158$ | $1.74\%$ | N/A (Excluded from Loss) |

* **Total Supervised Points**: $65,500$ ($98.26\%$)
* **Ignored Points**: $1,158$ ($1.74\%$)
* **Imbalance Ratio**: $\mathbf{4.75 : 1}$ (`static_obstacle` vs `dynamic_object`)

---

## 4. Dataset Limitation Statement

> [!IMPORTANT]
> **DATASET AVAILABILITY STATE: STATE C (Single Local Scan)**  
> The current workspace contains $1$ representative LiDAR frame (`000000.bin`). The entire ML pipeline (Foveated Preprocessing, Normalization, Training, Validation, Checkpointing, ML $\to$ Mapping Adapter) is fully functional and regression-tested. However, independent statistical generalization across unseen driving environments requires downloading multi-sequence driving datasets (`sequences 00--05`).
