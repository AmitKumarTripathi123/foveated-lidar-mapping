# Phase 13 — Semantic Aggregation & Distribution Validation Report

## 1. Executive Summary & Objective
Phase 13 validates the mathematical and functional integration of **Atul's SPVCNN Semantic Predictions** into the **Foveated 2.5D LiDAR Grid Engine**. Rather than compressing the entire cell into a single hard label and losing minority classes, the Grid Engine preserves the full per-class discrete distribution:

$$\text{semantic\_counts} = \{ c : N_c \}$$

From this preserved distribution, the engine derives:
1. **Dominant Class**: $\text{argmax}_c N_c$ (breaking ties with deterministic safety-first priority: Dynamic > Static > Non-Drivable > Drivable).
2. **Class Probability**: $P(c) = \frac{\text{semantic\_counts}[c]}{\text{valid\_semantic\_count}}$.
3. **Semantic Confidence**: Arithmetic mean confidence score for points assigned to the cell.

---

## 2. Super-Class Ontology Contract

| Super-Class ID | Enumeration | Description | Priority (Tie-Breaker) |
| :---: | :--- | :--- | :---: |
| **0** | `DRIVABLE_TERRAIN` | Road, asphalt, walkable surfaces | 1 |
| **1** | `NON_DRIVABLE_TERRAIN` | Sidewalks, terrain, vegetation, curbs | 2 |
| **2** | `STATIC_OBSTACLE` | Poles, fences, buildings, barriers | 3 |
| **3** | `DYNAMIC_OBJECT` | Vehicles, pedestrians, cyclists | 4 (Highest) |
| **255** | `IGNORE_LABEL` | Outliers, invalid range, unclassified | 0 |

---

## 3. Mathematical Semantic Aggregation Contract

For any populated cell $C$ receiving $N$ points:

$$\text{semantic\_counts}[c] = \sum_{i \in \mathcal{P}_C} \mathbb{I}(\text{label}_i = c)$$

$$\text{valid\_semantic\_count} = \sum_{c \in \{0, 1, 2, 3\}} \text{semantic\_counts}[c]$$

$$P(c) = \frac{\text{semantic\_counts}[c]}{\text{valid\_semantic\_count}} \quad (\forall c \in \{0, 1, 2, 3\})$$

$$\text{Axiom Invariant}: \sum_{c=0}^3 P(c) \equiv 1.0 \quad (0 \le P(c) \le 1)$$

---

## 4. Key Validation Scenarios

### 4.1 Case 1: 100% Single Class
* **Input**: 100 points of `DRIVABLE_TERRAIN` (Class 0)
* **Result**: `dominant_class = 0`, $P(\text{road}) = 1.00$, $P(\text{dynamic}) = 0.00$ (`PASS`).

### 4.2 Case 2: 60/30/10 Distribution (Minority Preservation)
* **Input**: 60 road, 30 dynamic, 10 static points in cell.
* **Result**: `dominant_class = 0`, $P(\text{road}) = 0.60$, $P(\text{dynamic}) = 0.30$, $P(\text{static}) = 0.10$.
* **Minority Check**: Dynamic (30) and static (10) counts are preserved in `semantic_counts` and recoverable (`PASS`).

### 4.3 Case 3 & 4: 51% vs 49% Directionality
* **Vehicle 51%, Road 49%**: `dominant_class = 3`, $P(\text{vehicle}) = 0.51$ (`PASS`).
* **Road 51%, Vehicle 49%**: `dominant_class = 0`, $P(\text{road}) = 0.51$ (`PASS`).

### 4.4 Deterministic Tie-Breaking (50/50)
* **Input**: 50 road (0), 50 vehicle (3).
* **Policy**: Safety-critical priority rule selects `DYNAMIC_OBJECT` (3) (`PASS`).

---

## 5. Visualizations & Perception Artifacts

The generated visualizations in `docs/phase13_semantic_plots/` confirm:
* **Visualization A (`vis_a_dominant_class_map.png`)**: 2D BEV map with categorical color-coding of dominant semantic classes.
* **Visualization B (`vis_b_probability_map.png`)**: Continuous probability field map highlighting high-confidence core regions vs mixed boundary cells.
* **Visualization C (`vis_c_distribution_example.png`)**: Preserved discrete histogram for the 60/30/10 distribution.
* **Visualization D (`vis_d_combined_25d_map.png`)**: Unified 3D mesh rendering showing co-registered elevation and semantic classification.

---

## 6. Comprehensive Audit & Parity Results
* **3-Way Parity**: $\text{Python Reference} \equiv \text{C++ Grid Engine} \equiv \text{Independent Semantic Oracle}$ (`PASS`).
* **Point Conservation**: $\sum \text{semantic\_counts} \equiv N_{\text{valid}}$ with 0 discrepancy (`PASS`).
* **Multi-Seed Testing**: 100% pass across seeds 42, 123, 456, 999, 2026 (`PASS`).
