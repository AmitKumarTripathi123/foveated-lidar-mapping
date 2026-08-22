# Phase 1 <-> Phase 2 Metric Discrepancy & Root Cause Analysis

## 1. Context of Discrepancy
During initial benchmarking, two divergent metric profiles were observed:

| Profile Attribute | Metric Profile A (Previous Pipeline Log) | Metric Profile B (Uncalibrated Baseline) | Metric Profile C (Repaired & Normalized Model) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 77.73% | 64.25% | **78.96%** |
| **mIoU** | 44.92% | 27.51% | **53.22%** |
| **Drivable IoU (0)** | 39.36% | 0.00% | **28.12%** |
| **Non-Drivable IoU (1)** | 48.76% | 40.35% | **56.27%** |
| **Static Obstacle IoU (2)**| 85.67% | 69.70% | **88.20%** |
| **Dynamic Object IoU (3)** | 31.85% | 0.00% | **40.29%** |

---

## 2. Root Cause Investigation Findings

### A. Input Feature Scale Imbalance
In the uncalibrated baseline, raw $x, y$ coordinates range in $[0, 95\text{ m}]$ and range $r \in [0, 100\text{ m}]$, while elevation $z \in [-1.73, 6.0\text{ m}]$ and intensity $i \in [0, 1]$.
Because the linear projection layer was unscaled, gradients saturated along the large $x, y, r$ dimensions ($\approx 95$), drowning out the fine $15\text{ cm}$ elevation difference ($\Delta z = 0.15\text{ m}$) distinguishing road ($z \approx -1.73\text{ m}$) from sidewalk/terrain ($z \approx -1.58\text{ m}$).

### B. Class Imbalance Collapse
The SemanticPOSS sequence contains:
- Static Obstacles: $55.0\%$
- Non-Drivable Terrain: $25.8\%$
- Drivable Road: $14.2\%$
- Dynamic Objects: $4.8\%$
Without feature scaling and class weighting, the network collapsed into predicting only the majority classes (Static Obstacles & Non-Drivable Terrain), predicting exactly 0 points for Class 0 and Class 3, collapsing their IoU to $0.00\%$ and pulling total mIoU down to $27.51\%$.

### C. Repair Implemented
1. **Multi-Scale Input Normalization**: Scaled input coordinates ($x/50, y/50, z/3, i, r/50$) in `FoveatedPointSegNet` to preserve elevation sensitivity.
2. **Inverse Class Frequency Weighting**: Applied balanced loss weights ($[2.5, 1.5, 0.8, 4.0]$) in `Phase2Trainer`.

---

## 3. Authoritative Result
The repaired and calibrated model (`checkpoints/best_model.pth`) is authoritative:
- **Accuracy**: **78.96%**
- **mIoU**: **53.22%**
- **All 4 navigation super-classes actively predicted with positive IoU.**
