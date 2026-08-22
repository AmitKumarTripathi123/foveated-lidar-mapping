# PointNet++ Full Training, Validation & Experimentation (Phase 5)

This package contains the PyTorch training system, loss strategies, point-cloud evaluation metrics, and experiment tracking infrastructure for the **PointNet++ 3D Semantic Segmentation** model.

---

## 1. End-to-End Training & Validation Architecture

```text
                 RAW DATASET
                      │
                      ▼
               Phase 1 Loader
                      │
                      ▼
              Phase 2 Preprocess
                      │
                      ▼
             Phase 3 Remapping
                      │
              ┌───────┴────────┐
              ▼                ▼
           Points            Labels
           N × 4              N
                              │
                              ▼
                    Train / Val / Test
                              │
                              ▼
                         PointNet++
                              │
                              ▼
                         4 Logits
                              │
                              ▼
                     Cross Entropy
                     ignore=255
                              │
                              ▼
                         Optimizer
                              │
                              ▼
                          Training
                              │
                              ▼
                         Validation
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
             IoU          Precision       Recall
                │
                ▼
               mIoU
                │
                ▼
          Best Checkpoint
                │
                ▼
          Independent Test
                │
                ▼
        Best Trained Model
                │
                ▼
       PointNet2Predictor
                │
                ▼
 [x,y,z,predicted_class,confidence]
                │
                ▼
          FUTURE MAPPING
```

---

## 2. Dataset Split & Class Statistics

* **Dataset Sequences**: `sequence 00` ($66,658$ raw points/labels).
* **Deterministic Split**:
  * **Train Split**: `sequence 00` (sampled with `seed=42`)
  * **Validation Split**: `sequence 00` (sampled with `seed=1042`)
* **Class Distribution**:
  * `Class 0 (drivable_terrain)`: $34.50\%$ ($23,000$ points)
  * `Class 1 (non_drivable_terrain)`: $12.00\%$ ($8,000$ points)
  * `Class 2 (static_obstacle)`: $42.76\%$ ($28,500$ points)
  * `Class 3 (dynamic_object)`: $9.00\%$ ($6,000$ points)
  * `Class 255 (ignore)`: $1.74\%$ ($1,158$ points)

---

## 3. Loss Strategies & Class Imbalance Handling

1. **Plain Cross-Entropy (`baseline_ce`)**:
   $$\mathcal{L} = -\frac{1}{N_{\text{valid}}} \sum_{i: y_i \ne 255} \log P(y_i \mid x_i)$$
2. **Class-Weighted Cross-Entropy (`weighted_ce`)**:
   $$\mathcal{L}_{\text{weighted}} = -\frac{1}{N_{\text{valid}}} \sum_{i: y_i \ne 255} w_{y_i} \log P(y_i \mid x_i)$$
   where weights $w_c = \frac{1}{\text{count}_c + \epsilon}$ are normalized ($\sum w_c = 4$) and computed **strictly from training data**.
   * Training weights: `[0.4601 (C0), 1.4748 (C1), 0.3797 (C2), 1.6854 (C3)]`.

---

## 4. Evaluation Metrics & Confusion Matrix

All metrics strictly exclude points where $\text{target} == 255$:

* **Per-Class IoU**: $\text{IoU}_c = \frac{TP_c}{TP_c + FP_c + FN_c}$
* **Mean IoU (mIoU)**: $\text{mIoU} = \frac{1}{4} \sum_{c=0}^3 \text{IoU}_c$
* **Precision & Recall**: $\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$
* **Overall Accuracy**: $\frac{\sum TP_c}{\sum (TP_c + FN_c)}$
* **Primary Model Selection Metric**: Validation **mIoU** (higher is better).

---

## 5. Controlled Experiment Comparison

| Experiment | Description | Best Epoch | Val mIoU | Val Accuracy | Drivable IoU | Non-Drive IoU | Static IoU | Dynamic IoU |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline_ce`** | Plain CE (No weights, no aug) | **3** | **11.07%** | **44.28%** | 0.00% | 0.00% | 44.28% | 0.00% |
| **`weighted_ce`** | Inverse-frequency weighted CE | 1 | 3.28% | 13.13% | 0.00% | 13.13% | 0.00% | 0.00% |
| **`weighted_ce_aug`**| Weighted CE + 3D Augmentation | 1 | 3.28% | 13.13% | 0.00% | 13.13% | 0.00% | 0.00% |

---

## 6. Confusion Matrix & Diagnosis

Evaluation on validation scan (`experiments/baseline_ce/best_checkpoint.pt`):

```text
4x4 Confusion Matrix (Rows = Ground Truth, Cols = Predicted):
--------------------------------------------------------------
GT \ Pred    |       C0 |       C1 |       C2 |       C3
--------------------------------------------------------------
Class 0      |        0 |        0 |      350 |        0
Class 1      |        0 |        0 |      132 |        0
Class 2      |        0 |        0 |      445 |        0
Class 3      |        0 |        0 |       78 |        0
--------------------------------------------------------------
```

> [!WARNING]
> **Class Collapse Finding**: With a single representative frame under CPU training, the model achieves $44.28\%$ overall accuracy by predicting the dominant `static_obstacle` class ($445/1005 = 44.28\%$), resulting in $0\%$ IoU on dynamic objects and drivable terrain. This highlights why overall accuracy is deceptive and validates our use of **mIoU** as the primary gate.

---

## 7. CLI Usage

```bash
# 1. Run training experiment
python scripts/train.py --experiment baseline_ce --epochs 15 --num-points 1024

# 2. Evaluate checkpoint and verify contract
python scripts/evaluate.py --checkpoint experiments/baseline_ce/best_checkpoint.pt

# 3. Compare multiple experiments
python scripts/compare_experiments.py --experiments baseline_ce weighted_ce weighted_ce_aug
```
