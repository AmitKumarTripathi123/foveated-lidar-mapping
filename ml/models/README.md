# PointNet++ 3D Semantic Segmentation Baseline & Prediction Contract

This package provides the core **PointNet++** point-wise semantic segmentation architecture, prediction utilities, and interface contracts for the **Foveated 2.5D LiDAR Mapping for Autonomous Navigation** perception module.

---

## 1. End-to-End Perception Pipeline

```text
                     ATUL ML PIPELINE
                            │
                            ▼
                    Phase 1 Loader
               (load_point_cloud / load_labels)
                            │
                            ▼
                   Phase 2 Preprocessor
            (Sampling to N=16384, NaN removal)
                            │
                            ▼
                  Phase 3 Label Remapper
             (4-Class SIH Ontology + Ignore)
                            │
                  ┌─────────┴─────────┐
                  │                   │
                  ▼                   ▼
             Points N×4          Labels N
       [x, y, z, intensity]    {0,1,2,3,255}
                  │                   │
                  └─────────┬─────────┘
                            ▼
                       PointNet++
                  (Hierarchical SA + FP)
                            │
                            ▼
                       N × 4 logits
                            │
                            ▼
                         Softmax
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
       predicted_class              confidence
             N                           N
       {0, 1, 2, 3}                 [0.0, 1.0]
              │                           │
              └─────────────┬─────────────┘
                            │
                            ▼
                    Original XYZ N×3
                            │
                            ▼
              FROZEN ML OUTPUT CONTRACT
                            │
                            ▼
          [x,y,z,predicted_class,confidence]
                            │
                            ▼
                  FUTURE MAPPING MODULE
```

---

## 2. Model Architecture

The `PointNet2SemSeg` architecture is implemented in pure PyTorch and performs hierarchical local feature learning and multi-scale contextual aggregation:

* **Input Channels**: $4$ total channels:
  * **Coordinates**: $3$ channels (`[x, y, z]` in meters)
  * **Features**: $1$ channel (`[intensity]` reflectance)
* **Set Abstraction (SA) Hierarchy**:
  * **SA1**: $1024$ centroids, radius $0.2\text{m}$, $32$ neighbors, $\text{MLP}=[32, 32, 64]$
  * **SA2**: $256$ centroids, radius $0.4\text{m}$, $32$ neighbors, $\text{MLP}=[64, 64, 128]$
  * **SA3**: $64$ centroids, radius $0.8\text{m}$, $32$ neighbors, $\text{MLP}=[128, 128, 256]$
  * **SA4**: $16$ centroids, radius $1.6\text{m}$, $32$ neighbors, $\text{MLP}=[256, 256, 512]$
* **Feature Propagation (FP) Hierarchy**:
  * **FP4**: $16 \to 64$ points with 3-NN interpolation + skip connection, $\text{MLP}=[256, 256]$
  * **FP3**: $64 \to 256$ points with 3-NN interpolation + skip connection, $\text{MLP}=[256, 256]$
  * **FP2**: $256 \to 1024$ points with 3-NN interpolation + skip connection, $\text{MLP}=[128, 128]$
  * **FP1**: $1024 \to N$ points with 3-NN interpolation + skip connection, $\text{MLP}=[128, 128, 128]$
* **Point-Wise Segmentation Head**:
  * $\text{Conv1d}(128, 128) \to \text{BatchNorm1d} \to \text{ReLU} \to \text{Dropout}(0.5) \to \text{Conv1d}(128, 4)$
* **Total Parameters**: $909,252$ (all trainable)

---

## 3. Frozen SIH 4-Class Ontology & Ignore Semantics

The model classification head has exactly **4 output channels**:

| Output Channel | Class Name | Semantic Scope |
| :--- | :--- | :--- |
| `channel 0` | `drivable_terrain` | Traversable roadway, parking, lane markings |
| `channel 1` | `non_drivable_terrain` | Sidewalks, terrain, grass, rough ground |
| `channel 2` | `static_obstacle` | Buildings, fences, poles, vegetation, signs, barriers |
| `channel 3` | `dynamic_object` | Moving/parked vehicles, pedestrians, cyclists |

> [!IMPORTANT]
> `255` (`ignore`) is an ignore target index for supervised training loss (`ignore_index=255`), **NOT** a 5th output class.

---

## 4. Amit's Frozen ML $\to$ Mapping Contract

The prediction output interface strictly guarantees:

1. **`xyz`**: `(N, 3)` `float32` — Identical in value and ordering to input coordinates ($\text{input\_xyz}[i] == \text{output\_xyz}[i]$).
2. **`predicted_class`**: `(N,)` `int64` — Predicted class IDs strictly $\in \{0, 1, 2, 3\}$.
3. **`confidence`**: `(N,)` `float32` — Maximum softmax probability strictly $\in [0.0, 1.0]$.

---

## 5. Python API Usage

```python
import numpy as np
from ml.models import build_model, PointNet2Predictor

# 1. Build model and predictor
model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
predictor = PointNet2Predictor(model=model, device="cpu")

# 2. Input point cloud array (N, 4) [x, y, z, intensity]
points = np.random.randn(16384, 4).astype(np.float32)

# 3. Predict frame with contract guarantee
result = predictor.predict(points)

coordinates = result["xyz"]              # (16384, 3) float32
predicted_class = result["predicted_class"] # (16384,) int64 in {0, 1, 2, 3}
confidence = result["confidence"]        # (16384,) float32 in [0.0, 1.0]
```

---

## 6. CLI Baseline Evaluation

```bash
python scripts/evaluate_baseline.py \
    --scan dataset/sequences/00/velodyne/000000.bin \
    --label dataset/sequences/00/labels/000000.label \
    --num-points 16384 \
    --seed 42 \
    --overfit-check
```

---

## 7. Known Limitations (Deferred to Phase 5)

* **Supervised Training Pipeline**: 50+ epoch training, AdamW optimizer, and learning rate scheduling belong to Phase 5.
* **Loss Functions & Class Weighting**: Focal loss / weighted cross entropy for class imbalance belong to Phase 5.
* **Data Augmentation**: Random point jittering, rotations, and dropouts belong to Phase 5.
* **Validation Metrics**: Mean Intersection over Union (mIoU) benchmark belongs to Phase 5.
