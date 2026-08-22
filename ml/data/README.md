# LiDAR Preprocessing, SIH Label Remapping & Model-Ready Data Pipeline

This package provides the complete LiDAR data loading, quality validation, label-preserving preprocessing, SIH 4-class ontology remapping, and PyTorch `Dataset` integration for the **Foveated 2.5D LiDAR Mapping for Autonomous Navigation** project.

---

## 1. End-to-End Pipeline Architecture

```text
               Raw LiDAR (.bin) + Raw Labels (.label)
                               │
                               ▼
                       Phase 1 Loader
               (load_point_cloud / load_labels)
                               │
                               ▼
                    Data Quality Validation
                      (NaN / Inf Check)
                               │
                               ▼
                  Invalid-Point Removal
               (Shared Boolean Validity Mask)
                               │
                               ▼
                   Optional Range Filter
                  (3D Metric Bounding Box)
                               │
                               ▼
                  Point-Count Handling
              (keep_all / random / deterministic / pad)
                               │
                               ▼
                 Phase 3 SIH Label Remapping
             (Vectorized NumPy O(N) Remapper)
                               │
                               ▼
           ┌─────────────────────────────────────┐
           │        SIH 4-Class Targets          │
           │  0: drivable_terrain                │
           │  1: non_drivable_terrain            │
           │  2: static_obstacle                 │
           │  3: dynamic_object                  │
           │  255: ignore                        │
           └─────────────────────────────────────┘
                               │
                               ▼
                 Label Alignment Verification
                  (N_points == N_labels Check)
                               │
                               ▼
                  PyTorch LidarDataset / DataLoader
                 (with Batch Collation Support)
```

---

## 2. Frozen SIH 4-Class Semantic Ontology

| Class ID | Super-Class Name | Semantic Description | Included Raw SemanticKITTI Categories |
| :--- | :--- | :--- | :--- |
| `0` | `drivable_terrain` | Traversable roadway surfaces | `40: road`, `44: parking`, `60: lane-marking` |
| `1` | `non_drivable_terrain` | Non-traversable ground / terrain | `48: sidewalk`, `49: other-ground`, `72: terrain` |
| `2` | `static_obstacle` | Stationary structures, barriers, poles | `50: building`, `51: fence`, `52: other-structure`, `70: vegetation`, `71: trunk`, `80: pole`, `81: traffic-sign`, `99: other-object` |
| `3` | `dynamic_object` | Moving / active traffic participants | `10: car`, `11: bicycle`, `13: bus`, `15: motorcycle`, `16: on-rails`, `18: truck`, `20: other-vehicle`, `30: person`, `31: bicyclist`, `32: motorcyclist`, `252-259: moving-*` |
| `255` | `ignore` | Excluded from supervised loss | `0: unlabeled / noise`, `1: outlier`, unmapped categories |

---

## 3. Python API Usage

```python
from pathlib import Path
from ml.data.dataset import load_point_cloud, load_labels, LidarDataset
from ml.data.preprocessing import LidarPreprocessor, PreprocessingConfig, SamplingConfig
from ml.data.label_mapping import SemanticLabelRemapper

# 1. Direct label remapping
remapper = SemanticLabelRemapper()
raw_labels = load_labels(Path("dataset/sequences/00/labels/000000.label"))
sih_labels = remapper.remap(raw_labels)

# 2. PyTorch Dataset with preprocessor and remapper
config = PreprocessingConfig(
    sampling=SamplingConfig(strategy="random", num_points=16384, seed=42)
)
preprocessor = LidarPreprocessor(config)

dataset = LidarDataset(
    root="dataset",
    split="train",
    sequences=["00"],
    preprocessor=preprocessor,
    label_remapper=remapper,
    to_tensor=True,
)

sample = dataset[0]
points_tensor = sample["points"]  # (16384, 4) torch.float32
labels_tensor = sample["labels"]  # (16384,)   torch.int64 {0, 1, 2, 3, 255}
```

---

## 4. CLI Tools Summary

* **Phase 1 LiDAR Inspector**:
  ```bash
  python scripts/inspect_lidar.py --scan dataset/sequences/00/velodyne/000000.bin --label dataset/sequences/00/labels/000000.label
  ```
* **Phase 2 Preprocessing & Sampling**:
  ```bash
  python scripts/preprocess_lidar.py --scan dataset/sequences/00/velodyne/000000.bin --label dataset/sequences/00/labels/000000.label --num-points 16384 --seed 42 --to-tensor
  ```
* **Phase 3 Label Remapping & Audit**:
  ```bash
  python scripts/remap_labels.py --label dataset/sequences/00/labels/000000.label --config ml/configs/label_mapping.yaml
  ```
