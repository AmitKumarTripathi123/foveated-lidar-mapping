# SIH Four-Class Semantic Ontology & Label Remapping Reference

This document defines the final 4-class machine learning ontology for the **Foveated 2.5D LiDAR Mapping for Autonomous Navigation** perception module, the verified raw-to-SIH class mapping rules, and the vectorized remapping architecture.

---

## 1. Final Project Ontology

The ML segmentation model operates exclusively on these 4 semantic categories plus an explicit ignore class:

```text
┌─────────────────────────────────────────────────────────────┐
│                 FROZEN SIH ML ONTOLOGY                      │
├─────┬────────────────────────┬──────────────────────────────┤
│ ID  │ Class Name             │ Semantic Description         │
├─────┼────────────────────────┼──────────────────────────────┤
│  0  │ drivable_terrain       │ Traversable road surfaces    │
│  1  │ non_drivable_terrain   │ Non-traversable ground       │
│  2  │ static_obstacle        │ Permanent obstacles/barriers │
│  3  │ dynamic_object         │ Traffic participants/moving  │
│ 255 │ ignore                 │ Excluded from training/loss  │
└─────┴────────────────────────┴──────────────────────────────┘
```

---

## 2. Class Definitions & Semantics

### Class 0 — `drivable_terrain`
* **Definition**: Physical surfaces intended for safe, unobstructed vehicular driving and maneuvering.
* **SemanticKITTI Categories**:
  * `40: road`: Paved asphalt/concrete roadway surfaces.
  * `44: parking`: Designated drivable parking lots and bays.
  * `60: lane-marking`: Road paint markings embedded in drivable road surfaces.

### Class 1 — `non_drivable_terrain`
* **Definition**: Ground or terrain surfaces that are non-traversable or off-limits for autonomous driving.
* **SemanticKITTI Categories**:
  * `48: sidewalk`: Pedestrian walkways and curbs adjacent to roads.
  * `49: other-ground`: Rough, uneven, or unpaved ground areas.
  * `72: terrain`: Grass, soil, dirt, and vegetative ground cover.

### Class 2 — `static_obstacle`
* **Definition**: Stationary objects, permanent physical barriers, infrastructure, and architectural structures that present collision hazards.
* **SemanticKITTI Categories**:
  * `50: building`: Walls, houses, commercial buildings, architectural structures.
  * `51: fence`: Perimeter fences, guard rails, and barriers.
  * `52: other-structure`: Bridges, tunnels, retaining walls.
  * `70: vegetation`: Trees, bushes, shrubs, dense foliage.
  * `71: trunk`: Solid tree trunks.
  * `80: pole`: Lamp posts, utility poles, traffic light poles.
  * `81: traffic-sign`: Signboards, traffic signs, informational plates.
  * `99: other-object`: Miscellaneous stationary physical obstacles.

### Class 3 — `dynamic_object`
* **Definition**: Moving or potentially dynamic objects representing active traffic participants and pedestrians.
* **SemanticKITTI Categories**:
  * Vehicles: `10: car`, `13: bus`, `15: motorcycle`, `16: on-rails`, `18: truck`, `20: other-vehicle`, `252: moving-car`, `257: moving-bus`, `258: moving-truck`, `259: moving-other-vehicle`.
  * Vulnerable Road Users: `11: bicycle`, `30: person`, `31: bicyclist`, `32: motorcyclist`, `253: moving-bicyclist`, `254: moving-pedestrian`, `255: moving-motorcyclist`, `256: moving-on-rails`.

### Class 255 — `ignore`
* **Definition**: Points explicitly excluded from supervised loss calculation and evaluation metrics.
* **SemanticKITTI Categories**:
  * `0: unlabeled`: Noise, unannotated points, or sensor artifacts.
  * `1: outlier`: Stray laser reflections, atmospheric scattering, water spray.
  * Any unmapped or ambiguous raw semantic ID.

---

## 3. Label Remapping Architecture

```text
               Raw SemanticKITTI .label File
                              │
                              ▼
                      Label Decoder
                 (raw_labels & 0xFFFF)
                              │
                              ▼
                      Raw Semantic IDs
                              │
                              ▼
                   SIH Label Remapper
              (Vectorized NumPy O(N) Lookup)
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
                 Model-Ready PyTorch Target
```

---

## 4. Complete Raw to SIH Mapping Table

| Raw ID | SemanticKITTI Name | SIH ID | SIH Super-Class | Justification |
| :--- | :--- | :--- | :--- | :--- |
| `0` | `unlabeled` | `255` | `ignore` | Noise / sensor artifact |
| `1` | `outlier` | `255` | `ignore` | Atmospheric reflection / outlier |
| `10` | `car` | `3` | `dynamic_object` | Moving / parkable vehicle |
| `11` | `bicycle` | `3` | `dynamic_object` | Active traffic participant |
| `13` | `bus` | `3` | `dynamic_object` | Large dynamic transit vehicle |
| `15` | `motorcycle` | `3` | `dynamic_object` | Powered two-wheeler |
| `16` | `on-rails` | `3` | `dynamic_object` | Trams / trains |
| `18` | `truck` | `3` | `dynamic_object` | Heavy commercial vehicle |
| `20` | `other-vehicle` | `3` | `dynamic_object` | Specialized dynamic vehicle |
| `30` | `person` | `3` | `dynamic_object` | Pedestrian |
| `31` | `bicyclist` | `3` | `dynamic_object` | Cyclist in motion |
| `32` | `motorcyclist` | `3` | `dynamic_object` | Motorcyclist in motion |
| `40` | `road` | `0` | `drivable_terrain` | Primary traversable roadway |
| `44` | `parking` | `0` | `drivable_terrain` | Traversable parking bay |
| `48` | `sidewalk` | `1` | `non_drivable_terrain` | Pedestrian walkway / non-drivable |
| `49` | `other-ground` | `1` | `non_drivable_terrain` | Rough / unpaved non-drivable ground |
| `50` | `building` | `2` | `static_obstacle` | Solid architectural wall / obstacle |
| `51` | `fence` | `2` | `static_obstacle` | Boundary barrier |
| `52` | `other-structure` | `2` | `static_obstacle` | Permanent civil infrastructure |
| `60` | `lane-marking` | `0` | `drivable_terrain` | Drivable road surface markings |
| `70` | `vegetation` | `2` | `static_obstacle` | Trees / shrubs |
| `71` | `trunk` | `2` | `static_obstacle` | Solid tree trunks |
| `72` | `terrain` | `1` | `non_drivable_terrain` | Grass / soil / unpaved terrain |
| `80` | `pole` | `2` | `static_obstacle` | Utility / lighting pole |
| `81` | `traffic-sign` | `2` | `static_obstacle` | Signboard / traffic sign |
| `99` | `other-object` | `2` | `static_obstacle` | Stationary obstacle |
| `252` | `moving-car` | `3` | `dynamic_object` | Vehicle in motion |
| `253` | `moving-bicyclist` | `3` | `dynamic_object` | Cyclist in motion |
| `254` | `moving-pedestrian` | `3` | `dynamic_object` | Pedestrian in motion |
| `255` | `moving-motorcyclist` | `3` | `dynamic_object` | Motorcyclist in motion |
| `256` | `moving-on-rails` | `3` | `dynamic_object` | Rail vehicle in motion |
| `257` | `moving-bus` | `3` | `dynamic_object` | Bus in motion |
| `258` | `moving-truck` | `3` | `dynamic_object` | Heavy vehicle in motion |
| `259` | `moving-other-vehicle`| `3` | `dynamic_object` | Dynamic vehicle in motion |

---

## 5. Representative Scan Audit Results (`000000.label`)

```text
Total Points: 66,658

Final SIH 4-Class Distribution:
  Class 0 (drivable_terrain)     : 23,000 pts (34.50%)
  Class 1 (non_drivable_terrain) :  8,000 pts (12.00%)
  Class 2 (static_obstacle)      : 28,500 pts (42.76%)
  Class 3 (dynamic_object)       :  6,000 pts ( 9.00%)
  Class 255 (ignore)             :  1,158 pts ( 1.74%)

Coverage:
  Supervised Points (Classes 0-3): 65,500 pts (98.26%)
  Ignored Points (Class 255)     :  1,158 pts ( 1.74%)
  Unmapped Anomalies             : 0
```

---

## 6. Python API Usage

```python
from pathlib import Path
from ml.data.dataset import load_labels
from ml.data.label_mapping import SemanticLabelRemapper

# 1. Load raw dataset labels
raw_labels = load_labels(Path("dataset/sequences/00/labels/000000.label"))

# 2. Instantiate remapper
remapper = SemanticLabelRemapper()

# 3. Vectorized remapping to SIH classes
sih_labels = remapper.remap(raw_labels)

# 4. Generate audit report
report = remapper.audit(raw_labels, sih_labels)
print(f"Total points: {report.total_points}")
for item in report.sih_distribution:
    print(f"  Class {item.class_id} ({item.class_name}): {item.count} pts ({item.percentage:.2f}%)")
```
