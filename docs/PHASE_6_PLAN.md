# Phase 6 Technical Plan: ML Perception to 2.5D Elevation & Semantic Grid Mapping

**Project**: Foveated 2.5D LiDAR Mapping for Autonomous Navigation  
**Lead Authors**: Atul (ML/AI Perception Lead) & Amit (Foveated Preprocessing & Mapping Lead)  
**Branch**: `atul/phase6-ml-mapping`  
**Status**: APPROVED & ACTIVE  

---

## 1. System Architecture & Module Boundaries

```text
                     Raw LiDAR Point Cloud [N, 4]
                                 │
                                 ▼
                     Amit''s Foveated Voxelizer
                 (3 Zones: 0-10m, 10-40m, 40-100m)
                                 │
                                 ▼
                    Point-Count Normalization
                         (Target N = 1024)
                                 │
                                 ▼
                    PointNet++ Semantic Head
                         (4-Class Logits)
                                 │
                                 ▼
                   PointNet2Predictor Output
              PredictionBatch [x, y, z, class, conf]
                                 │
                                 ▼
                   MLToMappingAdapter (Phase 6)
            (Validation, Projection & 2.5D Accumulation)
                                 │
                                 ▼
                 2.5D Elevation & Semantic Grid
             (Elevation, Semantics, Traversability)
                                 │
                                 ▼
                    Downstream Path Planner
```

---

## 2. Frozen ML $\to$ Mapping Contract

### `PredictionBatch` Data Transfer Object:
* **`xyz`**: `(N, 3)` `float32` in vehicle/sensor frame (meters).
* **`predicted_class`**: `(N,)` `int64` strictly $\in \{0, 1, 2, 3\}$:
  * `0`: `drivable_terrain`
  * `1`: `non_drivable_terrain`
  * `2`: `static_obstacle`
  * `3`: `dynamic_object`
* **`confidence`**: `(N,)` `float32` strictly $\in [0.0, 1.0]$.
* **Point Ordering**: $\text{in\_xyz}[i] == \text{out\_xyz}[i]$ for all $i \in [0, N-1]$.

---

## 3. 2.5D Grid Representation & Layers

The `MLToMappingAdapter` translates 3D semantic points into a structured multi-layer 2.5D occupancy grid:

1. **`elevation_min` & `elevation_max`**: `(H, W)` `float32` bounding heights per cell.
2. **`elevation_mean`**: `(H, W)` `float32` ground surface height.
3. **`semantic_layer`**: `(H, W)` `int64` majority semantic vote per cell ($255 = \text{unobserved}$).
4. **`confidence_layer`**: `(H, W)` `float32` mean prediction confidence per cell.
5. **`traversability_layer`**: `(H, W)` `float32` traversability cost:
   * `1.0`: Drivable terrain
   * `0.2`: Non-drivable terrain (curbs / rough ground)
   * `0.0`: Static obstacles & dynamic objects (blocked)
   * `-1.0`: Unobserved cell.
6. **`point_count_layer`**: `(H, W)` `int32` point density per cell.

---

## 4. Test Strategy & Verification

* **Unit Tests**: Coverage for shape validity, range enforcement, dtype verification, NaN/Inf rejection, and length mismatch handling in `tests/test_ml_mapping_integration.py`.
* **Smoke Integration**: End-to-end projection of real representative scan into `GridMap25D`.
* **Total Suite**: 108 tests passing across all project phases.
