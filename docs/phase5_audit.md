# Phase 5 Repository Audit: Foveated 2.5D Spatial Grid Engine

## 1. Executive Summary
This document provides a comprehensive audit of the spatial 2.5D LiDAR grid mapping architecture in the repository before implementing the pure C++ reference engine.

## 2. Core Python Grid Architecture & Data Flow
The 2.5D grid engine transforms continuous 3D point cloud measurements into discrete, foveated, multi-layer 2.5D elevation and traversability grid cells.

### Data Flow Pipeline:
```text
Raw LiDAR Scans (.bin)
       ↓
Range Filter (0.5m ≤ r < 100m)
       ↓
Multi-Band Foveation (Near / Mid / Mid-Far / Far)
       ↓
SPVCNN Sparse Point-Voxel Neural Inference
       ↓
ClassifiedPoint[] (x, y, z, intensity, class_id, confidence)
       ↓
FoveatedGrid25D.build_grid()
       ↓
2.5D Multi-Layer Grid Map (GridMap25D)
```

## 3. Key Components & Files
1. `src/types.py`:
   - `ClassifiedPoint`: `(x: float, y: float, z: float, intensity: float, class_id: uint8, confidence: float)`
   - `SuperClass`: Enum { `DRIVABLE_TERRAIN=0`, `NON_DRIVABLE_TERRAIN=1`, `STATIC_OBSTACLE=2`, `DYNAMIC_OBJECT=3`, `IGNORE_LABEL=255` }
   - `CellState`: Enum { `UNKNOWN=0`, `FREE=1`, `OCCUPIED=2` }
   - `FoveationBand`: `(name: str, min_range: float, max_range: float, voxel_size: float)`
   - `GridCell25D`: Full cell representation with elevation mean/min/max, semantic class, confidence, traversability, and bounds.
2. `src/foveated_grid.py`:
   - `DEFAULT_FROZEN_BANDS`:
     * `near_field`: `[0.0, 10.0) m`, resolution = `0.05 m`
     * `mid_near_field`: `[10.0, 30.0) m`, resolution = `0.10 m`
     * `mid_far_field`: `[30.0, 60.0) m`, resolution = `0.25 m`
     * `far_field`: `[60.0, 100.0) m`, resolution = `0.50 m`
   - `distance_to_band(r)`: Uses 2D radial distance $r = \sqrt{x^2 + y^2}$.
   - `xy_to_cell(x, y, resolution)`: Maps continuous $(x, y)$ to discrete cell index using mathematical floor:
     $$i_x = \lfloor x / \text{resolution} \rfloor, \quad i_y = \lfloor y / \text{resolution} \rfloor$$
   - `FoveatedGrid25D.build_grid(points, labels, confidences)`: Aggregates points per unique cell $(i_x, i_y)$ within each distance band.
3. `phase2/adapter.py`:
   - `MLToMappingAdapter.prediction_to_grid()`: Integrates `SemanticPrediction` into `GridMap25D`.

## 4. Aggregation Semantics
- **Point Count**: Total count of LiDAR points mapped into the cell.
- **Elevation Mean**: $\bar{z} = \frac{1}{N} \sum_{i=1}^N z_i$.
- **Elevation Min / Max**: Minimum and maximum $z$ coordinates among all points in the cell.
- **Semantic Priority Aggregation**:
  $$\text{dynamic\_object (3)} > \text{static\_obstacle (2)} > \text{non\_drivable (1)} > \text{drivable (0)} > \text{ignore (255)}$$
- **Traversability Score**:
  * $1.0$ for `DRIVABLE_TERRAIN` (0)
  * $0.2$ for `NON_DRIVABLE_TERRAIN` (1)
  * $0.0$ for `STATIC_OBSTACLE` (2), `DYNAMIC_OBJECT` (3), `IGNORE_LABEL` (255).
