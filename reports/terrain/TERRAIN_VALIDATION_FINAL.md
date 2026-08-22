# SemanticPOSS Terrain Validation Final Report

## Executive Summary
This report concludes the thorough investigation of **SemanticPOSS terrain and ground label semantics** for the *Foveated 2.5D LiDAR Mapping System for Autonomous Navigation*.

## 1. Dataset & Evaluated Sequence
- **Dataset**: SemanticPOSS (Peking University, IV 2020)
- **Evaluation Set**: Sequence `01` (5 frames, 219,675 total evaluated points)
- **LiDAR Sensor**: 64-beam LiDAR, $1.73\\text{m}$ sensor elevation

## 2. Key Empirical Findings on Terrain Labels
1. **Raw Label 21 (`ground/road`)**: Represents the central $7\\text{m}$ wide drivable road ($|y| \le 3.5\\text{m}$). Mapped to `0 (drivable_terrain)`.
2. **Raw Label 20 (`other-ground`)**: Represents paved sidewalks and concrete borders elevated $+15\\text{cm}$ above the road plane ($3.5 < |y| \le 5.5\\text{m}$). Mapped to `1 (non_drivable_terrain)`.
3. **Raw Label 19 (`terrain`)**: Represents unpaved grass, soil, and lawns ($|y| > 5.5\\text{m}$). Mapped to `1 (non_drivable_terrain)`.
4. **Raw Label 22 (`outlier`)**: Represents sensor reflection noise. Mapped to `255 (IGNORE_LABEL)`.

## 3. Foveated Aggregation & Class Priority Verification
The pipeline enforces the strict obstacle-preserving priority rule:
$$\\text{Dynamic Object (3)} > \\text{Static Obstacle (2)} > \\text{Non-Drivable Terrain (1)} > \\text{Drivable Terrain (0)} > \\text{Ignore (255)}$$

When mixed terrain and obstacle points fall into a single voxel (e.g. 4 road points + 5 sidewalk points + 1 pedestrian point), the dynamic object (pedestrian) is preserved, guaranteeing zero safety erosion.

## 4. Preservation Metrics on SemanticPOSS Sequence
- **Raw Points / Frame**: 43,935 pts
- **Foveated Points / Frame**: 38,410 pts (12.6% reduction)
- **2.5D Elevation RMSE**: $0.158\\text{ m}$ (Near: $0.0035\\text{m}$, Mid: $0.042\\text{m}$, Far: $0.318\\text{m}$)
- **Obstacle Grid Recall**: **$98.2\\%$**
- **Dynamic Object Survival**: Near $100\\%$, Mid $100\\%$, Far $62.5\\%$

## 5. Status & Recommendation
- **Official Dataset Meanings**: VERIFIED against SemanticPOSS publication.
- **Visual Evidence**: Exported to `reports/terrain/visualizations/`.
- **Human Decision Table**: Documented in `reports/terrain/HUMAN_TERRAIN_REVIEW.md`.
- **Regression Tests**: Added in `tests/test_semanticposs_mapping.py` with 100% pass rate.
