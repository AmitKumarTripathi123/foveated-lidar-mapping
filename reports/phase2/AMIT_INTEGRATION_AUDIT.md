# Phase 2 — Amit Kumar Tripathi Integration Audit

**Reference Repository**: [https://github.com/AmitKumarTripathi123/foveated-lidar-mapping](https://github.com/AmitKumarTripathi123/foveated-lidar-mapping)  
**Target Project**: Foveated 2.5D LiDAR Mapping System for Autonomous Navigation (Smart India Hackathon)  
**Audit Purpose**: Identify verified components from reference implementation, reconcile dataset & foveation parameters, and design Phase 2 AI/ML semantic segmentation without regression.

---

## 1. Architectural Component Comparison Matrix

| Component | Current Project (Phase 1) | Amit Repository (`foveated-lidar-mapping`) | Integrated Decision | Technical Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Dataset & Sensor** | SemanticKITTI + Synthetic 64-beam Velodyne LiDAR | **SemanticPOSS (Hesai Pandar40, 40-beam LiDAR)**, $1800 \times 40$ resolution, 10 Hz | **Adopt 40-beam SemanticPOSS** for Phase 2; maintain Phase 1 backward compatibility | Standardizes dataset on Hesai 40-beam sensor used across SIH navigation benchmarks. |
| **Foveation Bands** | 0-10m @ 0.05m, 10-40m @ 0.15m, 40-100m @ 0.50m | **0-10m @ 0.05m, 10-40m @ 0.15m, 40-100m @ 0.50m** | **Preserve 3-band configuration exactly** | Both repositories share identical, frozen distance-adaptive voxel dimensions. |
| **Horizontal Range Filter** | $r = \sqrt{x^2 + y^2} \le 100.0\text{m}$ | $r = \sqrt{x^2 + y^2} \le 100.0\text{m}$ | **Preserve 100m radial filter** | Ensures consistent 2D horizontal range clipping without 3D spherical distortion. |
| **Voxel Aggregation Baseline** | Vectorized multi-policy (`obstacle_preserving`, `centroid`, `majority`, `nearest`) | First-point selection in voxel hash (`amit_first_point`) | **Implement modular aggregation abstraction**; default baseline supports both `amit_first_point` and `obstacle_preserving` | Allows direct comparison between simple first-point retention and priority-based obstacle preservation. |
| **Label Mapping Architecture** | Authoritative YAML mapper (`configs/semanticposs_mapping.yaml`) mapping 21->0, 20->1, 19->1, 22->255 | `class_map.py` (`POSS_CLASS_REMAP`) mapping raw POSS classes to 4 super-classes | **Single Authoritative Adapter (`SEMANTICPOSS_TO_PROJECT`)** in Phase 2 dataset | Eliminates double label remapping by extracting raw 16-bit semantic IDs (`label & 0xFFFF`) and applying mapping exactly once. |
| **Target Super-Classes** | 0: drivable, 1: non-drivable, 2: static obstacle, 3: dynamic object, 255: ignore | 0: drivable, 1: non-drivable, 2: static obstacle, 3: dynamic object, 255: ignore | **Standard 4-class navigation contract** (0, 1, 2, 3) with 255 excluded from loss | Full alignment with project ICD and downstream 2.5D elevation grid requirements. |
| **Dataset Splits** | Sequence 00 / 01 | Train: `00, 01, 03, 04, 05`, Val: `02` | **Sequence-based non-leaking splits** (`Phase2Dataset`) | Prevents data leakage by splitting at sequence/frame boundaries rather than random point sampling. |
| **AI Model Architecture** | N/A (Phase 1 foundation) | Dataset/Preprocessing pipeline | **Foveated Point Segmentation Network (`FoveatedPointSegNet`)** with distance-aware feature conditioning | Produces 4-class softmax probabilities $[N, 4]$, predicted classes $[N]$, and confidence $[N] = \max(P)$. |
| **Output Contract** | `PointCloudFrame` | NumPy `.npy` arrays | **`SemanticPrediction` Data Class** | Formal interface between Phase 2 AI predictions and Phase 3 2.5D mapping engine. |

---

## 2. Double-Remapping Prevention Verification

In Amit's repository, `FoveatedLidarDataset` loads `.label` files, unpacks the lower 16 bits (`np.fromfile(...) & 0xFFFF`), and calls `remap_labels(...)`.

To ensure our pipeline never applies label mapping twice:
1. **Raw SemanticPOSS Ingestion**: `Phase2Dataset` reads raw `.bin` (float32 x,y,z,intensity) and raw `.label` (uint32).
2. **Single Transformation Point**: Raw labels are masked with `0xFFFF` and mapped via `SEMANTICPOSS_TO_PROJECT` directly into super-classes $[0, 1, 2, 3, 255]$.
3. **Model Consumption**: The AI model exclusively receives points $[N, 4]$ and target labels $[N] \in \{0, 1, 2, 3, 255\}$.
