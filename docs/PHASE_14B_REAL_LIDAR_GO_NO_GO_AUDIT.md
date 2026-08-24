# Phase 14B — 2.5D Grid Real LiDAR GO/NO-GO Audit Report

## 1. Executive Summary
This document records the forensic **GO / NO-GO Software Audit** for the **Foveated 2.5D LiDAR Grid Engine** evaluated against real LiDAR sensor data (SemanticPOSS / KITTI real scan sequence 00, 488 frames).

---

## 2. Real LiDAR Golden Frame Specifications (Sequence 00 / Frame 000000)

* **Source File**: `dataset/sequences/00/velodyne/000000.bin`
* **File SHA256**: `d87d504c4b32505ddda250c691b9af243cbe0c5f43ce7c56a4ca51619e13a4dd`
* **Raw Point Count**: **66,658 points**
* **Point Memory Layout**: `float32` $[x, y, z, \text{intensity}]$ (16 bytes/point)
* **Coordinate Bounds**:
  * $X \in [-182.79\text{ m}, +132.43\text{ m}]$
  * $Y \in [-154.79\text{ m}, +155.77\text{ m}]$
  * $Z \in [-6.00\text{ m}, +12.66\text{ m}]$
* **Range Accepted ($[0, 100\text{ m})$)**: **66,021 points** (637 points rejected as out-of-range beyond 100m)
* **Grid Populated Cells**: **35,726 cells**
* **Unexplained Point Loss**: **0 points** (100% exact point conservation)

---

## 3. Resolution & 5cm Fundamental Lattice Verification

The real LiDAR points distribute across the distance-aware bands as follows:

$$\begin{aligned}
\text{Near-Field } [0\text{ m}, 10\text{ m}) &\implies \text{Resolution } 0.05\text{ m (5 cm)}: & 10,695\text{ points} \\
\text{Mid-Near-Field } [10\text{ m}, 30\text{ m}) &\implies \text{Resolution } 0.10\text{ m (10 cm)}: & 36,991\text{ points} \\
\text{Mid-Far-Field } [30\text{ m}, 60\text{ m}) &\implies \text{Resolution } 0.25\text{ m (25 cm)}: & 14,760\text{ points} \\
\text{Far-Field } [60\text{ m}, 100\text{ m}) &\implies \text{Resolution } 0.50\text{ m (50 cm)}: & 3,575\text{ points}
\end{aligned}$$

All cell bounds rigorously satisfy the $0.05\text{ m}$ ($5\text{ cm}$) fundamental lattice quantum.

---

## 4. Multi-Frame Replay Latency & Stability (100 Consecutive Real Frames)

Across 100 consecutive real LiDAR scans:
* **Mean Latency**: **$18.38\text{ ms}$** ($54.4\text{ FPS}$)
* **Median Latency (P50)**: **$18.27\text{ ms}$**
* **P95 Latency**: **$20.93\text{ ms}$**
* **P99 Latency**: **$21.70\text{ ms}$**
* **Standard Deviation**: **$1.39\text{ ms}$**
* **Memory RSS Growth**: **$0.00\text{ MB}$** (Zero memory leak across 100 frames)

---

## 5. Visualizations

Generated plots are permanently archived in `docs/phase14b_real_lidar_plots/`:
1. `vis_1_raw_pointcloud.png` (Raw LiDAR point cloud BEV)
2. `vis_2_foveated_grid.png` (Multi-resolution foveated occupancy)
3. `vis_3_elevation_map.png` (2.5D mean elevation field)
4. `vis_4_height_range_map.png` (Obstacle vertical span $\Delta z$)
5. `vis_5_dominant_semantic_map.png` (Categorical dominant class map)
6. `vis_6_semantic_probability_map.png` (Confidence probability field)
7. `vis_7_combined_25d_semantic_elevation.png` (Unified 3D 2.5D surface)

---

## 6. Real-Time Hardware Verdict
* **GPU Pipeline (CUDA FP16 + C++ Grid)**: **$\approx 23.59\text{ ms}$** ($>42\text{ FPS}$, $<50\text{ ms}$ target met).
* **CPU Pipeline (SPVCNN on CPU)**: **$\approx 150\text{–}250\text{ ms}$** (Target not met on CPU).
