# Phase 2 — Foveated 2.5D Grid Correctness & Validation Report

**Specification**: Phase-2 Frozen 4-Band Distance-Aware 2.5D Grid Map  
**Sensor**: 40-beam Hesai Pandar40 LiDAR ($1800 \times 40$ resolution, 10 Hz)  
**Dataset**: 5 sequential SemanticPOSS evaluation scans  

---

## 1. Spatial Grid Specification

| Distance Band | Radial Range $r = \sqrt{x^2 + y^2}$ | Grid Resolution | Interval Type | Operational Status |
| :--- | :--- | :--- | :--- | :--- |
| **Near Field** | $[0.0, 10.0)\text{ m}$ | **0.05 m (5 cm)** | Half-open | Active |
| **Mid-Near Field** | $[10.0, 30.0)\text{ m}$ | **0.10 m (10 cm)** | Half-open | Active |
| **Mid-Far Field** | $[30.0, 60.0)\text{ m}$ | **0.25 m (25 cm)** | Half-open | Active |
| **Far Field** | $[60.0, 100.0)\text{ m}$ | **0.50 m (50 cm)** | Half-open | Active |
| **Out of Range** | $[100.0, \infty)\text{ m}$ | *Filtered Out* | Filtered | Discarded |

---

## 2. Frame-by-Frame Real-Data Benchmark (5 Scans)

|   Frame ID | Input Points   | 2.5D Cells   | Compression Ratio   | Spatial Reduction   | Alignment Invariant   | Pipeline Latency   |
|------------|----------------|--------------|---------------------|---------------------|-----------------------|--------------------|
|     000000 | 40,000         | 28,360       | 1.41x               | 29.1%               | 100.0% Verified       | 372.00 ms          |
|     000001 | 40,000         | 28,343       | 1.41x               | 29.1%               | 100.0% Verified       | 356.53 ms          |
|     000002 | 40,000         | 28,309       | 1.41x               | 29.2%               | 100.0% Verified       | 364.47 ms          |
|     000003 | 40,000         | 28,290       | 1.41x               | 29.3%               | 100.0% Verified       | 363.98 ms          |
|     000004 | 40,000         | 28,237       | 1.42x               | 29.4%               | 100.0% Verified       | 338.99 ms          |

---

## 3. Band-by-Band Point & Cell Distribution

| Distance Band   | Resolution   | Total Input Points   | 2.5D Spatial Cells   | Compression Ratio   | Band Reduction   |
|-----------------|--------------|----------------------|----------------------|---------------------|------------------|
| near_field      | 5 cm         | 9,849                | 9,648                | 1.02x               | 2.0%             |
| mid_near_field  | 10 cm        | 47,484               | 45,039               | 1.05x               | 5.1%             |
| mid_far_field   | 25 cm        | 71,275               | 54,149               | 1.32x               | 24.0%            |
| far_field       | 50 cm        | 71,392               | 32,703               | 2.18x               | 54.2%            |

---

## 4. Fundamental 2.5D Spatial Invariants Proven

1. **2D Spatial Identity vs Z Elevation Attribute**:
   $$\forall p = (x_p, y_p, z_p), \quad i_x = \lfloor x_p / s \rfloor, \quad i_y = \lfloor y_p / s \rfloor$$
   Elevation $z$ is never used for cell indexing. It is aggregated as `elevation_mean = mean(z)`, `elevation_min = min(z)`, `elevation_max = max(z)`.
2. **Mathematical Projection Bounds**:
   $$\forall p \in \text{Cell}(i_x, i_y), \quad i_x \cdot s \le x_p < (i_x + 1) \cdot s \quad \text{and} \quad i_y \cdot s \le y_p < (i_y + 1) \cdot s$$
   Verified across **200,000 consecutive points** with **0 violations (100% compliance)**.
3. **Obstacle-Preserving Semantic Priority**:
   $$\text{dynamic\_object (3)} > \text{static\_obstacle (2)} > \text{non\_drivable (1)} > \text{drivable (0)} > \text{ignore (255)}$$
   Tested across all multi-label cell combinations (road+obstacle, road+dynamic, obstacle+dynamic).
4. **Empty Cell Semantics**:
   Unobserved spatial cells strictly return `state = CellState.UNKNOWN` with `point_count = 0` and `elevation = NaN`, preventing false `FREE` space hallucination.

---

## 5. Phase-2 Foveated Grid Correctness Gate Decision

```text
PHASE 2 FOVEATED GRID CORRECTNESS: PASS
```
