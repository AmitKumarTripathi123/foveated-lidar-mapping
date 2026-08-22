# Phase 5 Reference Behavior Specification

## 1. Point Representation
```cpp
struct ClassifiedPoint {
    float x;
    float y;
    float z;
    float intensity;
    uint8_t class_id;
    float confidence;
};
```

## 2. Coordinate System & Reference Point
- Coordinate Frame: Sensor-centric LiDAR coordinates (+X forward, +Y left, +Z up).
- Origin: $(0, 0, 0)$ is the LiDAR sensor center.
- Operational Range: $[0.0, 100.0)\text{ meters}$ horizontal distance.

## 3. Distance Calculation
Horizontal 2D Euclidean distance:
$$r = \sqrt{x^2 + y^2}$$

## 4. Multi-Band Resolution Selection
Strict half-open intervals $[r_{\min}, r_{\max})$:
- `near_field`: $[0.0, 10.0)\text{ m} \implies s = 0.05\text{ m}$
- `mid_near_field`: $[10.0, 30.0)\text{ m} \implies s = 0.10\text{ m}$
- `mid_far_field`: $[30.0, 60.0)\text{ m} \implies s = 0.25\text{ m}$
- `far_field`: $[60.0, 100.0)\text{ m} \implies s = 0.50\text{ m}$
- Out of Range: $r < 0.0$ or $r \ge 100.0\text{ m}$ or non-finite ($\text{NaN}, \pm\infty$) $\implies$ Ignored.

## 5. 2D Cell Indexing Formula
Mathematical floor (handles negative coordinates without truncating toward zero):
$$i_x = \lfloor x / s \rfloor, \quad i_y = \lfloor y / s \rfloor$$

Bounds for cell $(i_x, i_y)$:
$$\text{min\_x} = i_x \cdot s, \quad \text{max\_x} = (i_x + 1) \cdot s$$
$$\text{min\_y} = i_y \cdot s, \quad \text{max\_y} = (i_y + 1) \cdot s$$

## 6. Cell Aggregation Behavior
For a set of $N$ points mapped to the same cell $(i_x, i_y)$ in band $B$:
1. **Count**: $N$
2. **Elevation Mean**: $\bar{z} = \frac{1}{N} \sum_{k=1}^N z_k$
3. **Elevation Min**: $z_{\min} = \min_{k} z_k$
4. **Elevation Max**: $z_{\max} = \max_{k} z_k$
5. **Confidence**: $\bar{c} = \frac{1}{N} \sum_{k=1}^N c_k$
6. **Semantic Class**: The class with highest obstacle priority:
   - Dynamic Object (3): Priority 4
   - Static Obstacle (2): Priority 3
   - Non-Drivable Terrain (1): Priority 2
   - Drivable Terrain (0): Priority 1
   - Ignore Label (255): Priority 0
7. **Traversability**:
   - $1.0$ if aggregated class is `DRIVABLE_TERRAIN` (0)
   - $0.2$ if aggregated class is `NON_DRIVABLE_TERRAIN` (1)
   - $0.0$ otherwise.

## 7. Output Format
Deterministic CSV format with header:
`band_name,ix,iy,resolution,point_count,elevation_mean,elevation_min,elevation_max,semantic_class,confidence,traversability`
Sorted deterministically by `(band_name, iy, ix)`.
