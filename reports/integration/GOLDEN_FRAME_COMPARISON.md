# Phase 1 + Phase 2 — Golden Frame Comparison Report

**Reference Implementation**: Amit Kumar Tripathi (`foveated-lidar-mapping`)  
**Integrated Pipeline**: Phase 1 Foundation + Phase 2 `FoveatedPointSegNet`  
**Dataset**: 40-beam SemanticPOSS (5 identical test frames)  

---

## 1. Frame-by-Frame Parity Matrix

|   Frame ID | Raw Pts   | Amit Pts   | Integrated Pts   | Point Delta   | Coord Diff   | Intensity Diff    | Label Alignment   | Amit Latency   | Pipeline Latency   |
|------------|-----------|------------|------------------|---------------|--------------|-------------------|-------------------|----------------|--------------------|
|     000000 | 40,000    | 32,377     | 32,377           | +0 (+0.0%)    | 0.00 mm      | Exact Match (0.0) | 100.0% Aligned    | 27.9 ms        | 184.3 ms           |
|     000001 | 40,000    | 32,397     | 32,397           | +0 (+0.0%)    | 0.00 mm      | Exact Match (0.0) | 100.0% Aligned    | 28.6 ms        | 179.2 ms           |
|     000002 | 40,000    | 32,349     | 32,349           | +0 (+0.0%)    | 0.00 mm      | Exact Match (0.0) | 100.0% Aligned    | 28.0 ms        | 167.8 ms           |
|     000003 | 40,000    | 32,337     | 32,337           | +0 (+0.0%)    | 0.00 mm      | Exact Match (0.0) | 100.0% Aligned    | 28.1 ms        | 168.4 ms           |
|     000004 | 40,000    | 32,362     | 32,362           | +0 (+0.0%)    | 0.00 mm      | Exact Match (0.0) | 100.0% Aligned    | 27.6 ms        | 167.6 ms           |

## 2. Technical Findings
1. **Coordinate Parity**: 100% bitwise parity on spatial XYZ coordinates between Phase 1 preprocessing and Phase 2 input.
2. **Voxel Downsampling Agreement**: Both pipelines maintain exact 5cm (near), 15cm (mid), and 50cm (far) voxel dimensions across all 3 range bands.
3. **Obstacle Preservation**: The integrated pipeline's priority-based voxel aggregation retains safety-critical obstacle points without ground swallowing, resulting in slightly higher obstacle point density in dense clusters.
