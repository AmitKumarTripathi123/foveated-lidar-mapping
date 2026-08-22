# Phase 1 + Phase 2 — Golden Frame Comparison Report

**Reference Implementation**: Amit Kumar Tripathi (`foveated-lidar-mapping`)  
**Integrated Pipeline**: Phase 1 Foundation + Phase 2 `FoveatedPointSegNet`  
**Dataset**: 40-beam SemanticPOSS (5 identical test frames)  

---

## 1. Frame-by-Frame Parity Matrix

|   Frame ID | Raw Pts   | Amit Pts   | Project Pts   | Point Delta   | XYZ Max Err   | Intensity Max Err   | Label Diff     | Status   |
|------------|-----------|------------|---------------|---------------|---------------|---------------------|----------------|----------|
|     000000 | 40,000    | 32,377     | 32,377        | +0 (+0.0%)    | 0.00 mm       | Exact Match (0.0)   | 100.0% Aligned | PASS     |
|     000001 | 40,000    | 32,397     | 32,397        | +0 (+0.0%)    | 0.00 mm       | Exact Match (0.0)   | 100.0% Aligned | PASS     |
|     000002 | 40,000    | 32,349     | 32,349        | +0 (+0.0%)    | 0.00 mm       | Exact Match (0.0)   | 100.0% Aligned | PASS     |
|     000003 | 40,000    | 32,337     | 32,337        | +0 (+0.0%)    | 0.00 mm       | Exact Match (0.0)   | 100.0% Aligned | PASS     |
|     000004 | 40,000    | 32,362     | 32,362        | +0 (+0.0%)    | 0.00 mm       | Exact Match (0.0)   | 100.0% Aligned | PASS     |

## 2. Technical Findings
1. **Coordinate Parity**: 100% bitwise parity on spatial XYZ coordinates between Phase 1 preprocessing and Phase 2 input (XYZ max error = 0.00 mm).
2. **Intensity Parity**: Preserved exactly in normalized $[0, 1]$ float32 range without double-scaling.
3. **Voxel Downsampling Agreement**: Both pipelines maintain exact 5cm (near), 15cm (mid), and 50cm (far) voxel dimensions across all 3 range bands.
4. **Voxel Aggregation Distinction**: Integrated pipeline supports both `amit_first_point` and `obstacle_preserving` priority aggregation to prevent small obstacle point erasure in multi-point cells.
