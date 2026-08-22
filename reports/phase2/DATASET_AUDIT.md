# Phase 2 — Dataset Audit Report

**Dataset**: SemanticPOSS (Hesai 40-beam LiDAR)  
**Configuration**: 40 channels, 10Hz, horizontal range $0 \le r \le 100\text{m}$  
**Total Evaluated Points**: 161,822 across 5 sequence frames  

## Super-Class Distribution

| Super-Class             |   Point Count | Percentage   |
|-------------------------|---------------|--------------|
| 0: drivable_terrain     |         22775 | 14.07%       |
| 1: non_drivable_terrain |         41312 | 25.53%       |
| 2: static_obstacle      |         88070 | 54.42%       |
| 3: dynamic_object       |          7721 | 4.77%        |
| 255: IGNORE_LABEL       |          1944 | 1.20%        |

## Data Validation Checklist
- Points shape: `[N, 4]` (float32 x, y, z, intensity)
- Labels shape: `[N]` (integer super-classes in (0, 1, 2, 3, 255))
- Point/label length consistency: 100% PASS
- NaN/Inf invalid points: 0 (0.0%)
- Single authoritative label adapter: VERIFIED
