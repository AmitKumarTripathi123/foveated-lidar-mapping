# Phase 1 + Phase 2 — End-to-End Pipeline Results

**Pipeline Architecture**: `Raw SemanticPOSS` -> `Range Filter (100m)` -> `Distance Foveation` -> `FoveatedPointSegNet` -> `SemanticPrediction`  

---

## 1. Frame-by-Frame End-to-End Metrics

|   Frame ID | Raw Pts   | Foveated Pts   | Accuracy   | mIoU   | Drivable IoU   | Non-Drivable IoU   | Obstacle IoU   |   Mean Conf | Latency   |
|------------|-----------|----------------|------------|--------|----------------|--------------------|----------------|-------------|-----------|
|     000000 | 40,000    | 32,377         | 64.34%     | 27.57% | 0.00%          | 40.38%             | 69.88%         |      0.3871 | 184.28 ms |
|     000001 | 40,000    | 32,397         | 64.16%     | 27.48% | 0.00%          | 40.08%             | 69.86%         |      0.3873 | 179.16 ms |
|     000002 | 40,000    | 32,349         | 64.26%     | 27.51% | 0.00%          | 40.41%             | 69.63%         |      0.3867 | 167.78 ms |
|     000003 | 40,000    | 32,337         | 64.09%     | 27.43% | 0.00%          | 40.32%             | 69.40%         |      0.3865 | 168.38 ms |
|     000004 | 40,000    | 32,362         | 64.42%     | 27.57% | 0.00%          | 40.54%             | 69.75%         |      0.3868 | 167.56 ms |

## 2. Global Multi-Frame Aggregate Metrics
- **Total Points Evaluated**: 161,822
- **Overall Accuracy**: **64.25%**
- **Mean IoU (mIoU)**: **27.51%**
- **Drivable Terrain IoU**: **0.00%**
- **Non-Drivable Terrain IoU**: **40.35%**
- **Static Obstacle IoU**: **69.70%**
- **Dynamic Object IoU**: **0.00%**

## 3. Distance-Band Semantic Analysis
- **Near Band (0–10m @ 0.05m)**: mIoU = **24.80%** (Drivable: 0.00%, Sidewalk: 40.36%)
- **Mid Band (10–40m @ 0.15m)**: mIoU = **27.82%** (Obstacles: 67.22%)
- **Far Band (40–100m @ 0.50m)**: mIoU = **26.77%** (Obstacles: 71.20%)
