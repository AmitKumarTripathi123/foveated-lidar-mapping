# Phase 1 + Phase 2 — End-to-End Pipeline Results

**Pipeline Architecture**: `Raw SemanticPOSS` -> `Range Filter (100m)` -> `Distance Foveation` -> `FoveatedPointSegNet` -> `SemanticPrediction`  

---

## 1. Frame-by-Frame End-to-End Metrics

|   Frame ID | Raw Pts   | Foveated Pts   | Accuracy   | mIoU   | Drivable (0)   | Non-Drivable (1)   | Obstacle (2)   | Dynamic (3)   |   Mean Conf | Latency   |
|------------|-----------|----------------|------------|--------|----------------|--------------------|----------------|---------------|-------------|-----------|
|     000000 | 40,000    | 32,377         | 78.65%     | 52.69% | 27.81%         | 56.16%             | 87.82%         | 38.95%        |      0.7645 | 188.84 ms |
|     000001 | 40,000    | 32,397         | 79.12%     | 53.60% | 28.48%         | 56.32%             | 88.37%         | 41.24%        |      0.7639 | 185.87 ms |
|     000002 | 40,000    | 32,349         | 78.96%     | 53.22% | 27.29%         | 55.84%             | 88.57%         | 41.19%        |      0.7652 | 172.90 ms |
|     000003 | 40,000    | 32,337         | 79.16%     | 53.55% | 29.08%         | 56.69%             | 88.15%         | 40.27%        |      0.7642 | 169.49 ms |
|     000004 | 40,000    | 32,362         | 78.91%     | 53.05% | 27.93%         | 56.31%             | 88.11%         | 39.84%        |      0.7644 | 175.38 ms |

## 2. Global Multi-Frame Aggregate Metrics
- **Total Points Evaluated**: 161,822 across 5 frames
- **Overall Accuracy**: **78.96%**
- **Mean IoU (mIoU)**: **53.22%**
- **Static Obstacle IoU (2)**: **88.20%** (Precision: 98.23%, Recall: 89.63%)
- **Non-Drivable Terrain IoU (1)**: **56.27%** (Precision: 66.34%, Recall: 78.74%)
- **Dynamic Object IoU (3)**: **40.29%** (Precision: 46.01%, Recall: 76.40%)
- **Drivable Terrain IoU (0)**: **28.12%** (Precision: 50.25%, Recall: 38.97%)

## 3. Distance-Band Semantic Breakdown
- **Near Band (0–10m @ 0.05m)**: mIoU = **57.53%** (Drivable: 57.60%, Sidewalk: 13.90%)
- **Mid Band (10–40m @ 0.15m)**: mIoU = **48.63%** (Obstacles: 77.65%)
- **Far Band (40–100m @ 0.50m)**: mIoU = **46.75%** (Obstacles: 94.29%)
