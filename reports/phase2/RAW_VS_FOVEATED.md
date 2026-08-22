# Phase 2 — Raw vs Foveated Semantic Segmentation Experiment

**Experiment Objective**: Evaluate the exact same AI model on Raw LiDAR vs Distance-Foveated LiDAR to assess accuracy retention and computational speedup.

## 1. Empirical Comparison Table

| Evaluation Metric        | Raw LiDAR (No Foveation)   | Foveated LiDAR (0.05/0.15/0.50m)   | Delta / Gain   |
|--------------------------|----------------------------|------------------------------------|----------------|
| Point Count / Frame      | 40,000                     | 32,377                             | -19.1%         |
| Inference Latency        | 231.71 ms                  | 202.91 ms                          | 12.4% speedup  |
| Throughput (FPS)         | 4.3 FPS                    | 4.9 FPS                            | +15.2%         |
| Mean IoU (mIoU)          | 27.98%                     | 27.57%                             | -0.41%         |
| Drivable Terrain IoU     | 0.00%                      | 0.00%                              | +0.00%         |
| Non-Drivable Terrain IoU | 42.71%                     | 40.38%                             | -2.33%         |
| Static Obstacle IoU      | 69.20%                     | 69.88%                             | +0.68%         |
| Dynamic Object IoU       | 0.00%                      | 0.00%                              | +0.00%         |
| Overall Accuracy         | 61.66%                     | 64.34%                             | +2.68%         |

## 2. Distance-Band Semantic Performance (Foveated Model)

| Distance Band | Band mIoU | Drivable IoU | Non-Drivable IoU | Static Obstacle IoU | Dynamic Object IoU | Points Retained |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Near (0–10m @ 0.05m)** | **23.18%** | 0.00% | 39.46% | 53.25% | 0.00% | 1,934 |
| **Mid (10–40m @ 0.15m)** | **27.92%** | 0.00% | 43.93% | 67.74% | 0.00% | 13,816 |
| **Far (40–100m @ 0.50m)** | **26.91%** | 0.00% | 36.36% | 71.30% | 0.00% | 16,240 |
