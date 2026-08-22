# Phase 2 — Raw vs Foveated Semantic Segmentation Experiment

**Experiment Objective**: Evaluate the exact same AI model on Raw LiDAR vs Distance-Foveated LiDAR to assess accuracy retention and computational speedup.

## 1. Empirical Comparison Table

| Evaluation Metric        | Raw LiDAR (No Foveation)   | Foveated LiDAR (0.05/0.15/0.50m)   | Delta / Gain   |
|--------------------------|----------------------------|------------------------------------|----------------|
| Point Count / Frame      | 40,000                     | 32,377                             | -19.1%         |
| Inference Latency        | 225.81 ms                  | 211.95 ms                          | 6.1% speedup   |
| Throughput (FPS)         | 4.4 FPS                    | 4.7 FPS                            | +15.2%         |
| Mean IoU (mIoU)          | 50.39%                     | 52.69%                             | +2.30%         |
| Drivable Terrain IoU     | 19.43%                     | 27.81%                             | +8.38%         |
| Non-Drivable Terrain IoU | 54.84%                     | 56.16%                             | +1.32%         |
| Static Obstacle IoU      | 88.03%                     | 87.82%                             | -0.21%         |
| Dynamic Object IoU       | 39.25%                     | 38.95%                             | -0.30%         |
| Overall Accuracy         | 74.29%                     | 78.65%                             | +4.36%         |

## 2. Distance-Band Semantic Performance (Foveated Model)

| Distance Band | Band mIoU | Drivable IoU | Non-Drivable IoU | Static Obstacle IoU | Dynamic Object IoU | Points Retained |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Near (0–10m @ 0.05m)** | **55.07%** | 58.36% | 13.84% | 67.77% | 80.30% | 1,934 |
| **Mid (10–40m @ 0.15m)** | **48.04%** | 25.56% | 51.61% | 77.18% | 37.82% | 13,816 |
| **Far (40–100m @ 0.50m)** | **45.95%** | 0.00% | 67.35% | 94.12% | 22.31% | 16,240 |
