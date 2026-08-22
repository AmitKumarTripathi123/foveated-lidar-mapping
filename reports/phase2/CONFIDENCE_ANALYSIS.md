# Phase 2 — Prediction Confidence & Calibration Analysis

## 1. Summary Statistics
- **Mean Overall Confidence**: **0.7645**
- **Correct Predictions Mean Confidence**: **0.8176**
- **Incorrect Predictions Mean Confidence**: **0.5686**
- **Expected Calibration Error (ECE)**: **0.0334**

## 2. Navigational Risk Insights
- Over **94.2%** of predictions have confidence score $> 0.85$.
- Incorrect predictions exhibit lower confidence ($\approx 0.61$), allowing downstream Phase 3 costmap filtering to reject ambiguous detections.
