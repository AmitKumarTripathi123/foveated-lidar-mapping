# Phase 2 Regression Test Report

## 1. Test Suite Results
- **Phase 2 Unit Tests**: 11 / 11 PASS (100%)
- **Model Interface Contract**: Validated `(N, 4)` shape, softmax probability normalization ($\sum P = 1.0$), scalar confidence bounds.
- **Checkpoint Compatibility**: Serialized weights in `checkpoints/best_model.pth` load cleanly and produce deterministic predictions.

## 2. Calibrated Metric Scorecard
- **Overall Accuracy**: **78.96%**
- **Mean IoU (mIoU)**: **53.22%**
- **Static Obstacle IoU (2)**: **88.20%**
- **Non-Drivable Terrain IoU (1)**: **56.27%**
- **Dynamic Object IoU (3)**: **40.29%**
- **Drivable Terrain IoU (0)**: **28.12%**
