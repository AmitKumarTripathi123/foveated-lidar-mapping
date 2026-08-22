# Phase 2 Regression Test Report

## 1. Test Suite Results
- **Phase 2 Unit Tests**: 11 / 11 PASS (100%)
- **Model Interface Contract**: Validated `(N, 4)` shape, softmax probability normalization ($\sum P = 1.0$), scalar confidence bounds.
- **Checkpoint Compatibility**: Serialized weights in `checkpoints/best_model.pth` load cleanly and produce deterministic predictions.

## 2. Metric Scorecard
- **Overall Accuracy**: 77.73%
- **Mean IoU (mIoU)**: 44.92% (Baseline)
- **Static Obstacle IoU**: 85.67%
- **Non-Drivable Terrain IoU**: 48.76%
- **Drivable Terrain IoU**: 39.36%
- **Phase 2 Integrity**: Fully verified.
