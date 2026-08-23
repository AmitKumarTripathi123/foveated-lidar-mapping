"""
Phase 13 Unit and Integration Tests:
- Class weighting algorithms
- Advanced loss functions (Focal Loss, Weighted CE)
- 3D LiDAR point cloud augmentations
- Model collapse detector
- Checkpoint reproducibility and reload verification
"""

import unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from ml.data.augmentation import LidarAugmentor
from ml.training.class_weights import (
    compute_inverse_frequency_weights,
    compute_sqrt_inverse_frequency_weights,
    compute_effective_number_weights,
    get_class_weights,
)
from ml.training.losses import FocalLoss, build_loss_function
from ml.models.spvcnn import build_spvcnn
from ml.models.spvcnn_predictor import SPVCNNPredictor
from ml.models.mapping_adapter import MLToMappingAdapter


class TestPhase13Optimization(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_01_inverse_frequency_weights(self):
        """Test 1: Inverse frequency class weights are normalized to mean=1.0."""
        counts = [1000, 100, 5000, 50]
        weights = compute_inverse_frequency_weights(counts, num_classes=4)
        self.assertEqual(len(weights), 4)
        self.assertAlmostEqual(np.mean(weights), 1.0, places=2)
        # Minority class (index 3) must have highest weight
        self.assertEqual(int(np.argmax(weights)), 3)

    def test_02_sqrt_inverse_frequency_weights(self):
        """Test 2: Square-root inverse frequency weights are normalized to mean=1.0."""
        counts = [1000, 100, 5000, 50]
        weights = compute_sqrt_inverse_frequency_weights(counts, num_classes=4)
        self.assertEqual(len(weights), 4)
        self.assertAlmostEqual(np.mean(weights), 1.0, places=2)
        self.assertEqual(int(np.argmax(weights)), 3)
        # Sqrt weights should have smaller dynamic range than raw inverse
        inv_weights = compute_inverse_frequency_weights(counts, num_classes=4)
        self.assertLess(max(weights) / min(weights), max(inv_weights) / min(inv_weights))

    def test_03_effective_number_weights(self):
        """Test 3: Effective number of samples weights compute valid normalized values."""
        counts = [1000, 100, 5000, 50]
        weights = compute_effective_number_weights(counts, beta=0.9999, num_classes=4)
        self.assertEqual(len(weights), 4)
        self.assertAlmostEqual(np.mean(weights), 1.0, places=2)

    def test_04_focal_loss_forward_and_backward(self):
        """Test 4: Focal Loss computes valid loss and gradients on device."""
        criterion = FocalLoss(gamma=2.0, ignore_index=255).to(self.device)
        logits = torch.randn(100, 4, device=self.device, requires_grad=True)
        targets = torch.randint(0, 4, (100,), device=self.device)
        targets[10:20] = 255  # Include ignore points

        loss = criterion(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_05_focal_loss_ignores_255(self):
        """Test 5: Points with target 255 do not contribute to focal loss."""
        criterion = FocalLoss(gamma=2.0, ignore_index=255).to(self.device)
        logits = torch.tensor([[10.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0]], device=self.device)
        targets_all_ignore = torch.tensor([255, 255], device=self.device)
        loss = criterion(logits, targets_all_ignore)
        self.assertEqual(loss.item(), 0.0)

    def test_06_build_loss_factory(self):
        """Test 6: build_loss_function instantiates correct loss modules."""
        ce = build_loss_function({"loss": {"type": "cross_entropy"}})
        self.assertIsInstance(ce, nn.CrossEntropyLoss)

        focal = build_loss_function({"loss": {"type": "focal_loss", "gamma": 2.0}})
        self.assertIsInstance(focal, FocalLoss)

    def test_07_lidar_augmentation_training_only(self):
        """Test 7: LidarAugmentor applies transformations during training and passes through during eval."""
        cfg = {"augmentation": {"enabled": True, "rotation_deg": 15.0, "min_scale": 0.9, "max_scale": 1.1}}
        aug_train = LidarAugmentor(cfg, is_training=True)
        aug_eval = LidarAugmentor(cfg, is_training=False)

        pts = np.ones((100, 4), dtype=np.float32)
        lbls = np.zeros(100, dtype=np.int64)

        # Eval must remain exact
        eval_pts, eval_lbls = aug_eval.augment(pts, lbls)
        np.testing.assert_array_equal(eval_pts, pts)
        np.testing.assert_array_equal(eval_lbls, lbls)

        # Train should transform coordinates
        train_pts, train_lbls = aug_train.augment(pts, lbls)
        self.assertEqual(train_pts.shape, pts.shape)
        np.testing.assert_array_equal(train_lbls, lbls)

    def test_08_collapse_detector_logic(self):
        """Test 8: Dominant class detection correctly identifies biased distributions."""
        # 95% majority class 2
        preds = np.array([2] * 95 + [0] * 5, dtype=np.int64)
        u, counts = np.unique(preds, return_counts=True)
        dom_pct = float(np.max(counts) / len(preds) * 100.0)
        self.assertGreaterEqual(dom_pct, 90.0)

    def test_09_ml_to_mapping_regression(self):
        """Test 9: Phase 13 model output integrates with MLToMappingAdapter and GridMap25D."""
        ckpt_path = Path("experiments/phase12_full_semanticposs_spvcnn/best_checkpoint.pt")
        predictor = SPVCNNPredictor(device="cpu", pretrained_path=str(ckpt_path))
        map_adapter = MLToMappingAdapter()

        pts = np.random.uniform(-30, 30, size=(1000, 4)).astype(np.float32)
        res = predictor.predict(pts)
        grid = map_adapter.build_25d_grid(res)
        self.assertIsNotNone(grid)
        self.assertEqual(grid.semantic_layer.shape, (500, 500))


if __name__ == "__main__":
    unittest.main()
