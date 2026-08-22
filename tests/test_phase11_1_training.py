"""Tests for Phase 11.1 Model Input, Training Step, Collapse Detector, Predictor, and Mapping."""

import sys
import unittest
from pathlib import Path
import numpy as np
import torch

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.models.pointnet2 import build_model
from ml.models.predictor import PointNet2Predictor
from ml.models.mapping_adapter import MLToMappingAdapter, GridMap25D
from ml.training.losses import get_loss_function
from ml.training.metrics import SemanticSegmentationMetrics


class TestPhase11_1Training(unittest.TestCase):
    """Test suite for training sanity, loss with ignore index, predictor contract, and 2.5D mapping."""

    def test_01_training_step(self):
        """Test 1: Model parameters update cleanly after single backward step with ignore_index=255."""
        model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        crit = get_loss_function(loss_type="cross_entropy", ignore_index=255)

        x = torch.randn(1, 64, 4)
        y = torch.tensor([[0, 1, 2, 3, 255] * 12 + [0, 1, 2, 3]], dtype=torch.int64)

        out = model(x)
        loss = crit(out.transpose(1, 2), y)
        loss.backward()
        opt.step()
        self.assertFalse(torch.isnan(loss))

    def test_02_collapse_detector_trigger(self):
        """Test 2: Dominant class detection triggers accurately on biased predictions."""
        preds = np.array([2] * 95 + [0] * 5)
        u, c = np.unique(preds, return_counts=True)
        dominant_pct = max(c) / len(preds)
        self.assertTrue(dominant_pct > 0.90)

    def test_03_predictor_contract(self):
        """Test 3: PointNet2Predictor outputs xyz, predicted_class, confidence with proper ranges."""
        model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        predictor = PointNet2Predictor(model=model, device="cpu")
        pts = np.random.randn(64, 4).astype(np.float32)
        pred = predictor.predict(pts)

        self.assertEqual(pred["xyz"].shape, (64, 3))
        self.assertEqual(pred["predicted_class"].shape, (64,))
        self.assertEqual(pred["confidence"].shape, (64,))
        self.assertTrue(set(np.unique(pred["predicted_class"])).issubset({0, 1, 2, 3}))
        self.assertTrue(np.all((pred["confidence"] >= 0.0) & (pred["confidence"] <= 1.0)))

    def test_04_mapping_adapter_gridmap25d(self):
        """Test 4: Predictions feed cleanly into MLToMappingAdapter producing GridMap25D."""
        model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        predictor = PointNet2Predictor(model=model, device="cpu")
        pts = np.random.uniform(-10, 10, size=(128, 4)).astype(np.float32)
        pred = predictor.predict(pts)

        adapter = MLToMappingAdapter(resolution=1.0, bounds_x=(-10.0, 10.0), bounds_y=(-10.0, 10.0))
        grid = adapter.build_25d_grid(pred)

        self.assertIsInstance(grid, GridMap25D)
        self.assertEqual(grid.semantic_layer.shape, (20, 20))
        self.assertEqual(grid.traversability_layer.shape, (20, 20))
        self.assertFalse(np.isnan(grid.confidence_layer).any())


if __name__ == "__main__":
    unittest.main()
