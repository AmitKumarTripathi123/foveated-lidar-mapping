"""Tests for Phase 11.2 Predictor Output Contract and 2.5D Mapping Integration."""

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


class TestPhase11_2Mapping(unittest.TestCase):
    """Test suite for predictor contract, ML to mapping adapter, and GridMap25D integrity."""

    def test_01_predictor_contract(self):
        """Test 1: PointNet2Predictor returns [x, y, z, predicted_class, confidence]."""
        model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        predictor = PointNet2Predictor(model=model, device="cpu")
        pts = np.random.randn(64, 4).astype(np.float32)
        pred = predictor.predict(pts)

        self.assertEqual(pred["xyz"].shape, (64, 3))
        self.assertEqual(pred["predicted_class"].shape, (64,))
        self.assertEqual(pred["confidence"].shape, (64,))
        self.assertTrue(set(np.unique(pred["predicted_class"])).issubset({0, 1, 2, 3}))
        self.assertTrue(np.all((pred["confidence"] >= 0.0) & (pred["confidence"] <= 1.0)))

    def test_02_gridmap25d_layers(self):
        """Test 2: MLToMappingAdapter produces valid elevation, semantic, traversability, and confidence layers."""
        model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        predictor = PointNet2Predictor(model=model, device="cpu")
        pts = np.random.uniform(-10, 10, size=(128, 4)).astype(np.float32)
        pred = predictor.predict(pts)

        adapter = MLToMappingAdapter(resolution=1.0, bounds_x=(-10.0, 10.0), bounds_y=(-10.0, 10.0))
        grid = adapter.build_25d_grid(pred)

        self.assertIsInstance(grid, GridMap25D)
        self.assertEqual(grid.semantic_layer.shape, (20, 20))
        self.assertEqual(grid.traversability_layer.shape, (20, 20))
        self.assertEqual(grid.confidence_layer.shape, (20, 20))
        self.assertFalse(np.isnan(grid.confidence_layer).any())


if __name__ == "__main__":
    unittest.main()
