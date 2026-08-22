"""Tests for Phase 11 Foveated vs Full-Resolution Experiments & Integration."""

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
from scripts.compare_foveated_vs_full import run_comparison


class TestPhase11FoveatedVsFull(unittest.TestCase):
    """Test suite for controlled comparative benchmark and mapping integration."""

    @classmethod
    def setUpClass(cls):
        """Set up dataset paths."""
        cls.bin_file = repo_root / "dataset/sequences/00/velodyne/000000.bin"
        cls.lbl_file = repo_root / "dataset/sequences/00/labels/000000.label"

    def test_01_controlled_comparison_run(self):
        """Test 1: Side-by-side comparison script runs and measures positive reduction."""
        res = run_comparison(self.bin_file, self.lbl_file, num_points=256, iterations=1, device_str="cpu")
        self.assertEqual(res["raw_points"], 66658)
        self.assertEqual(res["foveated_points"], 50571)
        self.assertAlmostEqual(res["point_reduction_pct"], 24.13, places=1)
        self.assertGreater(res["full_latency_ms"], 0.0)
        self.assertGreater(res["fov_latency_ms"], 0.0)

    def test_02_mapping_adapter_full(self):
        """Test 2: Full resolution predictions feed into MLToMappingAdapter."""
        model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        predictor = PointNet2Predictor(model=model, device="cpu")
        pts = np.random.uniform(-30, 30, size=(128, 4)).astype(np.float32)
        pred = predictor.predict(pts)

        adapter = MLToMappingAdapter()
        grid = adapter.build_25d_grid(pred)
        self.assertIsInstance(grid, GridMap25D)

    def test_03_mapping_adapter_foveated(self):
        """Test 3: Foveated predictions feed into MLToMappingAdapter."""
        model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        predictor = PointNet2Predictor(model=model, device="cpu")
        pts = np.random.uniform(-30, 30, size=(128, 4)).astype(np.float32)
        pred = predictor.predict(pts)

        adapter = MLToMappingAdapter()
        grid = adapter.build_25d_grid(pred)
        self.assertIsInstance(grid, GridMap25D)


if __name__ == "__main__":
    unittest.main()
