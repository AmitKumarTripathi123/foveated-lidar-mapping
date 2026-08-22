"""Integration tests for SPVCNN Predictor and Frozen ML -> Mapping Contract (Phase 12)."""

import sys
import unittest
from pathlib import Path
import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.dataset import load_point_cloud
from ml.data.amit_adapter import FoveatedVoxelSampler
from ml.models.spvcnn_predictor import SPVCNNPredictor
from ml.models.mapping_adapter import MLToMappingAdapter, GridMap25D


class TestSPVCNNContract(unittest.TestCase):
    """Test suite for SPVCNN predictor contract compliance and 2.5D mapping integration."""

    @classmethod
    def setUpClass(cls):
        cls.predictor = SPVCNNPredictor(device="cpu", voxel_size=0.05)
        cls.mapping_adapter = MLToMappingAdapter(
            bounds_x=(-50.0, 50.0),
            bounds_y=(-50.0, 50.0),
            resolution=0.20,
        )
        cls.real_scan_path = repo_root / "dataset/sequences/00/velodyne/000000.bin"

    def test_01_frozen_contract_structure(self):
        """Test 1: Predictor output matches Amit's frozen contract [xyz, predicted_class, confidence]."""
        pts = np.random.uniform(-10, 10, (200, 4)).astype(np.float32)
        res = self.predictor.predict(pts)

        self.assertIn("xyz", res)
        self.assertIn("predicted_class", res)
        self.assertIn("confidence", res)

        self.assertEqual(res["xyz"].shape, (200, 3))
        self.assertEqual(res["predicted_class"].shape, (200,))
        self.assertEqual(res["confidence"].shape, (200,))

        # Verify exact input XYZ coordinates
        np.testing.assert_array_equal(res["xyz"], pts[:, :3])

    def test_02_contract_bounds_and_classes(self):
        """Test 2: Confidence in [0.0, 1.0] and classes in {0, 1, 2, 3, 255}."""
        pts = np.random.uniform(-20, 20, (500, 4)).astype(np.float32)
        res = self.predictor.predict(pts)

        self.assertTrue(np.all(res["confidence"] >= 0.0) and np.all(res["confidence"] <= 1.0))
        self.assertTrue(set(np.unique(res["predicted_class"])).issubset({0, 1, 2, 3, 255}))

    def test_03_gridmap25d_integration(self):
        """Test 3: SPVCNN predictions successfully feed MLToMappingAdapter and generate GridMap25D."""
        pts = np.random.uniform(-25, 25, (1000, 4)).astype(np.float32)
        res = self.predictor.predict(pts)

        grid_map = self.mapping_adapter.build_25d_grid(res)
        self.assertIsInstance(grid_map, GridMap25D)
        self.assertEqual(grid_map.grid_shape, (500, 500))
        self.assertFalse(np.isnan(grid_map.confidence_layer).any())
        self.assertFalse(np.isinf(grid_map.confidence_layer).any())

    def test_04_real_scan_end_to_end(self):
        """Test 4: Full real LiDAR scan -> Amit foveation -> SPVCNN -> GridMap25D."""
        if not self.real_scan_path.exists():
            self.skipTest("Real scan 000000.bin not found")

        raw_pts = load_point_cloud(self.real_scan_path)
        self.assertEqual(raw_pts.shape[0], 66658)

        # Foveated downsampler
        sampler = FoveatedVoxelSampler(
            near_dist=10.0, near_voxel=0.05,
            mid_dist=40.0, mid_voxel=0.15,
            far_dist=100.0, far_voxel=0.50,
        )
        fov_pts, _, report = sampler.sample(raw_pts)
        self.assertGreater(fov_pts.shape[0], 40000)
        self.assertLess(fov_pts.shape[0], raw_pts.shape[0])

        # SPVCNN Predictor
        res = self.predictor.predict(fov_pts)
        self.assertEqual(res["xyz"].shape[0], fov_pts.shape[0])
        np.testing.assert_array_equal(res["xyz"], fov_pts[:, :3])

        # GridMap25D
        grid = self.mapping_adapter.build_25d_grid(res)
        self.assertIsInstance(grid, GridMap25D)


if __name__ == "__main__":
    unittest.main()
