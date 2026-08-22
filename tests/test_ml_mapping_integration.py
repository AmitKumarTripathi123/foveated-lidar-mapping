"""Phase 6 ML -> 2.5D Mapping Adapter and Contract Integration Tests.

Covers:
  1. ML output shape
  2. ML output dtype
  3. Class range strictly in {0, 1, 2, 3}
  4. Confidence range strictly in [0.0, 1.0]
  5. Point ordering preservation (input_xyz == output_xyz)
  6. Mapping adapter input acceptance
  7. Mapping adapter output 2.5D grid dimensions and layers
  8. Coordinate spatial preservation
  9. Real-frame smoke integration (Raw -> Foveated -> PointNet++ -> Mapping Adapter)
  10. Invalid prediction length mismatch rejection
  11. Missing required field rejection
  12. NaN / Inf coordinate & confidence rejection
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import torch

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.dataset import load_point_cloud, load_labels
from ml.data.preprocessing import filter_invalid_points
from ml.data.amit_adapter import FoveatedVoxelSampler
from ml.data.foveated_dataset import normalize_point_count
from ml.models.pointnet2 import build_model
from ml.models.predictor import PointNet2Predictor
from ml.models.mapping_adapter import PredictionBatch, GridMap25D, MLToMappingAdapter


class TestMLMappingIntegration(unittest.TestCase):
    """Test suite for ML -> Mapping interface and 2.5D grid adapter."""

    @classmethod
    def setUpClass(cls):
        """Set up model and adapter fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
        cls.bin_file = repo_root / "dataset/sequences/00/velodyne/000000.bin"
        cls.lbl_file = repo_root / "dataset/sequences/00/labels/000000.label"
        cls.model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        cls.model.eval()
        cls.predictor = PointNet2Predictor(model=cls.model, device="cpu")
        cls.adapter = MLToMappingAdapter(
            bounds_x=(-50.0, 50.0), bounds_y=(-50.0, 50.0), resolution=0.50
        )

    # 1. ML output shape
    def test_01_ml_output_shape(self):
        """Test 1: Predictor output dictionary contains arrays with exact point dimension N."""
        dummy_pts = np.random.randn(256, 4).astype(np.float32)
        pred = self.predictor.predict(dummy_pts)
        self.assertEqual(pred["xyz"].shape, (256, 3))
        self.assertEqual(pred["predicted_class"].shape, (256,))
        self.assertEqual(pred["confidence"].shape, (256,))

    # 2. ML output dtype
    def test_02_ml_output_dtype(self):
        """Test 2: Predictor returns float32 for xyz/confidence and int64 for classes."""
        dummy_pts = np.random.randn(128, 4).astype(np.float32)
        pred = self.predictor.predict(dummy_pts)
        self.assertEqual(pred["xyz"].dtype, np.float32)
        self.assertEqual(pred["predicted_class"].dtype, np.int64)
        self.assertEqual(pred["confidence"].dtype, np.float32)

    # 3. Class range
    def test_03_class_range(self):
        """Test 3: Predicted class IDs strictly belong to {0, 1, 2, 3}."""
        dummy_pts = np.random.randn(256, 4).astype(np.float32)
        pred = self.predictor.predict(dummy_pts)
        unique_classes = set(np.unique(pred["predicted_class"]))
        self.assertTrue(unique_classes.issubset({0, 1, 2, 3}))

    # 4. Confidence range
    def test_04_confidence_range(self):
        """Test 4: Confidence values strictly lie in [0.0, 1.0]."""
        dummy_pts = np.random.randn(256, 4).astype(np.float32)
        pred = self.predictor.predict(dummy_pts)
        self.assertTrue((pred["confidence"] >= 0.0).all())
        self.assertTrue((pred["confidence"] <= 1.0).all())

    # 5. Point ordering
    def test_05_point_ordering(self):
        """Test 5: Predictor output XYZ strictly preserves input XYZ coordinates and order."""
        dummy_pts = np.random.uniform(-40, 40, size=(512, 4)).astype(np.float32)
        pred = self.predictor.predict(dummy_pts)
        np.testing.assert_array_equal(pred["xyz"], dummy_pts[:, :3])

    # 6. Mapping adapter input validation
    def test_06_mapping_adapter_input_acceptance(self):
        """Test 6: Mapping adapter validates and accepts valid PredictionBatch."""
        dummy_pts = np.random.randn(100, 4).astype(np.float32)
        pred = self.predictor.predict(dummy_pts)
        batch = self.adapter.validate_prediction(pred)
        self.assertIsInstance(batch, PredictionBatch)
        self.assertEqual(batch.xyz.shape[0], 100)

    # 7. Mapping adapter output grid shape & layers
    def test_07_mapping_adapter_output_layers(self):
        """Test 7: 2.5D grid has correct shape (200x200 for 100m at 0.5m) and all required layers."""
        dummy_pts = np.random.uniform(-45, 45, size=(500, 4)).astype(np.float32)
        pred = self.predictor.predict(dummy_pts)
        grid = self.adapter.build_25d_grid(pred)

        expected_shape = (200, 200)
        self.assertEqual(grid.grid_shape, expected_shape)
        self.assertEqual(grid.elevation_min.shape, expected_shape)
        self.assertEqual(grid.elevation_max.shape, expected_shape)
        self.assertEqual(grid.elevation_mean.shape, expected_shape)
        self.assertEqual(grid.semantic_layer.shape, expected_shape)
        self.assertEqual(grid.confidence_layer.shape, expected_shape)
        self.assertEqual(grid.traversability_layer.shape, expected_shape)
        self.assertEqual(grid.point_count_layer.shape, expected_shape)

    # 8. Coordinate preservation
    def test_08_coordinate_spatial_preservation(self):
        """Test 8: Known 3D points project into expected 2.5D grid indices."""
        adapter = MLToMappingAdapter(bounds_x=(0.0, 10.0), bounds_y=(0.0, 10.0), resolution=1.0)
        pts = np.array([[2.5, 3.5, 1.0]], dtype=np.float32)  # Should map to row 3, col 2
        classes = np.array([0], dtype=np.int64)
        confs = np.array([0.95], dtype=np.float32)

        batch = PredictionBatch(xyz=pts, predicted_class=classes, confidence=confs)
        grid = adapter.build_25d_grid(batch)

        self.assertEqual(grid.point_count_layer[3, 2], 1)
        self.assertAlmostEqual(grid.elevation_mean[3, 2], 1.0, places=4)
        self.assertEqual(grid.semantic_layer[3, 2], 0)
        self.assertAlmostEqual(grid.traversability_layer[3, 2], 1.0, places=4)

    # 9. Real-frame smoke integration
    def test_09_real_frame_smoke_integration(self):
        """Test 9: Full chain from real raw scan to 2.5D elevation & semantic grid."""
        raw_pts = load_point_cloud(self.bin_file)
        raw_lbls = load_labels(self.lbl_file)

        v_pts, v_lbls, _ = filter_invalid_points(raw_pts, raw_lbls)
        sampler = FoveatedVoxelSampler()
        fov_pts, _, rep = sampler.sample(v_pts, v_lbls)

        norm_pts, _ = normalize_point_count(fov_pts, None, target_num_points=512, seed=42)
        pred = self.predictor.predict(norm_pts)
        grid = self.adapter.build_25d_grid(pred)

        self.assertGreater(grid.point_count_layer.sum(), 0)
        self.assertIsInstance(grid, GridMap25D)

    # 10. Invalid prediction length mismatch rejection
    def test_10_length_mismatch_rejection(self):
        """Test 10: Reject predictions where classes/confidence length != points length."""
        bad_pred = {
            "xyz": np.random.randn(10, 3).astype(np.float32),
            "predicted_class": np.array([0, 1, 2], dtype=np.int64),  # Length 3 != 10
            "confidence": np.random.rand(10).astype(np.float32),
        }
        with self.assertRaises(ValueError):
            self.adapter.validate_prediction(bad_pred)

    # 11. Missing required field rejection
    def test_11_missing_field_rejection(self):
        """Test 11: Reject dictionary missing required contract keys."""
        bad_pred = {
            "xyz": np.random.randn(10, 3).astype(np.float32),
            "predicted_class": np.zeros(10, dtype=np.int64),
            # Missing 'confidence'
        }
        with self.assertRaises(ValueError):
            self.adapter.validate_prediction(bad_pred)

    # 12. NaN / Inf rejection
    def test_12_nan_inf_rejection(self):
        """Test 12: Reject predictions containing NaN or Inf in coordinates or confidence."""
        nan_pts = np.array([[np.nan, 0.0, 0.0]], dtype=np.float32)
        bad_pred_nan_pts = {
            "xyz": nan_pts,
            "predicted_class": np.array([0], dtype=np.int64),
            "confidence": np.array([0.5], dtype=np.float32),
        }
        with self.assertRaises(ValueError):
            self.adapter.validate_prediction(bad_pred_nan_pts)

        nan_conf = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        bad_pred_nan_conf = {
            "xyz": nan_conf,
            "predicted_class": np.array([0], dtype=np.int64),
            "confidence": np.array([np.nan], dtype=np.float32),
        }
        with self.assertRaises(ValueError):
            self.adapter.validate_prediction(bad_pred_nan_conf)


if __name__ == "__main__":
    unittest.main()
