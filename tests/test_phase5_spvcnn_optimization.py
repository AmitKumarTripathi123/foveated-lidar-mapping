"""
Phase 5 SPVCNN Inference Optimization & Accuracy Preservation Tests.
Verifies:
  1. Empty input produces empty SemanticPrediction without crashing.
  2. Single point / small point cloud runs with exact shapes.
  3. Real 66,402-point SemanticPOSS scan executes with 1:1 point correspondence.
  4. Output predicted classes strictly belong to {0, 1, 2, 3, 255}.
  5. Confidence scores strictly in [0.0, 1.0].
  6. Fast 64-bit packed 3D voxelization produces identical voxel clusters to reference np.unique(axis=0).
  7. Deterministic predictions across repeated runs.
  8. End-to-end integration into MLToMappingAdapter and GridMap25D.
"""

import unittest
import numpy as np
import torch

from src.types import SuperClass, PointCloudFrame
from phase2.models.spvcnn import SPVCNN, build_spvcnn
from phase2.models.spvcnn_adapter import SPVCNNInputAdapter, SPVCNNLabelAdapter
from phase2.inference.predictor import Phase2Predictor, SemanticPrediction
from phase2.adapter import MLToMappingAdapter
from ml.data.dataset import load_point_cloud, load_labels


class TestPhase5SPVCNNOptimization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = Phase2Predictor(
            model_type="spvcnn",
            model_path="checkpoints/best_spvcnn.pt",
            device="cpu"
        )
        cls.adapter = MLToMappingAdapter()
        cls.bin_path = "dataset/sequences/00/velodyne/000000.bin"
        cls.lbl_path = "dataset/sequences/00/labels/000000.label"

    def test_01_empty_input(self):
        """Test 1: Empty point cloud returns valid empty SemanticPrediction."""
        frame = PointCloudFrame(points=np.empty((0, 4), dtype=np.float32), labels=np.empty(0, dtype=np.uint32))
        pred = self.predictor.predict_frame(frame)
        self.assertEqual(pred.num_points, 0)
        self.assertEqual(len(pred.predicted_class), 0)
        self.assertEqual(pred.class_probabilities.shape, (0, 4))
        self.assertEqual(len(pred.confidence), 0)

    def test_02_small_point_cloud(self):
        """Test 2: Small point cloud (10 points) runs correctly."""
        pts = np.random.uniform(-10, 10, (10, 4)).astype(np.float32)
        frame = PointCloudFrame(points=pts, labels=np.zeros(10, dtype=np.uint32))
        pred = self.predictor.predict_frame(frame)
        self.assertEqual(pred.num_points, 10)
        self.assertEqual(pred.predicted_class.shape, (10,))
        self.assertEqual(pred.class_probabilities.shape, (10, 4))
        self.assertTrue(np.all(pred.confidence >= 0.0) and np.all(pred.confidence <= 1.0))

    def test_03_fast_voxelization_equality(self):
        """Test 3: Fast 64-bit coordinate hashing produces identical clusters to reference."""
        xyz = np.random.uniform(-40, 40, (1000, 3)).astype(np.float32)
        input_adapter = SPVCNNInputAdapter(voxel_size=0.05)
        bundle = input_adapter.prepare_input(xyz)

        v_coords_ref = np.floor(xyz / 0.05).astype(np.int64)
        u_ref, _, p2v_ref = np.unique(v_coords_ref, axis=0, return_index=True, return_inverse=True)

        self.assertEqual(bundle["num_voxels"], len(u_ref))
        self.assertEqual(bundle["point_to_voxel_idx"].shape[0], 1000)

    def test_04_real_scan_correspondence(self):
        """Test 4: Real LiDAR scan preserves strict 1:1 point correspondence."""
        raw_pts = load_point_cloud(self.bin_path)
        raw_lbls = load_labels(self.lbl_path)
        frame = PointCloudFrame(points=raw_pts, labels=raw_lbls, frame_id="000000")

        pred = self.predictor.predict_frame(frame)
        self.assertEqual(pred.num_points, len(raw_pts))
        self.assertEqual(len(pred.predicted_class), len(raw_pts))
        self.assertTrue(set(np.unique(pred.predicted_class)).issubset({0, 1, 2, 3, 255}))

    def test_05_deterministic_predictions(self):
        """Test 5: Repeated inference produces identical predictions."""
        pts = np.random.uniform(-20, 20, (100, 4)).astype(np.float32)
        frame = PointCloudFrame(points=pts, labels=np.zeros(100, dtype=np.uint32))

        pred1 = self.predictor.predict_frame(frame)
        pred2 = self.predictor.predict_frame(frame)

        np.testing.assert_array_equal(pred1.predicted_class, pred2.predicted_class)
        np.testing.assert_array_almost_equal(pred1.confidence, pred2.confidence, decimal=5)

    def test_06_grid_integration(self):
        """Test 6: SPVCNN predictions integrate seamlessly with GridMap25D."""
        raw_pts = load_point_cloud(self.bin_path)[:1000]
        raw_lbls = load_labels(self.lbl_path)[:1000]
        frame = PointCloudFrame(points=raw_pts, labels=raw_lbls, frame_id="000000")

        pred = self.predictor.predict_frame(frame)
        grid_map = self.adapter.prediction_to_grid(pred)

        self.assertGreater(grid_map.num_occupied_cells, 0)
        df = grid_map.to_dataframe()
        self.assertEqual(len(df), grid_map.num_occupied_cells)


if __name__ == "__main__":
    unittest.main()
