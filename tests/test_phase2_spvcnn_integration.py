"""Integration tests for SPVCNN in Phase 2 pipeline & downstream 2.5D grid mapping."""

import unittest
import numpy as np
import torch

from src.types import SuperClass, PointCloudFrame
from phase2.inference.predictor import Phase2Predictor, SemanticPrediction
from phase2.adapter import MLToMappingAdapter
from src.foveated_grid import GridMap25D


class TestPhase2SPVCNNIntegration(unittest.TestCase):
    def test_01_spvcnn_predictor_execution(self):
        """Tests Phase2Predictor with model_type='spvcnn' on synthetic scan."""
        predictor = Phase2Predictor(model_type="spvcnn", device="cpu")
        self.assertEqual(predictor.model_info["model_type"], "SPVCNN")

        pts = np.array([
            [2.5, 1.0, -1.7, 0.4],
            [5.0, 3.0, 0.5, 0.8],
            [12.0, -4.0, 1.2, 0.3],
            [35.0, 20.0, -0.2, 0.1]
        ], dtype=np.float32)
        frame = PointCloudFrame(points=pts, labels=np.zeros(4, dtype=np.uint32), frame_id="000001")

        prediction = predictor.predict_frame(frame)
        self.assertIsInstance(prediction, SemanticPrediction)
        self.assertTrue(prediction.validate_interface())
        self.assertEqual(prediction.num_points, 4)
        self.assertEqual(prediction.predicted_class.shape, (4,))
        self.assertEqual(prediction.class_probabilities.shape, (4, 4))
        self.assertEqual(prediction.confidence.shape, (4,))

    def test_02_fallback_foveated_pointnet_execution(self):
        """Tests Phase2Predictor fallback to model_type='foveated_pointnet'."""
        predictor = Phase2Predictor(model_type="foveated_pointnet", device="cpu")
        self.assertEqual(predictor.model_info["model_type"], "FoveatedPointSegNet")

        pts = np.random.uniform(-10, 10, size=(20, 4)).astype(np.float32)
        frame = PointCloudFrame(points=pts, labels=np.zeros(20, dtype=np.uint32), frame_id="000002")

        prediction = predictor.predict_frame(frame)
        self.assertIsInstance(prediction, SemanticPrediction)
        self.assertTrue(prediction.validate_interface())
        self.assertEqual(prediction.num_points, 20)

    def test_03_spvcnn_to_gridmap_adapter_pipeline(self):
        """Tests SPVCNN SemanticPrediction seamlessly feeding into MLToMappingAdapter."""
        predictor = Phase2Predictor(model_type="spvcnn", device="cpu")
        adapter = MLToMappingAdapter()

        pts = np.random.uniform(-20, 20, size=(100, 4)).astype(np.float32)
        frame = PointCloudFrame(points=pts, labels=np.zeros(100, dtype=np.uint32), frame_id="000003")

        prediction = predictor.predict_frame(frame)
        grid_map = adapter.prediction_to_grid(prediction)

        self.assertIsInstance(grid_map, GridMap25D)
        self.assertGreater(grid_map.num_occupied_cells, 0)
        self.assertTrue(adapter.validate_spatial_alignment(pts, grid_map))

    def test_04_empty_frame_handling(self):
        """Tests SPVCNN handles empty point cloud frames without crashing."""
        predictor = Phase2Predictor(model_type="spvcnn", device="cpu")
        empty_frame = PointCloudFrame(points=np.empty((0, 4), dtype=np.float32), labels=np.empty(0, dtype=np.uint32))

        prediction = predictor.predict_frame(empty_frame)
        self.assertEqual(prediction.num_points, 0)


if __name__ == "__main__":
    unittest.main()
