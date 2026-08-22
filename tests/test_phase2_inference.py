"""Unit tests for Phase2Predictor and SemanticPrediction interface."""
import unittest
import numpy as np

from phase2.inference.predictor import Phase2Predictor, SemanticPrediction
from src.types import PointCloudFrame


class TestPhase2Inference(unittest.TestCase):
    def setUp(self):
        self.predictor = Phase2Predictor(model_path=None, device="cpu")

    def test_semantic_prediction_contract(self):
        pts = np.random.uniform(-30, 30, size=(300, 4)).astype(np.float32)
        lbls = np.zeros(300, dtype=np.uint32)
        frame = PointCloudFrame(points=pts, labels=lbls, frame_id="000001")

        pred = self.predictor.predict_frame(frame)
        self.assertIsInstance(pred, SemanticPrediction)
        self.assertEqual(pred.num_points, 300)
        self.assertEqual(pred.frame_id, "000001")
        self.assertTrue(pred.validate_interface())

    def test_deterministic_inference(self):
        pts = np.random.uniform(-20, 20, size=(100, 4)).astype(np.float32)
        frame = PointCloudFrame(points=pts, labels=np.zeros(100, dtype=np.uint32))

        pred1 = self.predictor.predict_frame(frame)
        pred2 = self.predictor.predict_frame(frame)

        np.testing.assert_allclose(pred1.class_probabilities, pred2.class_probabilities, atol=1e-6)
        np.testing.assert_array_equal(pred1.predicted_class, pred2.predicted_class)
        np.testing.assert_allclose(pred1.confidence, pred2.confidence, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
