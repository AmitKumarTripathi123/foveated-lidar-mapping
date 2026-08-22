"""Unit tests for Phase2SemanticEvaluator."""
import unittest
import numpy as np

from phase2.metrics.semantic_evaluator import Phase2SemanticEvaluator


class TestPhase2Metrics(unittest.TestCase):
    def setUp(self):
        self.evaluator = Phase2SemanticEvaluator()

    def test_perfect_prediction(self):
        targets = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
        preds = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
        ranges = np.array([5.0, 8.0, 15.0, 25.0, 50.0, 60.0, 70.0, 80.0], dtype=np.float32)

        res = self.evaluator.evaluate(preds, targets, ranges=ranges)
        self.assertEqual(res["overall_accuracy"], 1.0)
        self.assertEqual(res["mIoU"], 1.0)
        self.assertEqual(res["drivable_terrain_IoU"], 1.0)
        self.assertEqual(res["non_drivable_terrain_IoU"], 1.0)
        self.assertEqual(res["static_obstacle_IoU"], 1.0)
        self.assertEqual(res["dynamic_object_IoU"], 1.0)

    def test_ignore_label_exclusion(self):
        targets = np.array([0, 1, 2, 3, 255, 255], dtype=np.int64)
        preds = np.array([0, 1, 2, 3, 0, 1], dtype=np.int64)

        res = self.evaluator.evaluate(preds, targets)
        self.assertEqual(res["overall_accuracy"], 1.0)
        self.assertEqual(res["mIoU"], 1.0)


if __name__ == "__main__":
    unittest.main()
