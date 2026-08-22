"""Unit tests for preservation metrics."""
import unittest
import numpy as np

from src.types import PointCloudFrame, SuperClass
from src.metrics.elevation_preservation import ElevationPreservationValidator
from src.metrics.obstacle_preservation import ObstaclePreservationValidator
from src.metrics.semantic_preservation import SemanticPreservationValidator


class TestPreservationMetrics(unittest.TestCase):
    def test_elevation_metrics(self):
        elev_val = ElevationPreservationValidator(grid_resolution=0.20, max_range=100.0)

        # Create identical raw and foveated frame
        pts = np.array([
            [5.0, 5.0, 1.0, 0.5],
            [20.0, 20.0, 2.0, 0.5],
            [60.0, 60.0, 3.0, 0.5]
        ], dtype=np.float32)
        lbls = np.zeros(3, dtype=np.uint32)

        raw = PointCloudFrame(points=pts, labels=lbls)
        fov = PointCloudFrame(points=pts, labels=lbls)

        rep = elev_val.evaluate(raw, fov)
        self.assertAlmostEqual(rep.overall_rmse, 0.0)
        self.assertAlmostEqual(rep.overall_mae, 0.0)

    def test_obstacle_metrics(self):
        obs_val = ObstaclePreservationValidator(grid_resolution=0.25, max_range=100.0)

        raw_pts = np.array([[5.0, 5.0, 0.0, 0.5], [20.0, 20.0, 0.0, 0.5]], dtype=np.float32)
        raw_lbls = np.array([SuperClass.STATIC_OBSTACLE, SuperClass.DYNAMIC_OBJECT], dtype=np.uint32)
        raw_frame = PointCloudFrame(points=raw_pts, labels=raw_lbls)

        fov_pts = np.array([[5.0, 5.0, 0.0, 0.5], [20.0, 20.0, 0.0, 0.5]], dtype=np.float32)
        fov_lbls = np.array([SuperClass.STATIC_OBSTACLE, SuperClass.DYNAMIC_OBJECT], dtype=np.uint32)
        fov_frame = PointCloudFrame(points=fov_pts, labels=fov_lbls)

        rep = obs_val.evaluate(raw_frame, fov_frame)
        self.assertEqual(rep.obstacle_grid_recall, 100.0)
        self.assertEqual(rep.obstacle_loss_percentage, 0.0)


if __name__ == "__main__":
    unittest.main()
