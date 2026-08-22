"""
Reproducibility and Determinism Verification Test Suite.
Asserts that running identical frames and configurations produces exact bitwise/numerical parity.
"""

import unittest
import numpy as np

from src.types import PointCloudFrame, AggregationPolicy
from src.validator import PointCloudValidator
from src.range_filter import RangeFilter
from src.foveation import FoveatedVoxelizer
from src.label_mapper import LabelMapper
from src.metrics.elevation_preservation import ElevationPreservationValidator
from src.metrics.obstacle_preservation import ObstaclePreservationValidator


class TestPipelineReproducibility(unittest.TestCase):
    def setUp(self):
        self.validator = PointCloudValidator(max_allowed_range=100.0)
        self.range_filter = RangeFilter(min_range=0.0, max_range=100.0)
        self.voxelizer = FoveatedVoxelizer(config_path="configs/foveation_default.yaml")
        self.label_mapper = LabelMapper(mapping_config_path="configs/semantickitti_mapping.yaml")
        self.elevation_val = ElevationPreservationValidator(grid_resolution=0.20, max_range=100.0)
        self.obstacle_val = ObstaclePreservationValidator(grid_resolution=0.25, max_range=100.0)

    def test_deterministic_foveation(self):
        """Running foveation twice on same frame produces identical outputs."""
        np.random.seed(12345)
        pts = np.random.uniform(-50, 50, size=(10000, 4)).astype(np.float32)
        pts[:, 3] = np.random.uniform(0.1, 0.9, size=10000)
        lbls = np.random.choice([0, 10, 30, 40, 48, 50, 70, 80], size=10000).astype(np.uint32)

        frame = PointCloudFrame(points=pts, labels=lbls)
        mapped = self.label_mapper.map_frame(frame)
        filtered, _ = self.range_filter.filter_frame(mapped)

        # Run 1
        res_1 = self.voxelizer.voxelize(filtered, policy=AggregationPolicy.OBSTACLE_PRESERVING)
        # Run 2
        res_2 = self.voxelizer.voxelize(filtered, policy=AggregationPolicy.OBSTACLE_PRESERVING)

        self.assertEqual(res_1.foveated_points, res_2.foveated_points)
        self.assertEqual(res_1.point_reduction_percentage, res_2.point_reduction_percentage)

        # Assert exact coordinate parity
        np.testing.assert_allclose(res_1.foveated_frame.points, res_2.foveated_frame.points, rtol=1e-6, atol=1e-6)
        # Assert exact label parity
        np.testing.assert_array_equal(res_1.foveated_frame.labels, res_2.foveated_frame.labels)

    def test_deterministic_metrics(self):
        """Preservation metrics are deterministic."""
        np.random.seed(999)
        pts = np.random.uniform(-40, 40, size=(5000, 4)).astype(np.float32)
        lbls = np.random.choice([0, 1, 2, 3, 255], size=5000).astype(np.uint32)

        frame = PointCloudFrame(points=pts, labels=lbls)
        fov_res = self.voxelizer.voxelize(frame)

        elev_1 = self.elevation_val.evaluate(frame, fov_res.foveated_frame)
        elev_2 = self.elevation_val.evaluate(frame, fov_res.foveated_frame)
        self.assertEqual(elev_1.overall_rmse, elev_2.overall_rmse)
        self.assertEqual(elev_1.overall_mae, elev_2.overall_mae)

        obs_1 = self.obstacle_val.evaluate(frame, fov_res.foveated_frame)
        obs_2 = self.obstacle_val.evaluate(frame, fov_res.foveated_frame)
        self.assertEqual(obs_1.obstacle_grid_recall, obs_2.obstacle_grid_recall)
        self.assertEqual(obs_1.obstacle_grid_iou, obs_2.obstacle_grid_iou)


if __name__ == "__main__":
    unittest.main()
