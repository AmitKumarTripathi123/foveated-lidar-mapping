"""Unit tests for FoveatedVoxelizer."""
import unittest
import numpy as np

from src.types import PointCloudFrame, AggregationPolicy, SuperClass
from src.foveation import FoveatedVoxelizer


class TestFoveatedVoxelizer(unittest.TestCase):
    def setUp(self):
        self.voxelizer = FoveatedVoxelizer(config_path="configs/foveation_default.yaml")

    def test_multi_band_partitioning(self):
        # Create points in near (r=5), mid (r=25), far (r=60)
        near_pts = np.column_stack([np.full(100, 3.0), np.full(100, 4.0), np.zeros(100), np.ones(100)])
        mid_pts = np.column_stack([np.full(100, 15.0), np.full(100, 20.0), np.zeros(100), np.ones(100)])
        far_pts = np.column_stack([np.full(100, 40.0), np.full(100, 50.0), np.zeros(100), np.ones(100)])
        all_pts = np.vstack([near_pts, mid_pts, far_pts]).astype(np.float32)
        lbls = np.zeros(300, dtype=np.uint32)

        frame = PointCloudFrame(points=all_pts, labels=lbls)
        res = self.voxelizer.voxelize(frame)

        self.assertEqual(res.raw_points, 300)
        self.assertEqual(res.foveated_points, 3)  # 1 point per band voxel
        self.assertGreater(res.point_reduction_percentage, 95.0)

    def test_obstacle_preserving_policy(self):
        # 9 ground points (0) and 1 dynamic object (3) sharing the same voxel
        pts = np.zeros((10, 4), dtype=np.float32)
        pts[:, 0] = 5.02 + np.random.uniform(-0.002, 0.002, 10)
        pts[:, 1] = 2.02 + np.random.uniform(-0.002, 0.002, 10)
        pts[:, 2] = -1.72 + np.random.uniform(-0.002, 0.002, 10)
        pts[:, 3] = 0.5

        lbls = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 3], dtype=np.uint32)
        frame = PointCloudFrame(points=pts, labels=lbls)

        # 1. Majority policy -> would pick 0 (ground)
        res_maj = self.voxelizer.voxelize(frame, policy=AggregationPolicy.MAJORITY)
        self.assertEqual(res_maj.foveated_frame.labels[0], SuperClass.DRIVABLE_TERRAIN)

        # 2. Obstacle-preserving policy -> MUST pick 3 (dynamic object)
        res_obs = self.voxelizer.voxelize(frame, policy=AggregationPolicy.OBSTACLE_PRESERVING)
        self.assertEqual(res_obs.foveated_frame.labels[0], SuperClass.DYNAMIC_OBJECT)

    def test_uniform_baseline(self):
        pts = np.random.uniform(0, 2, size=(1000, 4)).astype(np.float32)
        lbls = np.zeros(1000, dtype=np.uint32)
        frame = PointCloudFrame(points=pts, labels=lbls)

        res_u005 = self.voxelizer.uniform_voxelize(frame, voxel_size=0.05)
        res_u050 = self.voxelizer.uniform_voxelize(frame, voxel_size=0.50)

        self.assertGreater(res_u050.point_reduction_percentage, res_u005.point_reduction_percentage)


if __name__ == "__main__":
    unittest.main()
