"""
Comprehensive Edge-Case Testing Suite for Phase 1 LiDAR Pipeline.
Covers boundary transitions, degenerate inputs, invalid floats, extreme densities, and mixed classes.
"""

import unittest
import numpy as np

from src.types import PointCloudFrame, SuperClass, AggregationPolicy, FoveationBand
from src.validator import PointCloudValidator
from src.range_filter import RangeFilter
from src.foveation import FoveatedVoxelizer
from src.label_mapper import LabelMapper
from src.metrics.elevation_preservation import ElevationPreservationValidator
from src.metrics.obstacle_preservation import ObstaclePreservationValidator


class TestPipelineEdgeCases(unittest.TestCase):
    def setUp(self):
        self.validator = PointCloudValidator(max_allowed_range=100.0)
        self.range_filter = RangeFilter(min_range=0.0, max_range=100.0)
        self.voxelizer = FoveatedVoxelizer(config_path="configs/foveation_default.yaml")
        self.label_mapper = LabelMapper(mapping_config_path="configs/semantickitti_mapping.yaml")
        self.elevation_val = ElevationPreservationValidator(grid_resolution=0.20, max_range=100.0)
        self.obstacle_val = ObstaclePreservationValidator(grid_resolution=0.25, max_range=100.0)

    def test_empty_point_cloud(self):
        """Empty frame with 0 points."""
        empty_pts = np.empty((0, 4), dtype=np.float32)
        empty_lbls = np.empty((0,), dtype=np.uint32)
        frame = PointCloudFrame(points=empty_pts, labels=empty_lbls)

        val = self.validator.validate_frame(frame)
        self.assertTrue(val.is_valid_frame)
        self.assertEqual(val.coordinate_validation.total_points, 0)

        filt_frame, filt_rep = self.range_filter.filter_frame(frame)
        self.assertEqual(filt_frame.num_points, 0)
        self.assertEqual(filt_rep.output_points, 0)

        fov_res = self.voxelizer.voxelize(filt_frame)
        self.assertEqual(fov_res.foveated_points, 0)
        self.assertEqual(fov_res.point_reduction_percentage, 0.0)

        elev_rep = self.elevation_val.evaluate(frame, fov_res.foveated_frame)
        self.assertEqual(elev_rep.total_evaluated_cells, 0)

    def test_single_point(self):
        """Single point within operational range."""
        pts = np.array([[5.0, 3.0, 1.0, 0.5]], dtype=np.float32)
        lbls = np.array([40], dtype=np.uint32)  # road
        frame = PointCloudFrame(points=pts, labels=lbls)

        mapped = self.label_mapper.map_frame(frame)
        filt, rep = self.range_filter.filter_frame(mapped)
        self.assertEqual(filt.num_points, 1)

        fov_res = self.voxelizer.voxelize(filt)
        self.assertEqual(fov_res.foveated_points, 1)
        self.assertEqual(fov_res.foveated_frame.labels[0], SuperClass.DRIVABLE_TERRAIN)

    def test_all_points_outside_100m(self):
        """All points beyond 100m cutoff."""
        pts = np.array([
            [105.0, 0.0, 0.0, 0.5],
            [80.0, 80.0, 0.0, 0.5],   # r = 113.1m
            [0.0, 120.0, 0.0, 0.5]
        ], dtype=np.float32)
        lbls = np.full(3, 50, dtype=np.uint32)
        frame = PointCloudFrame(points=pts, labels=lbls)

        val = self.validator.validate_frame(frame)
        self.assertEqual(val.range_validation.points_beyond_100m, 3)

        filt, rep = self.range_filter.filter_frame(frame)
        self.assertEqual(filt.num_points, 0)
        self.assertEqual(rep.removed_out_of_range_points, 3)

    def test_all_points_inside_10m(self):
        """All points in near field (0-10m)."""
        pts = np.random.uniform(-4, 4, size=(200, 4)).astype(np.float32)
        pts[:, 3] = 0.5
        lbls = np.zeros(200, dtype=np.uint32)
        frame = PointCloudFrame(points=pts, labels=lbls)

        fov_res = self.voxelizer.voxelize(frame)
        self.assertEqual(fov_res.band_stats[0].raw_points, 200)
        self.assertEqual(fov_res.band_stats[1].raw_points, 0)
        self.assertEqual(fov_res.band_stats[2].raw_points, 0)

    def test_exact_boundaries(self):
        """Points lying exactly at r = 0, 10.0, 40.0, 100.0."""
        pts = np.array([
            [0.0, 0.0, 0.0, 0.5],      # r = 0.0
            [10.0, 0.0, 0.0, 0.5],     # r = 10.0
            [0.0, 40.0, 0.0, 0.5],     # r = 40.0
            [100.0, 0.0, 0.0, 0.5],    # r = 100.0
            [100.001, 0.0, 0.0, 0.5]   # r = 100.001 (>100)
        ], dtype=np.float32)
        lbls = np.zeros(5, dtype=np.uint32)
        frame = PointCloudFrame(points=pts, labels=lbls)

        filt, rep = self.range_filter.filter_frame(frame)
        self.assertEqual(filt.num_points, 4)
        self.assertEqual(rep.removed_out_of_range_points, 1)

    def test_boundary_epsilon_transitions(self):
        """Test fine transitions around 10m, 40m, and 100m boundaries."""
        pts = np.array([
            [9.999, 0.0, 0.0, 0.5],   # near band (0-10m)
            [10.000, 0.0, 0.0, 0.5],  # mid band [10-40m)
            [10.001, 0.0, 0.0, 0.5],  # mid band [10-40m)
            [39.999, 0.0, 0.0, 0.5],  # mid band [10-40m)
            [40.000, 0.0, 0.0, 0.5],  # far band [40-100m]
            [40.001, 0.0, 0.0, 0.5],  # far band [40-100m]
            [99.999, 0.0, 0.0, 0.5],  # far band [40-100m]
            [100.000, 0.0, 0.0, 0.5], # far band [40-100m]
            [100.001, 0.0, 0.0, 0.5]  # out of bounds
        ], dtype=np.float32)
        lbls = np.zeros(9, dtype=np.uint32)
        frame = PointCloudFrame(points=pts, labels=lbls)

        filt, rep = self.range_filter.filter_frame(frame)
        self.assertEqual(filt.num_points, 8)

        fov_res = self.voxelizer.voxelize(filt)
        self.assertEqual(fov_res.band_stats[0].raw_points, 1)  # 9.999
        self.assertEqual(fov_res.band_stats[1].raw_points, 3)  # 10.000, 10.001, 39.999
        self.assertEqual(fov_res.band_stats[2].raw_points, 4)  # 40.000, 40.001, 99.999, 100.000

    def test_nan_and_inf_handling(self):
        """NaN and +-Inf coordinates."""
        pts = np.array([
            [5.0, 5.0, 1.0, 0.5],
            [np.nan, 5.0, 1.0, 0.5],
            [5.0, np.inf, 1.0, 0.5],
            [5.0, 5.0, -np.inf, 0.5],
            [5.0, 5.0, 1.0, np.nan]
        ], dtype=np.float32)
        lbls = np.zeros(5, dtype=np.uint32)
        frame = PointCloudFrame(points=pts, labels=lbls)

        val = self.validator.validate_frame(frame)
        self.assertFalse(val.is_valid_frame)
        self.assertEqual(val.coordinate_validation.invalid_point_count, 4)

        filt, rep = self.range_filter.filter_frame(frame)
        self.assertEqual(filt.num_points, 1)
        self.assertEqual(rep.removed_invalid_points, 4)

    def test_negative_coordinates(self):
        """Points in all 4 lateral-longitudinal quadrants with negative elevation."""
        pts = np.array([
            [15.0, 15.0, -1.73, 0.5],   # Quadrant 1 (+X, +Y)
            [15.0, -15.0, -1.73, 0.5],  # Quadrant 4 (+X, -Y)
            [-15.0, 15.0, -1.73, 0.5],  # Quadrant 2 (-X, +Y)
            [-15.0, -15.0, -1.73, 0.5], # Quadrant 3 (-X, -Y)
        ], dtype=np.float32)
        lbls = np.zeros(4, dtype=np.uint32)
        frame = PointCloudFrame(points=pts, labels=lbls)

        fov_res = self.voxelizer.voxelize(frame)
        self.assertEqual(fov_res.foveated_points, 4)
        for i in range(4):
            self.assertAlmostEqual(fov_res.foveated_frame.points[i, 2], -1.73, places=2)

    def test_duplicate_points(self):
        """100 identical points at exact same coordinate."""
        pts = np.tile(np.array([5.0, 5.0, -1.5, 0.8], dtype=np.float32), (100, 1))
        lbls = np.full(100, SuperClass.STATIC_OBSTACLE, dtype=np.uint32)
        frame = PointCloudFrame(points=pts, labels=lbls)

        fov_res = self.voxelizer.voxelize(frame)
        self.assertEqual(fov_res.foveated_points, 1)
        self.assertEqual(fov_res.point_reduction_percentage, 99.0)

    def test_mixed_semantic_classes_in_voxel(self):
        """Voxel containing 4 ground (0), 5 static obstacle (2), 1 dynamic object (3)."""
        # 10 points sharing voxel at (5.02, 5.02, 0.02)
        pts = np.tile(np.array([5.02, 5.02, 0.02, 0.5], dtype=np.float32), (10, 1))
        lbls = np.array([0, 0, 0, 0, 2, 2, 2, 2, 2, 3], dtype=np.uint32)
        frame = PointCloudFrame(points=pts, labels=lbls)

        # Obstacle-preserving policy must prioritize dynamic object (3)
        fov_obs = self.voxelizer.voxelize(frame, policy=AggregationPolicy.OBSTACLE_PRESERVING)
        self.assertEqual(fov_obs.foveated_points, 1)
        self.assertEqual(fov_obs.foveated_frame.labels[0], SuperClass.DYNAMIC_OBJECT)

    def test_all_ignore_labels(self):
        """Points with all raw unlabeled/outlier labels (raw 0/1) mapping to superclass 255."""
        pts = np.random.uniform(0, 10, size=(50, 4)).astype(np.float32)
        lbls = np.zeros(50, dtype=np.uint32)  # raw 0 (unlabeled)
        frame = PointCloudFrame(points=pts, labels=lbls)

        mapped = self.label_mapper.map_frame(frame)
        self.assertTrue(np.all(mapped.labels == SuperClass.IGNORE_LABEL))

        fov_res = self.voxelizer.voxelize(mapped)
        self.assertTrue(np.all(fov_res.foveated_frame.labels == SuperClass.IGNORE_LABEL))

    def test_very_dense_voxel(self):
        """10,000 points packed inside one 0.05m voxel."""
        pts = np.random.uniform([5.01, 5.01, 0.01, 0.5], [5.03, 5.03, 0.03, 0.5], size=(10000, 4)).astype(np.float32)
        lbls = np.zeros(10000, dtype=np.uint32)
        frame = PointCloudFrame(points=pts, labels=lbls)

        fov_res = self.voxelizer.voxelize(frame)
        self.assertEqual(fov_res.foveated_points, 1)
        self.assertEqual(fov_res.point_reduction_percentage, 99.99)

    def test_very_sparse_frame(self):
        """Very sparse scan with 5 points spread across 90 meters."""
        pts = np.array([
            [5.0, 0.0, 0.0, 0.5],
            [25.0, 0.0, 0.0, 0.5],
            [50.0, 0.0, 0.0, 0.5],
            [75.0, 0.0, 0.0, 0.5],
            [90.0, 0.0, 0.0, 0.5]
        ], dtype=np.float32)
        lbls = np.zeros(5, dtype=np.uint32)
        frame = PointCloudFrame(points=pts, labels=lbls)

        fov_res = self.voxelizer.voxelize(frame)
        self.assertEqual(fov_res.foveated_points, 5)
        self.assertEqual(fov_res.point_reduction_percentage, 0.0)


if __name__ == "__main__":
    unittest.main()
