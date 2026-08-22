"""
Unit and regression tests for Phase-2 Foveated 2.5D Grid Correctness and Spatial Alignment.
Tests:
  1. Boundary transitions (9.99/10.00/10.01, 29.99/30.00/30.01, 59.99/60.00/60.01, 99.99/100.00/100.01m).
  2. Negative coordinate indexing (mathematical floor vs truncation).
  3. Same-cell 100-point aggregation (count, min/max/mean elevation, semantic priority).
  4. Neighboring cell isolation.
  5. Deterministic obstacle-preserving semantic priority.
  6. Empty input and unobserved cell semantics (UNKNOWN state).
  7. Mathematical spatial alignment invariants (ix*s <= x < (ix+1)*s).
  8. End-to-end MLToMappingAdapter integration.
"""

import unittest
import math
import numpy as np

from src.types import SuperClass, CellState, GridCell25D, PointCloudFrame
from src.foveated_grid import (
    distance_to_resolution,
    distance_to_band,
    xy_to_cell,
    cell_to_bounds,
    point_to_cell,
    FoveatedGrid25D,
    GridMap25D,
    DEFAULT_FROZEN_BANDS
)
from phase2.inference.predictor import SemanticPrediction
from phase2.adapter import MLToMappingAdapter


class TestFoveatedGridCorrectness(unittest.TestCase):
    def test_boundary_epsilons(self):
        """Validates exact half-open resolution transitions across all 4 distance bands."""
        # Near Band [0, 10) -> 0.05m
        self.assertAlmostEqual(distance_to_resolution(0.00), 0.05)
        self.assertAlmostEqual(distance_to_resolution(5.00), 0.05)
        self.assertAlmostEqual(distance_to_resolution(9.99), 0.05)

        # Mid-Near Band [10, 30) -> 0.10m
        self.assertAlmostEqual(distance_to_resolution(10.00), 0.10)
        self.assertAlmostEqual(distance_to_resolution(10.01), 0.10)
        self.assertAlmostEqual(distance_to_resolution(29.99), 0.10)

        # Mid-Far Band [30, 60) -> 0.25m
        self.assertAlmostEqual(distance_to_resolution(30.00), 0.25)
        self.assertAlmostEqual(distance_to_resolution(30.01), 0.25)
        self.assertAlmostEqual(distance_to_resolution(59.99), 0.25)

        # Far Band [60, 100) -> 0.50m
        self.assertAlmostEqual(distance_to_resolution(60.00), 0.50)
        self.assertAlmostEqual(distance_to_resolution(60.01), 0.50)
        self.assertAlmostEqual(distance_to_resolution(99.99), 0.50)

        # Out of Range >= 100.0m
        self.assertIsNone(distance_to_resolution(100.00))
        self.assertIsNone(distance_to_resolution(100.01))
        self.assertIsNone(distance_to_resolution(150.0))

        # Invalid and edge cases
        self.assertIsNone(distance_to_resolution(-0.01))
        self.assertIsNone(distance_to_resolution(-10.0))
        self.assertIsNone(distance_to_resolution(float("nan")))
        self.assertIsNone(distance_to_resolution(float("inf")))
        self.assertIsNone(distance_to_resolution(float("-inf")))

    def test_negative_xy_coordinates_floor(self):
        """Verifies mathematical floor indexing on negative coordinates (no truncation toward zero)."""
        res = 0.05
        self.assertEqual(xy_to_cell(0.01, 0.01, res), (0, 0))
        self.assertEqual(xy_to_cell(0.00, 0.00, res), (0, 0))
        self.assertEqual(xy_to_cell(-0.01, -0.01, res), (-1, -1))
        self.assertEqual(xy_to_cell(-0.05, -0.05, res), (-1, -1))
        self.assertEqual(xy_to_cell(-0.05001, -0.05001, res), (-2, -2))
        self.assertEqual(xy_to_cell(-1.01, -1.01, res), (-21, -21))

    def test_same_cell_100_points_aggregation(self):
        """Verifies 100 points inside one spatial cell aggregate into exactly 1 cell with correct metrics."""
        # 100 points inside x in [2.01, 2.04], y in [2.01, 2.04] (all fall in ix=40, iy=40 @ 0.05m)
        n_pts = 100
        x = np.random.uniform(2.01, 2.04, n_pts)
        y = np.random.uniform(2.01, 2.04, n_pts)
        z = np.linspace(1.0, 3.0, n_pts)  # min=1.0, max=3.0, mean=2.0
        i = np.full(n_pts, 0.5, dtype=np.float32)
        pts = np.column_stack([x, y, z, i]).astype(np.float32)
        lbls = np.full(n_pts, SuperClass.DRIVABLE_TERRAIN, dtype=np.int64)
        confs = np.full(n_pts, 0.95, dtype=np.float32)

        grid_builder = FoveatedGrid25D()
        grid_map = grid_builder.build_grid(pts, lbls, confs)

        self.assertEqual(grid_map.num_occupied_cells, 1)
        cell = grid_map.get_cell("near_field", 40, 40)
        self.assertEqual(cell.point_count, 100)
        self.assertAlmostEqual(cell.elevation_min, 1.0, places=4)
        self.assertAlmostEqual(cell.elevation_max, 3.0, places=4)
        self.assertAlmostEqual(cell.elevation_mean, 2.0, places=4)
        self.assertAlmostEqual(cell.confidence, 0.95, places=4)
        self.assertEqual(cell.semantic_class, SuperClass.DRIVABLE_TERRAIN)
        self.assertEqual(cell.state, CellState.OCCUPIED)

    def test_neighboring_cells_isolation(self):
        """Verifies points in adjacent cells remain isolated and do not merge."""
        # Point 1 at (0.04, 0.04) -> cell (0, 0) @ 0.05m
        # Point 2 at (0.06, 0.04) -> cell (1, 0) @ 0.05m
        pts = np.array([
            [0.04, 0.04, 0.5, 0.5],
            [0.06, 0.04, 1.5, 0.5]
        ], dtype=np.float32)
        lbls = np.array([SuperClass.DRIVABLE_TERRAIN, SuperClass.STATIC_OBSTACLE], dtype=np.int64)

        grid_builder = FoveatedGrid25D()
        grid_map = grid_builder.build_grid(pts, lbls)

        self.assertEqual(grid_map.num_occupied_cells, 2)
        cell_0 = grid_map.get_cell("near_field", 0, 0)
        cell_1 = grid_map.get_cell("near_field", 1, 0)

        self.assertEqual(cell_0.point_count, 1)
        self.assertEqual(cell_0.semantic_class, SuperClass.DRIVABLE_TERRAIN)
        self.assertAlmostEqual(cell_0.elevation_mean, 0.5)

        self.assertEqual(cell_1.point_count, 1)
        self.assertEqual(cell_1.semantic_class, SuperClass.STATIC_OBSTACLE)
        self.assertAlmostEqual(cell_1.elevation_mean, 1.5)

    def test_semantic_priority_aggregation(self):
        """
        Verifies deterministic obstacle-preserving priority rule:
        dynamic (3) > static (2) > non-drivable (1) > drivable (0) > ignore (255).
        """
        grid_builder = FoveatedGrid25D()

        # Case A: 5 Road (0) + 1 Static Obstacle (2) -> Static Obstacle (2)
        x_a = np.full(6, 1.02, dtype=np.float32)
        y_a = np.full(6, 1.02, dtype=np.float32)
        z_a = np.full(6, -1.7, dtype=np.float32)
        i_a = np.full(6, 0.3, dtype=np.float32)
        pts_a = np.column_stack([x_a, y_a, z_a, i_a])
        lbls_a = np.array([0, 0, 0, 0, 0, 2], dtype=np.int64)
        map_a = grid_builder.build_grid(pts_a, lbls_a)
        cell_a = map_a.get_cell("near_field", 20, 20)
        self.assertEqual(cell_a.semantic_class, SuperClass.STATIC_OBSTACLE)

        # Case B: 5 Road (0) + 1 Dynamic Object (3) -> Dynamic Object (3)
        lbls_b = np.array([0, 0, 0, 0, 0, 3], dtype=np.int64)
        map_b = grid_builder.build_grid(pts_a, lbls_b)
        cell_b = map_b.get_cell("near_field", 20, 20)
        self.assertEqual(cell_b.semantic_class, SuperClass.DYNAMIC_OBJECT)

        # Case C: 5 Static (2) + 1 Dynamic (3) -> Dynamic Object (3)
        lbls_c = np.array([2, 2, 2, 2, 2, 3], dtype=np.int64)
        map_c = grid_builder.build_grid(pts_a, lbls_c)
        cell_c = map_c.get_cell("near_field", 20, 20)
        self.assertEqual(cell_c.semantic_class, SuperClass.DYNAMIC_OBJECT)

        # Case D: 5 Road (0) + 5 Ignore (255) -> Road (0)
        lbls_d = np.array([0, 0, 0, 0, 0, 255], dtype=np.int64)
        map_d = grid_builder.build_grid(pts_a, lbls_d)
        cell_d = map_d.get_cell("near_field", 20, 20)
        self.assertEqual(cell_d.semantic_class, SuperClass.DRIVABLE_TERRAIN)

        # Case E: Only Ignore (255) -> Ignore (255)
        lbls_e = np.full(6, 255, dtype=np.int64)
        map_e = grid_builder.build_grid(pts_a, lbls_e)
        cell_e = map_e.get_cell("near_field", 20, 20)
        self.assertEqual(cell_e.semantic_class, SuperClass.IGNORE_LABEL)

    def test_empty_input_and_unobserved_cell_semantics(self):
        """Verifies empty frames produce 0 occupied cells and unobserved cells return UNKNOWN."""
        grid_builder = FoveatedGrid25D()
        empty_pts = np.empty((0, 4), dtype=np.float32)
        grid_map = grid_builder.build_grid(empty_pts)

        self.assertEqual(grid_map.num_cells, 0)
        self.assertEqual(grid_map.num_occupied_cells, 0)

        # Query an unobserved cell
        unobserved = grid_map.get_cell("near_field", 100, 100)
        self.assertEqual(unobserved.state, CellState.UNKNOWN)
        self.assertEqual(unobserved.point_count, 0)
        self.assertTrue(math.isnan(unobserved.elevation_mean))
        self.assertEqual(unobserved.semantic_class, SuperClass.IGNORE_LABEL)

    def test_cell_alignment_spatial_invariants(self):
        """
        Mathematical proof test:
        For every point p=(x,y,z), verifies that ix*s <= x < (ix+1)*s and iy*s <= y < (iy+1)*s.
        """
        np.random.seed(42)
        # Generate 1000 points spanning all 4 distance bands and 4 quadrants
        angles = np.random.uniform(-np.pi, np.pi, 1000)
        radii = np.random.uniform(0.1, 99.5, 1000)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        z = np.random.uniform(-2.0, 5.0, 1000)
        i = np.random.uniform(0.1, 0.9, 1000)
        pts = np.column_stack([x, y, z, i]).astype(np.float32)
        lbls = np.random.choice([0, 1, 2, 3, 255], size=1000)

        adapter = MLToMappingAdapter()
        grid_builder = FoveatedGrid25D()
        grid_map = grid_builder.build_grid(pts, lbls)

        # Validate spatial invariant on every single point
        is_aligned = adapter.validate_spatial_alignment(pts, grid_map)
        self.assertTrue(is_aligned, "Spatial invariant violated: point fell outside cell bounds!")

    def test_distance_resolution_cell_end_to_end(self):
        """End-to-end verification of Point -> Distance -> Resolution -> 2D Cell -> Cell Bounds."""
        test_points = [
            # (x, y, z, expected_band, expected_res, expected_ix, expected_iy)
            (5.00, 5.00, 1.2, "near_field", 0.05, 100, 100),       # r = 7.07m
            (15.00, 10.00, -0.5, "mid_near_field", 0.10, 150, 100), # r = 18.03m
            (35.00, -20.00, 2.0, "mid_far_field", 0.25, 140, -80), # r = 40.31m
            (-50.00, 50.00, 0.0, "far_field", 0.50, -100, 100),    # r = 70.71m
        ]

        for x, y, z, exp_band, exp_res, exp_ix, exp_iy in test_points:
            cell = point_to_cell(x, y, z, semantic_class=SuperClass.DRIVABLE_TERRAIN)
            self.assertIsNotNone(cell)
            self.assertEqual(cell.band_name, exp_band)
            self.assertAlmostEqual(cell.resolution, exp_res)
            self.assertEqual(cell.ix, exp_ix)
            self.assertEqual(cell.iy, exp_iy)
            self.assertTrue(cell.contains_point(x, y))

    def test_ml_to_mapping_adapter(self):
        """Tests MLToMappingAdapter seamlessly ingesting SemanticPrediction into GridMap25D."""
        n_pts = 200
        pts = np.random.uniform(-30, 30, size=(n_pts, 4)).astype(np.float32)
        preds = np.random.choice([0, 1, 2, 3], size=n_pts).astype(np.int64)
        probs = np.random.uniform(0.0, 1.0, size=(n_pts, 4)).astype(np.float32)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        conf = np.max(probs, axis=1).astype(np.float32)

        pred_obj = SemanticPrediction(
            points=pts,
            predicted_class=preds,
            class_probabilities=probs,
            confidence=conf,
            frame_id="000042",
            timestamp=123.456
        )

        adapter = MLToMappingAdapter()
        grid_map = adapter.prediction_to_grid(pred_obj)

        self.assertIsInstance(grid_map, GridMap25D)
        self.assertEqual(grid_map.frame_id, "000042")
        self.assertAlmostEqual(grid_map.timestamp, 123.456)
        self.assertGreater(grid_map.num_occupied_cells, 0)
        self.assertTrue(adapter.validate_spatial_alignment(pts, grid_map))


if __name__ == "__main__":
    unittest.main()
