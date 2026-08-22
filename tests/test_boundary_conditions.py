"""
Test Boundary Conditions & Edge Cases via Python pybind11 C++ Engine:
  - Exact radius boundaries: r = 0, 9.999, 10.0, 10.001, 29.999, 30.0, 30.001, 59.999, 60.0, 60.001, 99.999, 100.0, 100.001
  - Out of range & negative radii
  - NaN, +inf, -inf
  - Empty point clouds
  - Single point cloud
  - Identical duplicate points
  - Negative coordinate indexing
"""

import unittest
import numpy as np
import foveated_grid_cpp
from src.foveated_grid import FoveatedGrid25D


class TestBoundaryConditions(unittest.TestCase):
    def setUp(self):
        self.engine = foveated_grid_cpp.FoveatedGridEngine()

    def test_01_exact_band_boundaries(self):
        """Test 1: Half-open interval behavior [min_range, max_range)."""
        radii_tests = [
            (0.0, "near_field", 0.05),
            (9.999, "near_field", 0.05),
            (10.0, "mid_near_field", 0.10),
            (10.001, "mid_near_field", 0.10),
            (29.999, "mid_near_field", 0.10),
            (30.0, "mid_far_field", 0.25),
            (30.001, "mid_far_field", 0.25),
            (59.999, "mid_far_field", 0.25),
            (60.0, "far_field", 0.50),
            (60.001, "far_field", 0.50),
            (99.999, "far_field", 0.50),
            (100.0, None, None),
            (100.001, None, None),
            (-1.0, None, None),
            (float("nan"), None, None),
            (float("inf"), None, None),
            (float("-inf"), None, None),
        ]

        for r, exp_band, exp_res in radii_tests:
            b = self.engine.resolve_band(r)
            if exp_band is None:
                self.assertIsNone(b, f"Expected None for r={r}, got {b}")
            else:
                self.assertIsNotNone(b, f"Expected band for r={r}")
                self.assertEqual(b.name, exp_band)
                self.assertAlmostEqual(b.voxel_size, exp_res)

    def test_02_negative_coordinate_indexing(self):
        """Test 2: Mathematical floor logic: floor(-0.01 / 0.05) = -1, floor(-0.05 / 0.05) = -1, floor(-0.051 / 0.05) = -2."""
        cases = [
            (0.0, 0.0, 0.05, 0, 0),
            (0.049, 0.049, 0.05, 0, 0),
            (0.05, 0.05, 0.05, 1, 1),
            (-0.001, -0.001, 0.05, -1, -1),
            (-0.05, -0.05, 0.05, -1, -1),
            (-0.051, -0.051, 0.05, -2, -2),
        ]
        for x, y, res, exp_ix, exp_iy in cases:
            ix, iy = foveated_grid_cpp.FoveatedGridEngine.xy_to_cell(x, y, res)
            self.assertEqual((ix, iy), (exp_ix, exp_iy), f"Failed for ({x}, {y}, {res})")

    def test_03_empty_and_single_point(self):
        """Test 3: Empty array and single point."""
        empty_pts = np.empty((0, 4), dtype=np.float32)
        res_empty = self.engine.build_grid_numpy(empty_pts)
        self.assertEqual(res_empty["num_cells"], 0)

        single_pt = np.array([[5.0, 0.0, 1.5, 0.8]], dtype=np.float32)
        res_single = self.engine.build_grid_numpy(single_pt)
        self.assertEqual(res_single["num_cells"], 1)
        self.assertEqual(res_single["bands"][0], "near_field")
        self.assertEqual(res_single["point_count"][0], 1)
        self.assertAlmostEqual(res_single["elevation_mean"][0], 1.5)

    def test_04_duplicate_points(self):
        """Test 4: 100 identical points in one cell."""
        pts = np.tile(np.array([[2.01, 3.02, 1.25, 0.5]], dtype=np.float32), (100, 1))
        res = self.engine.build_grid_numpy(pts)
        self.assertEqual(res["num_cells"], 1)
        self.assertEqual(res["point_count"][0], 100)
        self.assertAlmostEqual(res["elevation_mean"][0], 1.25)
        self.assertAlmostEqual(res["elevation_min"][0], 1.25)
        self.assertAlmostEqual(res["elevation_max"][0], 1.25)


if __name__ == "__main__":
    unittest.main()
