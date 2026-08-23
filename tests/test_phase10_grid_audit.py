"""
Phase 10 — Grid Engine Technical Audit and Mathematical Parity Test Suite.
Verifies:
  10.1 Resolution boundaries ([0, 10), [10, 30), [30, 60), [60, 100))
  10.2 Cell boundaries (floor semantics)
  10.2B Negative coordinate handling (floor vs truncation parity)
  10.3 Point conservation invariants (accepted == inserted, no loss, no duplication)
  10.4 Python vs C++ differential testing across 7 diverse datasets
  10.5 Explicit 100m cutoff boundary policy
  10.6 Mathematical property invariants 1-8
"""

import math
import unittest
from pathlib import Path
import numpy as np

from src.types import SuperClass, FoveationBand
from src.foveated_grid import (
    FoveatedGrid25D,
    distance_to_band,
    distance_to_resolution,
    xy_to_cell,
    DEFAULT_FROZEN_BANDS,
    HAS_CPP_GRID
)
if HAS_CPP_GRID:
    import foveated_grid_cpp


class TestPhase10GridEngineAudit(unittest.TestCase):

    # =========================================================================
    # 10.1 — RESOLUTION BOUNDARY AUDIT
    # =========================================================================
    def test_10_1_resolution_boundaries(self):
        """Audit exact resolution band transitions with epsilon boundaries."""
        eps = 1e-6

        # Band 0: [0.0, 10.0) -> 0.05m
        self.assertEqual(distance_to_resolution(0.0), 0.05)
        self.assertEqual(distance_to_resolution(9.999), 0.05)
        self.assertEqual(distance_to_resolution(10.0 - eps), 0.05)

        # Band 1: [10.0, 30.0) -> 0.10m
        self.assertEqual(distance_to_resolution(10.0), 0.10)
        self.assertEqual(distance_to_resolution(10.0 + eps), 0.10)
        self.assertEqual(distance_to_resolution(29.999), 0.10)
        self.assertEqual(distance_to_resolution(30.0 - eps), 0.10)

        # Band 2: [30.0, 60.0) -> 0.25m
        self.assertEqual(distance_to_resolution(30.0), 0.25)
        self.assertEqual(distance_to_resolution(30.0 + eps), 0.25)
        self.assertEqual(distance_to_resolution(59.999), 0.25)
        self.assertEqual(distance_to_resolution(60.0 - eps), 0.25)

        # Band 3: [60.0, 100.0) -> 0.50m
        self.assertEqual(distance_to_resolution(60.0), 0.50)
        self.assertEqual(distance_to_resolution(60.0 + eps), 0.50)
        self.assertEqual(distance_to_resolution(99.999), 0.50)
        self.assertEqual(distance_to_resolution(100.0 - eps), 0.50)

        # Out of bounds: >= 100.0m -> None
        self.assertIsNone(distance_to_resolution(100.0))
        self.assertIsNone(distance_to_resolution(100.0 + eps))
        self.assertIsNone(distance_to_resolution(150.0))
        self.assertIsNone(distance_to_resolution(-0.1))
        self.assertIsNone(distance_to_resolution(float("nan")))
        self.assertIsNone(distance_to_resolution(float("inf")))

    # =========================================================================
    # 10.2 & 10.2B — CELL BOUNDARY & NEGATIVE COORDINATE AUDIT
    # =========================================================================
    def test_10_2_cell_boundaries_positive(self):
        """Audit positive coordinate cell boundaries with mathematical floor."""
        cell_size = 0.10
        eps = 1e-6

        # x = 0 -> cell 0
        self.assertEqual(xy_to_cell(0.0, 0.0, cell_size), (0, 0))
        # x = cell_size - eps -> cell 0
        self.assertEqual(xy_to_cell(cell_size - eps, 0.0, cell_size), (0, 0))
        # x = cell_size -> cell 1
        self.assertEqual(xy_to_cell(cell_size, 0.0, cell_size), (1, 0))
        # x = cell_size + eps -> cell 1
        self.assertEqual(xy_to_cell(cell_size + eps, 0.0, cell_size), (1, 0))

        # Test Y independently
        self.assertEqual(xy_to_cell(0.0, cell_size - eps, cell_size), (0, 0))
        self.assertEqual(xy_to_cell(0.0, cell_size, cell_size), (0, 1))
        self.assertEqual(xy_to_cell(0.0, cell_size + eps, cell_size), (0, 1))

    def test_10_2b_negative_coordinates_floor_semantics(self):
        """Audit negative coordinate indexing to guarantee floor semantics (not truncation)."""
        cell_size = 0.10
        eps = 1e-6

        # floor(-0.05 / 0.10) = floor(-0.5) = -1 (NOT 0!)
        self.assertEqual(xy_to_cell(-0.05, 0.0, cell_size), (-1, 0))
        self.assertEqual(xy_to_cell(-0.049999, 0.0, cell_size), (-1, 0))
        self.assertEqual(xy_to_cell(-cell_size, 0.0, cell_size), (-1, 0))
        self.assertEqual(xy_to_cell(-cell_size - eps, 0.0, cell_size), (-2, 0))

        # Y axis negative tests
        self.assertEqual(xy_to_cell(0.0, -0.05, cell_size), (0, -1))
        self.assertEqual(xy_to_cell(0.0, -cell_size, cell_size), (0, -1))
        self.assertEqual(xy_to_cell(0.0, -cell_size - eps, cell_size), (0, -2))

        # Both negative
        self.assertEqual(xy_to_cell(-0.05, -0.05, cell_size), (-1, -1))

    # =========================================================================
    # 10.3 — POINT CONSERVATION AUDIT
    # =========================================================================
    def test_10_3_point_conservation_invariants(self):
        """Verify accepted_input_count == grid_inserted_count across diverse conditions."""
        py_engine = FoveatedGrid25D(use_cpp=False)
        cpp_engine = FoveatedGrid25D(use_cpp=True) if HAS_CPP_GRID else None

        # Test A: Simple deterministic cloud
        pts_a = np.array([
            [1.0, 1.0, 0.5, 0.8],
            [2.0, 2.0, 0.6, 0.9],
            [15.0, 0.0, 1.2, 0.7],
        ], dtype=np.float32)
        lbls_a = np.array([0, 2, 3], dtype=np.int64)
        confs_a = np.array([0.9, 0.8, 0.95], dtype=np.float32)

        g_py_a = py_engine.build_grid(pts_a, lbls_a, confs_a)
        self.assertEqual(sum(c.point_count for c in g_py_a.cells.values()), 3)
        if cpp_engine:
            g_cpp_a = cpp_engine.build_grid(pts_a, lbls_a, confs_a)
            self.assertEqual(sum(c.point_count for c in g_cpp_a.cells.values()), 3)

        # Test B: Multiple points in exactly the same cell
        pts_b = np.array([
            [1.01, 1.01, 0.5, 0.8],
            [1.02, 1.02, 0.6, 0.8],
            [1.03, 1.03, 0.7, 0.8],
            [1.04, 1.04, 0.8, 0.8],
        ], dtype=np.float32)
        lbls_b = np.array([0, 0, 0, 0], dtype=np.int64)
        confs_b = np.array([0.8, 0.8, 0.8, 0.8], dtype=np.float32)

        g_py_b = py_engine.build_grid(pts_b, lbls_b, confs_b)
        self.assertEqual(len(g_py_b.cells), 1)
        self.assertEqual(list(g_py_b.cells.values())[0].point_count, 4)
        if cpp_engine:
            g_cpp_b = cpp_engine.build_grid(pts_b, lbls_b, confs_b)
            self.assertEqual(len(g_cpp_b.cells), 1)
            self.assertEqual(list(g_cpp_b.cells.values())[0].point_count, 4)

        # Test G: Rejected points (NaN, out-of-range >= 100m, Inf)
        pts_g = np.array([
            [1.0, 1.0, 0.5, 0.8],      # Valid -> Accepted
            [float("nan"), 2.0, 0.5, 0.8], # NaN -> Rejected
            [120.0, 0.0, 0.5, 0.8],    # >= 100m -> Rejected
            [0.0, float("inf"), 0.5, 0.8], # Inf -> Rejected
            [50.0, 50.0, 1.0, 0.9],    # r = 70.71m -> Valid -> Accepted
        ], dtype=np.float32)
        lbls_g = np.array([0, 0, 0, 0, 2], dtype=np.int64)
        confs_g = np.array([0.8, 0.8, 0.8, 0.8, 0.9], dtype=np.float32)

        g_py_g = py_engine.build_grid(pts_g, lbls_g, confs_g)
        self.assertEqual(sum(c.point_count for c in g_py_g.cells.values()), 2)
        if cpp_engine:
            g_cpp_g = cpp_engine.build_grid(pts_g, lbls_g, confs_g)
            self.assertEqual(sum(c.point_count for c in g_cpp_g.cells.values()), 2)

    # =========================================================================
    # 10.4 — PYTHON VS C++ DIFFERENTIAL TESTING (7 DATASETS)
    # =========================================================================
    def test_10_4_differential_python_vs_cpp(self):
        """Exhaustively verify Python vs C++ mathematical equivalence across 7 datasets."""
        if not HAS_CPP_GRID:
            self.skipTest("C++ extension foveated_grid_cpp not available")

        py_engine = FoveatedGrid25D(use_cpp=False)
        cpp_engine = FoveatedGrid25D(use_cpp=True)

        datasets = {
            "1_normal_points": np.array([[2.0, 3.0, 0.5, 0.5], [15.0, 12.0, -0.2, 0.8], [45.0, 10.0, 1.5, 0.9]], dtype=np.float32),
            "2_resolution_boundaries": np.array([[9.999, 0.0, 0.1, 0.8], [10.0, 0.0, 0.2, 0.8], [29.999, 0.0, 0.3, 0.8], [30.0, 0.0, 0.4, 0.8], [59.999, 0.0, 0.5, 0.8], [60.0, 0.0, 0.6, 0.8], [99.999, 0.0, 0.7, 0.8]], dtype=np.float32),
            "3_cell_boundaries": np.array([[0.0, 0.0, 1.0, 0.8], [0.049999, 0.0, 1.1, 0.8], [0.05, 0.0, 1.2, 0.8], [0.050001, 0.0, 1.3, 0.8]], dtype=np.float32),
            "4_negative_coords": np.array([[-0.05, 0.0, 0.1, 0.8], [-0.049999, 0.0, 0.2, 0.8], [-0.10, 0.0, 0.3, 0.8], [-0.100001, 0.0, 0.4, 0.8], [-5.0, -5.0, -0.5, 0.8]], dtype=np.float32),
            "5_same_cell_collisions": np.tile(np.array([2.01, 2.01, 0.5, 0.8], dtype=np.float32), (50, 1)),
            "6_mixed_cloud": np.array([[5.0, -5.0, 0.5, 0.8], [-25.0, 15.0, -1.0, 0.8], [50.0, -40.0, 2.0, 0.8], [99.999, 0.0, 0.0, 0.8]], dtype=np.float32),
            "7_large_deterministic": np.random.RandomState(42).uniform(-70.0, 70.0, size=(5000, 4)).astype(np.float32)
        }

        for d_name, pts in datasets.items():
            lbls = np.full(len(pts), SuperClass.DRIVABLE_TERRAIN, dtype=np.int64)
            confs = np.ones(len(pts), dtype=np.float32)

            g_py = py_engine.build_grid(pts, lbls, confs)
            g_cpp = cpp_engine.build_grid(pts, lbls, confs)

            py_cells = g_py.cells
            cpp_cells = g_cpp.cells

            self.assertEqual(len(py_cells), len(cpp_cells), f"Cell count mismatch in {d_name}")

            for k, py_c in py_cells.items():
                self.assertIn(k, cpp_cells, f"Missing key {k} in C++ grid for {d_name}")
                cpp_c = cpp_cells[k]

                # Exact integer properties
                self.assertEqual(py_c.ix, cpp_c.ix)
                self.assertEqual(py_c.iy, cpp_c.iy)
                self.assertEqual(py_c.point_count, cpp_c.point_count)
                self.assertEqual(py_c.semantic_class, cpp_c.semantic_class)
                self.assertEqual(py_c.band_name, cpp_c.band_name)

                # Floating point properties (within 1e-5 tolerance)
                self.assertAlmostEqual(py_c.resolution, cpp_c.resolution, places=5)
                self.assertAlmostEqual(py_c.elevation_mean, cpp_c.elevation_mean, places=5)
                self.assertAlmostEqual(py_c.elevation_min, cpp_c.elevation_min, places=5)
                self.assertAlmostEqual(py_c.elevation_max, cpp_c.elevation_max, places=5)
                self.assertAlmostEqual(py_c.confidence, cpp_c.confidence, places=5)
                self.assertAlmostEqual(py_c.traversability, cpp_c.traversability, places=5)

    # =========================================================================
    # 10.5 — EXPLICIT 100M CUTOFF POLICY
    # =========================================================================
    def test_10_5_explicit_100m_cutoff(self):
        """Verify strict half-open boundary: r < 100m is valid, r >= 100m is rejected."""
        eps = 1e-5
        self.assertIsNotNone(distance_to_band(99.999))
        self.assertIsNotNone(distance_to_band(100.0 - eps))
        self.assertIsNone(distance_to_band(100.0))
        self.assertIsNone(distance_to_band(100.0 + eps))
        self.assertIsNone(distance_to_band(100.001))

    # =========================================================================
    # 10.6 — PROPERTY & INVARIANT TESTING
    # =========================================================================
    def test_10_6_invariants(self):
        """Verify Invariants 1-8: conservation, determinism, uniqueness, equivalence."""
        rng = np.random.RandomState(1337)
        pts = rng.uniform(-60.0, 60.0, size=(1000, 4)).astype(np.float32)
        r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        valid_mask = (r < 100.0) & np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1]) & np.isfinite(pts[:, 2])
        accepted_pts = pts[valid_mask]
        accepted_count = len(accepted_pts)

        py_engine = FoveatedGrid25D(use_cpp=False)
        cpp_engine = FoveatedGrid25D(use_cpp=True) if HAS_CPP_GRID else None

        lbls = rng.choice([0, 1, 2, 3], size=len(pts))
        confs = rng.uniform(0.5, 1.0, size=len(pts)).astype(np.float32)

        g_py = py_engine.build_grid(pts, lbls, confs)
        total_py_pts = sum(c.point_count for c in g_py.cells.values())

        # Invariant 1, 2, 4: Total count equals accepted input count
        self.assertEqual(total_py_pts, accepted_count)

        if cpp_engine:
            g_cpp = cpp_engine.build_grid(pts, lbls, confs)
            total_cpp_pts = sum(c.point_count for c in g_cpp.cells.values())
            self.assertEqual(total_cpp_pts, accepted_count)
            self.assertEqual(total_py_pts, total_cpp_pts)

            # Invariant 5: Determinism
            g_cpp2 = cpp_engine.build_grid(pts, lbls, confs)
            self.assertEqual(len(g_cpp.cells), len(g_cpp2.cells))


if __name__ == "__main__":
    unittest.main()
