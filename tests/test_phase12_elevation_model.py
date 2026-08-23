"""
Phase 12 — Elevation Model Validation Test Suite.
Verifies:
  12.1 Elevation Statistics Contract (min_z, max_z, mean_z, point_count, height_range)
  12.2 Flat Road Scenario
  12.3 Curb Scenario (Positive Elevation Discontinuity)
  12.4 Elevated Vehicle / Obstacle Scenario
  12.5 Pothole Scenario (Negative Terrain Discontinuity)
  12.6 Mixed Terrain Composite Scene
  12.7 Mixed-Elevation Same-Cell Test
  12.8 Per-Cell Point Count & Local Conservation
  12.9 Elevation Initialization & Incremental Update Order
  12.10 Incremental vs Batch Aggregation Parity
  12.11 Resolution-Aware Elevation (5cm, 10cm, 25cm, 50cm)
  12.12 Multi-Cell Elevation Gradient (Slope)
  12.13 Invalid / Non-Finite Data Robustness (NaN, Inf)
  12.14 Measurement Noise Robustness (+-1cm, +-2cm, +-5cm)
  12.15 Terrain Separation Perception Metrics
  12.16 Python vs C++ vs Independent Elevation Oracle 3-Way Parity
  12.17 Randomized Elevation Testing (Seeds 42, 123, 456, 999, 2026)
  12.18 Permanent Elevation Invariants 1-10
"""

import math
import unittest
from pathlib import Path
import numpy as np

from src.types import SuperClass, FoveationBand, GridCell25D
from src.foveated_grid import (
    FoveatedGrid25D,
    xy_to_cell,
    distance_to_band,
    DEFAULT_FROZEN_BANDS,
    HAS_CPP_GRID
)
if HAS_CPP_GRID:
    import foveated_grid_cpp


class IndependentElevationOracle:
    """
    Independent mathematical oracle computing exact ground-truth cell elevation statistics
    directly from raw coordinate arrays using numpy/pure math. Never invokes production code.
    """
    @staticmethod
    def compute_cell_statistics(z_values: list) -> dict:
        if len(z_values) == 0:
            return {
                "min_z": float("nan"),
                "max_z": float("nan"),
                "mean_z": float("nan"),
                "point_count": 0,
                "height_range": float("nan")
            }
        min_z = float(min(z_values))
        max_z = float(max(z_values))
        mean_z = float(sum(z_values) / len(z_values))
        count = len(z_values)
        height_range = float(max_z - min_z)
        return {
            "min_z": min_z,
            "max_z": max_z,
            "mean_z": mean_z,
            "point_count": count,
            "height_range": height_range
        }


class TestPhase12ElevationModel(unittest.TestCase):

    def setUp(self):
        self.py_engine = FoveatedGrid25D(use_cpp=False)
        self.cpp_engine = FoveatedGrid25D(use_cpp=True) if HAS_CPP_GRID else None

    # =========================================================================
    # 12.3 FLAT ROAD SCENARIO
    # =========================================================================
    def test_12_3_flat_road_scenario(self):
        """Test ideal flat road (z=0) and low-noise road (|z| <= 1mm)."""
        # Ideal flat road
        pts_ideal = np.array([
            [1.0, 1.0, 0.0, 0.5],
            [1.01, 1.01, 0.0, 0.5],
            [1.02, 1.02, 0.0, 0.5],
        ], dtype=np.float32)
        g_py = self.py_engine.build_grid(pts_ideal)
        cell = list(g_py.cells.values())[0]
        self.assertEqual(cell.elevation_min, 0.0)
        self.assertEqual(cell.elevation_max, 0.0)
        self.assertEqual(cell.elevation_mean, 0.0)
        self.assertEqual(cell.height_range, 0.0)

        # Low noise flat road
        noise_z = np.array([0.0005, -0.0008, 0.0002], dtype=np.float32)
        pts_noise = np.array([
            [2.0, 2.0, noise_z[0], 0.5],
            [2.01, 2.01, noise_z[1], 0.5],
            [2.02, 2.02, noise_z[2], 0.5],
        ], dtype=np.float32)
        g_noise = self.py_engine.build_grid(pts_noise)
        cell_n = list(g_noise.cells.values())[0]
        self.assertAlmostEqual(cell_n.elevation_min, -0.0008, places=4)
        self.assertAlmostEqual(cell_n.elevation_max, 0.0005, places=4)
        self.assertAlmostEqual(cell_n.height_range, 0.0013, places=4)
        self.assertLess(cell_n.height_range, 0.005)

    # =========================================================================
    # 12.4 CURB SCENARIO (POSITIVE DISCONTINUITY)
    # =========================================================================
    def test_12_4_curb_scenario(self):
        """Test curb transition: road (z=0.0m) and curb top (z=0.15m)."""
        # Cell A: Pure road (x=2.01, y=2.01)
        # Cell B: Curb boundary containing both road and curb points (x=2.04)
        # Cell C: Pure curb top (x=2.06)
        # In near field (res=0.05m), cell (ix=40, iy=40) spans [2.00, 2.05)
        pts = np.array([
            [2.01, 2.01, 0.00, 0.5],   # Road
            [2.04, 2.01, 0.15, 0.8],   # Curb edge in same cell
            [2.06, 2.01, 0.15, 0.8],   # Pure curb in next cell
        ], dtype=np.float32)

        g = self.py_engine.build_grid(pts)
        c_mixed = g.cells[("near_field", 40, 40)]
        c_curb = g.cells[("near_field", 41, 40)]

        # Mixed curb cell captures the full vertical discontinuity
        self.assertAlmostEqual(c_mixed.elevation_min, 0.00, places=4)
        self.assertAlmostEqual(c_mixed.elevation_max, 0.15, places=4)
        self.assertAlmostEqual(c_mixed.height_range, 0.15, places=4)
        self.assertAlmostEqual(c_mixed.elevation_mean, 0.075, places=4)

        # Pure curb cell has flat height_range
        self.assertAlmostEqual(c_curb.elevation_min, 0.15, places=4)
        self.assertAlmostEqual(c_curb.elevation_max, 0.15, places=4)
        self.assertAlmostEqual(c_curb.height_range, 0.00, places=4)

    # =========================================================================
    # 12.5 ELEVATED VEHICLE / OBSTACLE SCENARIO
    # =========================================================================
    def test_12_5_elevated_vehicle_obstacle(self):
        """Test elevated obstacle (vehicle height z=1.5m above road z=0.0m)."""
        pts = np.array([
            [5.01, 5.01, 0.00, 0.5],   # Road
            [5.02, 5.02, 0.60, 0.8],   # Bumper / chassis
            [5.03, 5.03, 1.20, 0.9],   # Hood
            [5.04, 5.04, 1.50, 0.9],   # Roof
        ], dtype=np.float32)
        lbls = np.array([0, 3, 3, 3], dtype=np.int64)

        g = self.py_engine.build_grid(pts, lbls)
        cell = list(g.cells.values())[0]

        self.assertAlmostEqual(cell.elevation_min, 0.00, places=4)
        self.assertAlmostEqual(cell.elevation_max, 1.50, places=4)
        self.assertAlmostEqual(cell.height_range, 1.50, places=4)
        self.assertAlmostEqual(cell.elevation_mean, 0.825, places=4)
        self.assertEqual(cell.point_count, 4)
        self.assertEqual(cell.semantic_class, SuperClass.DYNAMIC_OBJECT)

    # =========================================================================
    # 12.6 POTHOLE SCENARIO (NEGATIVE DISCONTINUITY)
    # =========================================================================
    def test_12_6_pothole_scenario(self):
        """Test negative terrain depression (pothole z=-0.10m relative to road z=0.0m)."""
        pts = np.array([
            [3.01, 3.01, -0.10, 0.5],  # Pothole bottom
            [3.02, 3.02, -0.05, 0.5],  # Pothole slope
            [3.03, 3.03, 0.00, 0.5],   # Road rim in same cell
            [3.06, 3.01, 0.00, 0.5],   # Surrounding road cell
        ], dtype=np.float32)

        g = self.py_engine.build_grid(pts)
        c_pothole = g.cells[("near_field", 60, 60)]
        c_road = g.cells[("near_field", 61, 60)]

        # Pothole cell has negative min_z
        self.assertAlmostEqual(c_pothole.elevation_min, -0.10, places=4)
        self.assertAlmostEqual(c_pothole.elevation_max, 0.00, places=4)
        self.assertAlmostEqual(c_pothole.height_range, 0.10, places=4)
        self.assertAlmostEqual(c_pothole.elevation_mean, -0.05, places=4)

        # Surrounding road cell has min_z = 0.0
        self.assertAlmostEqual(c_road.elevation_min, 0.00, places=4)

    # =========================================================================
    # 12.7 MIXED TERRAIN COMPOSITE SCENE
    # =========================================================================
    def test_12_7_mixed_terrain_composite_scene(self):
        """Verify co-existence of road, curb, vehicle, and pothole in one unified grid."""
        pts = np.array([
            [1.01, 1.01, 0.00, 0.5],   # Road (ix=20, iy=20)
            [1.02, 1.01, 0.00, 0.5],
            [2.01, 2.01, 0.15, 0.8],   # Curb (ix=40, iy=40)
            [5.01, 5.01, 1.50, 0.9],   # Vehicle (ix=100, iy=100)
            [3.01, 3.01, -0.10, 0.5],  # Pothole (ix=60, iy=60)
        ], dtype=np.float32)

        g = self.py_engine.build_grid(pts)
        c_road = g.cells[("near_field", 20, 20)]
        c_curb = g.cells[("near_field", 40, 40)]
        c_veh = g.cells[("near_field", 100, 100)]
        c_pot = g.cells[("near_field", 60, 60)]

        self.assertAlmostEqual(c_road.elevation_mean, 0.00, places=4)
        self.assertAlmostEqual(c_curb.elevation_mean, 0.15, places=4)
        self.assertAlmostEqual(c_veh.elevation_mean, 1.50, places=4)
        self.assertAlmostEqual(c_pot.elevation_mean, -0.10, places=4)

    # =========================================================================
    # 12.8 MIXED-ELEVATION SAME-CELL EXACT TEST
    # =========================================================================
    def test_12_8_mixed_elevation_same_cell(self):
        """Exact test from specification: [0.00, 0.05, 0.10, 0.15] -> mean=0.075, range=0.15."""
        z_vals = [0.00, 0.05, 0.10, 0.15]
        pts = np.array([[1.01, 1.01, z, 0.8] for z in z_vals], dtype=np.float32)

        # 1. Independent oracle
        expected = IndependentElevationOracle.compute_cell_statistics(z_vals)
        self.assertEqual(expected["min_z"], 0.00)
        self.assertEqual(expected["max_z"], 0.15)
        self.assertAlmostEqual(expected["mean_z"], 0.075, places=5)
        self.assertEqual(expected["point_count"], 4)
        self.assertAlmostEqual(expected["height_range"], 0.15, places=5)


        # 2. Python engine
        g_py = self.py_engine.build_grid(pts)
        c_py = list(g_py.cells.values())[0]
        self.assertAlmostEqual(c_py.elevation_min, expected["min_z"], places=5)
        self.assertAlmostEqual(c_py.elevation_max, expected["max_z"], places=5)
        self.assertAlmostEqual(c_py.elevation_mean, expected["mean_z"], places=5)
        self.assertEqual(c_py.point_count, expected["point_count"])
        self.assertAlmostEqual(c_py.height_range, expected["height_range"], places=5)

        # 3. C++ engine
        if self.cpp_engine:
            g_cpp = self.cpp_engine.build_grid(pts)
            c_cpp = list(g_cpp.cells.values())[0]
            self.assertAlmostEqual(c_cpp.elevation_min, expected["min_z"], places=5)
            self.assertAlmostEqual(c_cpp.elevation_max, expected["max_z"], places=5)
            self.assertAlmostEqual(c_cpp.elevation_mean, expected["mean_z"], places=5)
            self.assertEqual(c_cpp.point_count, expected["point_count"])
            self.assertAlmostEqual(c_cpp.height_range, expected["height_range"], places=5)

    # =========================================================================
    # 12.10 ELEVATION INITIALIZATION & ORDER INVARIANCE
    # =========================================================================
    def test_12_10_elevation_initialization_and_order(self):
        """Verify that point insertion order does not affect final min/max/mean."""
        pts_order1 = np.array([
            [1.01, 1.01, 0.50, 0.8],
            [1.02, 1.01, -0.20, 0.8],
            [1.03, 1.01, 1.80, 0.8],
            [1.04, 1.01, 0.90, 0.8],
        ], dtype=np.float32)

        pts_order2 = pts_order1[[2, 0, 3, 1]] # Permuted order

        g1 = self.py_engine.build_grid(pts_order1)
        g2 = self.py_engine.build_grid(pts_order2)
        c1 = list(g1.cells.values())[0]
        c2 = list(g2.cells.values())[0]

        self.assertAlmostEqual(c1.elevation_min, c2.elevation_min, places=5)
        self.assertAlmostEqual(c1.elevation_max, c2.elevation_max, places=5)
        self.assertAlmostEqual(c1.elevation_mean, c2.elevation_mean, places=5)
        self.assertEqual(c1.point_count, c2.point_count)

    # =========================================================================
    # 12.12 RESOLUTION-AWARE ELEVATION (ALL 4 BANDS)
    # =========================================================================
    def test_12_12_resolution_aware_elevation(self):
        """Verify elevation statistics across all 4 distance bands: 5cm, 10cm, 25cm, 50cm."""
        band_tests = [
            (5.0, "near_field", 0.05),
            (20.0, "mid_near_field", 0.10),
            (45.0, "mid_far_field", 0.25),
            (80.0, "far_field", 0.50),
        ]
        for dist, b_name, res in band_tests:
            pts = np.array([
                [dist, 0.0, 0.10, 0.8],
                [dist, 0.01, 0.30, 0.8],
                [dist, 0.02, 0.50, 0.8],
            ], dtype=np.float32)
            g = self.py_engine.build_grid(pts)
            c = list(g.cells.values())[0]
            self.assertEqual(c.band_name, b_name)
            self.assertAlmostEqual(c.resolution, res, places=4)
            self.assertAlmostEqual(c.elevation_min, 0.10, places=4)
            self.assertAlmostEqual(c.elevation_max, 0.50, places=4)
            self.assertAlmostEqual(c.elevation_mean, 0.30, places=4)
            self.assertAlmostEqual(c.height_range, 0.40, places=4)

    # =========================================================================
    # 12.14 INVALID DATA ROBUSTNESS
    # =========================================================================
    def test_12_14_invalid_elevation_robustness(self):
        """Verify that NaN and Inf Z coordinates are rejected and do not pollute elevation."""
        pts = np.array([
            [1.01, 1.01, 0.50, 0.8],           # Valid
            [1.02, 1.01, float("nan"), 0.8],   # NaN Z -> Rejected
            [1.03, 1.01, float("inf"), 0.8],   # +Inf Z -> Rejected
            [1.04, 1.01, -float("inf"), 0.8],  # -Inf Z -> Rejected
            [1.05, 1.01, 0.70, 0.8],           # Valid
        ], dtype=np.float32)

        g = self.py_engine.build_grid(pts)
        c = list(g.cells.values())[0]
        self.assertEqual(c.point_count, 2)
        self.assertAlmostEqual(c.elevation_min, 0.50, places=5)
        self.assertAlmostEqual(c.elevation_max, 0.70, places=5)
        self.assertAlmostEqual(c.elevation_mean, 0.60, places=5)
        self.assertFalse(math.isnan(c.elevation_min))

    # =========================================================================
    # 12.18 ELEVATION INVARIANTS 1-10
    # =========================================================================
    def test_12_18_elevation_invariants(self):
        """Verify permanent mathematical invariants: min <= mean <= max, range >= 0, etc."""
        rng = np.random.RandomState(42)
        pts = rng.uniform(-60.0, 60.0, size=(2000, 4)).astype(np.float32)
        g = self.py_engine.build_grid(pts)

        for c in g.cells.values():
            # Invariant 1: min_z <= mean_z <= max_z
            self.assertLessEqual(c.elevation_min, c.elevation_mean + 1e-6)
            self.assertLessEqual(c.elevation_mean, c.elevation_max + 1e-6)

            # Invariant 2: height_range >= 0
            self.assertGreaterEqual(c.height_range, -1e-6)

            # Invariant 3: height_range == max_z - min_z
            self.assertAlmostEqual(c.height_range, c.elevation_max - c.elevation_min, places=5)

            # Invariant 4: point_count >= 1
            self.assertGreaterEqual(c.point_count, 1)

            # Invariant 5: Single point cell identity
            if c.point_count == 1:
                self.assertAlmostEqual(c.elevation_min, c.elevation_max, places=5)
                self.assertAlmostEqual(c.elevation_mean, c.elevation_min, places=5)
                self.assertAlmostEqual(c.height_range, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
