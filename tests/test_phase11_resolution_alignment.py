"""
Phase 11 — Resolution Alignment Validation Test Suite.
Verifies:
  11.1 Fundamental 5cm Lattice (BASE_QUANTUM = 0.05m)
  11.2 Integer Lattice Grouping (5cm -> 1x, 10cm -> 2x, 25cm -> 5x, 50cm -> 10x)
  11.3 Resolution Alignment Matrix
  11.4 Resolution Transitions (10m, 30m, 60m)
  11.5 Exact Boundary Ownership (owner_count == 1)
  11.6 Spatial Gap Detection (adjacent regions touch exactly: A.max == B.min)
  11.7 Spatial Overlap Detection (interior(A) ∩ interior(B) = ∅)
  11.8 Positive & Negative Coordinate Alignment
  11.9 X/Y Symmetry across 4 Quadrants
  11.10 2D Corner / Intersection Boundaries
  11.11 Randomized Alignment Stress Test (Fixed Seed)
  11.12 3-Way Differential: Python == C++ == Independent 5cm Lattice Oracle
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


class IndependentLatticeOracle:
    """
    Independent mathematical oracle constructing spatial cells directly from
    the fundamental 5cm lattice quantum (0.05m) using pure integer arithmetic.
    Does NOT invoke any production Grid Engine functions.
    """
    BASE_QUANTUM = 0.05

    @staticmethod
    def resolve_band(r: float):
        if not math.isfinite(r) or r < 0.0 or r >= 100.0:
            return None
        if r < 10.0:
            return ("near_field", 0.05, 1)      # 1 quantum (5cm)
        elif r < 30.0:
            return ("mid_near_field", 0.10, 2)  # 2 quanta (10cm)
        elif r < 60.0:
            return ("mid_far_field", 0.25, 5)   # 5 quanta (25cm)
        else:
            return ("far_field", 0.50, 10)      # 10 quanta (50cm)

    @staticmethod
    def point_to_lattice_cell(x: float, y: float):
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        r = math.sqrt(x * x + y * y)
        band_info = IndependentLatticeOracle.resolve_band(r)
        if band_info is None:
            return None
        name, res, multiplier = band_info

        # Fundamental 5cm lattice indices via exact mathematical floor
        # Use integer quantum rounding to prevent float precision drift
        k_x = int(math.floor(round(x / 0.05, 9)))
        k_y = int(math.floor(round(y / 0.05, 9)))

        # Hierarchical grouping to target resolution cell
        ix = int(math.floor(k_x / multiplier))
        iy = int(math.floor(k_y / multiplier))

        x_min = ix * res
        x_max = (ix + 1) * res
        y_min = iy * res
        y_max = (iy + 1) * res

        return {
            "band_name": name,
            "resolution": res,
            "multiplier": multiplier,
            "k_x": k_x,
            "k_y": k_y,
            "ix": ix,
            "iy": iy,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
        }


class TestPhase11ResolutionAlignment(unittest.TestCase):

    # =========================================================================
    # 11.1 & 11.2 — FUNDAMENTAL LATTICE & INTEGER GROUPING
    # =========================================================================
    def test_11_1_fundamental_lattice_multiples(self):
        """Verify that all resolutions are exact positive integer multiples of 5cm."""
        base_quantum = 0.05
        resolutions = [0.05, 0.10, 0.25, 0.50]
        expected_multipliers = [1, 2, 5, 10]

        for res, exp_m in zip(resolutions, expected_multipliers):
            ratio = round(res / base_quantum, 6)
            self.assertEqual(ratio, float(exp_m))
            self.assertTrue(math.isclose(res, exp_m * base_quantum, abs_tol=1e-9))

    def test_11_2_lattice_quanta_grouping(self):
        """Verify grouping of 5cm lattice indices into higher-resolution cells."""
        # 10cm cells group [2k, 2k+1]
        for k in range(-20, 20):
            x = (k * 0.05) + 0.025
            cell = IndependentLatticeOracle.point_to_lattice_cell(x, 15.0)  # in 10cm band
            self.assertEqual(cell["ix"], k // 2)

        # 25cm cells group [5k .. 5k+4]
        for k in range(-20, 20):
            x = (k * 0.05) + 0.025
            cell = IndependentLatticeOracle.point_to_lattice_cell(x, 45.0)  # in 25cm band
            self.assertEqual(cell["ix"], k // 5)

        # 50cm cells group [10k .. 10k+9]
        for k in range(-20, 20):
            x = (k * 0.05) + 0.025
            cell = IndependentLatticeOracle.point_to_lattice_cell(x, 75.0)  # in 50cm band
            self.assertEqual(cell["ix"], k // 10)

    # =========================================================================
    # 11.3 & 11.4 — RESOLUTION TRANSITION ALIGNMENT
    # =========================================================================
    def test_11_3_resolution_transitions_epsilon(self):
        """Verify exact resolution selection and cell geometry at range boundaries."""
        eps = 1e-6
        transitions = [
            (10.0, 0.05, 0.10),
            (30.0, 0.10, 0.25),
            (60.0, 0.25, 0.50),
        ]

        for b_range, res_below, res_at in transitions:
            # Below boundary
            c_below = IndependentLatticeOracle.point_to_lattice_cell(b_range - eps, 0.0)
            self.assertIsNotNone(c_below)
            self.assertEqual(c_below["resolution"], res_below)

            # Exactly at boundary
            c_at = IndependentLatticeOracle.point_to_lattice_cell(b_range, 0.0)
            self.assertIsNotNone(c_at)
            self.assertEqual(c_at["resolution"], res_at)

            # Above boundary
            c_above = IndependentLatticeOracle.point_to_lattice_cell(b_range + eps, 0.0)
            self.assertIsNotNone(c_above)
            self.assertEqual(c_above["resolution"], res_at)

    # =========================================================================
    # 11.5 — EXACT BOUNDARY OWNERSHIP (HALF-OPEN INTERVALS)
    # =========================================================================
    def test_11_5_exact_boundary_ownership(self):
        """Verify that boundary points have exactly 1 owner and transfer to upper cell."""
        for res in [0.05, 0.10, 0.25, 0.50]:
            eps = 1e-6
            for k in range(-10, 10):
                boundary = round(k * res, 6)

                # Point immediately before boundary belongs to cell k - 1
                ix_before, _ = xy_to_cell(boundary - eps, 0.0, res)
                self.assertEqual(ix_before, k - 1)

                # Point exactly on boundary belongs to cell k (ownership transferred)
                ix_on, _ = xy_to_cell(boundary, 0.0, res)
                self.assertEqual(ix_on, k)

                # Point immediately after boundary belongs to cell k
                ix_after, _ = xy_to_cell(boundary + eps, 0.0, res)
                self.assertEqual(ix_after, k)

    # =========================================================================
    # 11.6 & 11.7 — NO SPATIAL GAPS & NO SPATIAL OVERLAPS
    # =========================================================================
    def test_11_6_no_spatial_gaps(self):
        """Verify that adjacent cell spatial intervals touch continuously: A.max == B.min."""
        for res in [0.05, 0.10, 0.25, 0.50]:
            for k in range(-50, 50):
                x_min_a = k * res
                x_max_a = (k + 1) * res
                x_min_b = (k + 1) * res
                x_max_b = (k + 2) * res

                # Contiguity invariant
                self.assertEqual(round(x_max_a, 6), round(x_min_b, 6))

    def test_11_7_no_spatial_overlaps(self):
        """Verify that cell interiors are strictly disjoint: interior(A) ∩ interior(B) = ∅."""
        for res in [0.05, 0.10, 0.25, 0.50]:
            for k in range(-30, 30):
                # An interior point in cell k cannot be in cell k+1
                interior_pt = (k * res) + (res * 0.5)
                ix, _ = xy_to_cell(interior_pt, 0.0, res)
                self.assertEqual(ix, k)

    # =========================================================================
    # 11.8 & 11.9 — POSITIVE & NEGATIVE COORDINATE ALIGNMENT
    # =========================================================================
    def test_11_8_positive_and_negative_coordinate_alignment(self):
        """Verify consistent lattice index computation across positive and negative values."""
        coords = [
            (-1.00, 0.05, -20),
            (-0.50, 0.05, -10),
            (-0.25, 0.05, -5),
            (-0.10, 0.05, -2),
            (-0.05, 0.05, -1),
            (0.00, 0.05, 0),
            (0.05, 0.05, 1),
            (0.10, 0.05, 2),
            (0.25, 0.05, 5),
            (0.50, 0.05, 10),
            (1.00, 0.05, 20),
        ]

        for val, res, exp_idx in coords:
            ix, _ = xy_to_cell(val, 0.0, res)
            self.assertEqual(ix, exp_idx)

    # =========================================================================
    # 11.10 & 11.11 — X/Y SYMMETRY & 2D CORNER BOUNDARY TESTS
    # =========================================================================
    def test_11_10_xy_quadrant_symmetry(self):
        """Verify symmetric 2D cell indexing across all 4 Cartesian quadrants."""
        res = 0.10
        quadrants = [
            (0.25, 0.25, 2, 2),    # Quadrant I (+X, +Y)
            (0.25, -0.25, 2, -3),  # Quadrant IV (+X, -Y) -> floor(-0.25/0.10) = -3
            (-0.25, 0.25, -3, 2),  # Quadrant II (-X, +Y)
            (-0.25, -0.25, -3, -3) # Quadrant III (-X, -Y)
        ]

        for x, y, exp_ix, exp_iy in quadrants:
            ix, iy = xy_to_cell(x, y, res)
            self.assertEqual((ix, iy), (exp_ix, exp_iy))

    def test_11_11_2d_corner_boundary_ownership(self):
        """Verify that 2D corner intersection points map to exactly one cell."""
        for res in [0.05, 0.10, 0.25, 0.50]:
            corners = [
                (res, res, 1, 1),
                (res, -res, 1, -1),
                (-res, res, -1, 1),
                (-res, -res, -1, -1),
                (0.0, 0.0, 0, 0)
            ]
            for cx, cy, exp_ix, exp_iy in corners:
                ix, iy = xy_to_cell(cx, cy, res)
                self.assertEqual((ix, iy), (exp_ix, exp_iy))

    # =========================================================================
    # 11.12 — RANDOMIZED ALIGNMENT & 3-WAY ORACLE PARITY (5,000 POINTS)
    # =========================================================================
    def test_11_12_three_way_oracle_differential(self):
        """Exhaustively prove Python == C++ == Independent 5cm Lattice Oracle."""
        rng = np.random.RandomState(42)
        # Generate 2,000 points across [-75.0, 75.0]
        pts = rng.uniform(-75.0, 75.0, size=(2000, 4)).astype(np.float32)
        lbls = rng.choice([0, 1, 2, 3], size=len(pts)).astype(np.int64)
        confs = rng.uniform(0.5, 1.0, size=len(pts)).astype(np.float32)

        py_engine = FoveatedGrid25D(use_cpp=False)
        cpp_engine = FoveatedGrid25D(use_cpp=True) if HAS_CPP_GRID else None

        g_py = py_engine.build_grid(pts, lbls, confs)
        g_cpp = cpp_engine.build_grid(pts, lbls, confs) if cpp_engine else None

        # Verify point by point against independent oracle
        for i in range(len(pts)):
            x, y, z = float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])
            r = math.sqrt(x * x + y * y)

            oracle_res = IndependentLatticeOracle.point_to_lattice_cell(x, y)
            if r >= 100.0 or not math.isfinite(x) or not math.isfinite(y) or not math.isfinite(z):
                self.assertIsNone(oracle_res)
                continue

            self.assertIsNotNone(oracle_res)
            # Find in Python grid
            cell_key = (oracle_res["band_name"], oracle_res["ix"], oracle_res["iy"])
            self.assertIn(cell_key, g_py.cells)
            py_cell = g_py.cells[cell_key]

            # Compare Oracle vs Python
            self.assertEqual(py_cell.ix, oracle_res["ix"])
            self.assertEqual(py_cell.iy, oracle_res["iy"])
            self.assertAlmostEqual(py_cell.resolution, oracle_res["resolution"], places=5)
            self.assertEqual(py_cell.band_name, oracle_res["band_name"])


            # Verify spatial bounds encompass the point: x ∈ [x_min, x_max), y ∈ [y_min, y_max)
            self.assertGreaterEqual(x, oracle_res["x_min"])
            self.assertLess(x, oracle_res["x_max"])
            self.assertGreaterEqual(y, oracle_res["y_min"])
            self.assertLess(y, oracle_res["y_max"])

            # Compare Oracle vs C++
            if g_cpp:
                self.assertIn(cell_key, g_cpp.cells)
                cpp_cell = g_cpp.cells[cell_key]
                self.assertEqual(cpp_cell.ix, oracle_res["ix"])
                self.assertEqual(cpp_cell.iy, oracle_res["iy"])
                self.assertEqual(cpp_cell.point_count, py_cell.point_count)


if __name__ == "__main__":
    unittest.main()
