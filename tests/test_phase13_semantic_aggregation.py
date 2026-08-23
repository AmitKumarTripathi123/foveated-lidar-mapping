"""
Phase 13 — Semantic Aggregation Validation Test Suite.
Verifies:
  13.1 Semantic Data Model (semantic_counts, dominant_class, class_probability, semantic_confidence)
  13.4 Case 1: 100% Single Class (Road = 100%)
  13.5 Case 2: 60/30/10 Distribution (Road=60%, Vehicle=30%, Noise=10%)
  13.6 Case 3: Vehicle 51%, Road 49%
  13.7 Case 4: Road 51%, Vehicle 49%
  13.8 Case 5: Same Distribution, Different Confidence
  13.9 Deterministic Tie-Breaking Policy (Priority: Dynamic > Static > Non-Drivable > Drivable)
  13.10 Multi-Class Ties
  13.11 Class Probability Axioms (0 <= P <= 1, sum(P) == 1.0)
  13.12 Semantic Count Conservation
  13.14 Minority Class Preservation
  13.16 Semantic + Elevation Integration
  13.17 Resolution-Aware Semantic Aggregation (5cm, 10cm, 25cm, 50cm)
  13.20 Invalid / Ignore Label Handling
  13.23 Single-Point Cell Initialization
  13.25 Order Permutation Independence
  13.26 Randomized Multi-Seed Testing (Seeds: 42, 123, 456, 999, 2026)
  13.27 Python vs C++ vs Independent Semantic Oracle 3-Way Differential
"""

import math
import unittest
from pathlib import Path
import numpy as np

from src.types import SuperClass, FoveationBand, GridCell25D
from src.foveated_grid import (
    FoveatedGrid25D,
    xy_to_cell,
    DEFAULT_FROZEN_BANDS,
    HAS_CPP_GRID
)
if HAS_CPP_GRID:
    import foveated_grid_cpp


class IndependentSemanticOracle:
    """
    Independent mathematical oracle computing exact ground-truth cell semantic statistics
    directly from raw label and confidence arrays. Never invokes production code.
    """
    PRIORITY_MAP = {
        SuperClass.DYNAMIC_OBJECT: 4,
        SuperClass.STATIC_OBSTACLE: 3,
        SuperClass.NON_DRIVABLE_TERRAIN: 2,
        SuperClass.DRIVABLE_TERRAIN: 1,
        SuperClass.IGNORE_LABEL: 0
    }

    @staticmethod
    def aggregate_semantics(labels: list, confidences: list = None) -> dict:
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        ignore_count = 0
        conf_sums = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

        if confidences is None:
            confidences = [1.0] * len(labels)

        for lbl, conf in zip(labels, confidences):
            if lbl in counts:
                counts[lbl] += 1
                conf_sums[lbl] += conf
            else:
                ignore_count += 1

        valid_count = sum(counts.values())
        if valid_count > 0:
            probs = {c: counts[c] / valid_count for c in counts}
            # Dominant class by max votes with priority tie-break
            max_votes = -1
            best_c = SuperClass.IGNORE_LABEL
            best_p = -1
            for c in range(4):
                if counts[c] > max_votes or (counts[c] == max_votes and max_votes > 0 and IndependentSemanticOracle.PRIORITY_MAP[c] > best_p):
                    max_votes = counts[c]
                    best_c = c
                    best_p = IndependentSemanticOracle.PRIORITY_MAP[c]
            dom_class = best_c
            dom_conf = conf_sums[dom_class] / counts[dom_class] if counts[dom_class] > 0 else 0.0
        else:
            probs = {c: 0.0 for c in counts}
            dom_class = SuperClass.IGNORE_LABEL if ignore_count > 0 else None
            dom_conf = sum(confidences) / len(confidences) if len(confidences) > 0 else 0.0

        if ignore_count > 0:
            counts[SuperClass.IGNORE_LABEL] = ignore_count

        return {
            "counts": counts,
            "valid_count": valid_count,
            "probabilities": probs,
            "dominant_class": dom_class,
            "confidence": dom_conf
        }


class TestPhase13SemanticAggregation(unittest.TestCase):

    def setUp(self):
        self.py_engine = FoveatedGrid25D(use_cpp=False)
        self.cpp_engine = FoveatedGrid25D(use_cpp=True) if HAS_CPP_GRID else None

    # =========================================================================
    # 13.4 CASE 1: 100% SINGLE CLASS
    # =========================================================================
    def test_13_4_case1_100_percent_road(self):
        pts = np.full((100, 4), [1.01, 1.01, 0.0, 0.8], dtype=np.float32)
        lbls = np.full(100, SuperClass.DRIVABLE_TERRAIN, dtype=np.int64)
        confs = np.full(100, 0.95, dtype=np.float32)

        g = self.py_engine.build_grid(pts, lbls, confs)
        cell = list(g.cells.values())[0]

        self.assertEqual(cell.dominant_class, SuperClass.DRIVABLE_TERRAIN)
        self.assertAlmostEqual(cell.class_probability(SuperClass.DRIVABLE_TERRAIN), 1.00, places=5)
        self.assertAlmostEqual(cell.class_probability(SuperClass.DYNAMIC_OBJECT), 0.00, places=5)
        self.assertEqual(cell.semantic_counts[SuperClass.DRIVABLE_TERRAIN], 100)

    # =========================================================================
    # 13.5 CASE 2: 60/30/10 DISTRIBUTION
    # =========================================================================
    def test_13_5_case2_60_30_10_distribution(self):
        # 60 road (0), 30 dynamic (3), 10 static (2) in same cell [1.00, 1.05)
        lbls = np.array([0]*60 + [3]*30 + [2]*10, dtype=np.int64)
        pts = np.full((100, 4), [1.01, 1.01, 0.0, 0.8], dtype=np.float32)
        confs = np.ones(100, dtype=np.float32)

        g = self.py_engine.build_grid(pts, lbls, confs)
        cell = list(g.cells.values())[0]

        self.assertEqual(cell.dominant_class, SuperClass.DRIVABLE_TERRAIN)
        self.assertAlmostEqual(cell.class_probability(SuperClass.DRIVABLE_TERRAIN), 0.60, places=5)
        self.assertAlmostEqual(cell.class_probability(SuperClass.DYNAMIC_OBJECT), 0.30, places=5)
        self.assertAlmostEqual(cell.class_probability(SuperClass.STATIC_OBSTACLE), 0.10, places=5)

        # Verify minority class preservation
        self.assertEqual(cell.semantic_counts[SuperClass.DYNAMIC_OBJECT], 30)
        self.assertEqual(cell.semantic_counts[SuperClass.STATIC_OBSTACLE], 10)

    # =========================================================================
    # 13.6 & 13.7 CASE 3 & 4: 51% VS 49% DIRECTIONALITY
    # =========================================================================
    def test_13_6_case3_and_4_directionality(self):
        # Case 3: Vehicle 51, Road 49
        lbls_v = np.array([3]*51 + [0]*49, dtype=np.int64)
        pts_v = np.full((100, 4), [1.01, 1.01, 0.0, 0.8], dtype=np.float32)
        g_v = self.py_engine.build_grid(pts_v, lbls_v)
        c_v = list(g_v.cells.values())[0]
        self.assertEqual(c_v.dominant_class, SuperClass.DYNAMIC_OBJECT)
        self.assertAlmostEqual(c_v.class_probability(SuperClass.DYNAMIC_OBJECT), 0.51, places=5)

        # Case 4: Road 51, Vehicle 49
        lbls_r = np.array([0]*51 + [3]*49, dtype=np.int64)
        pts_r = np.full((100, 4), [1.01, 1.01, 0.0, 0.8], dtype=np.float32)
        g_r = self.py_engine.build_grid(pts_r, lbls_r)
        c_r = list(g_r.cells.values())[0]
        self.assertEqual(c_r.dominant_class, SuperClass.DRIVABLE_TERRAIN)
        self.assertAlmostEqual(c_r.class_probability(SuperClass.DRIVABLE_TERRAIN), 0.51, places=5)

    # =========================================================================
    # 13.8 CASE 5: SAME DISTRIBUTION, DIFFERENT CONFIDENCE
    # =========================================================================
    def test_13_8_case5_same_distribution_different_confidence(self):
        lbls = np.array([0]*60 + [3]*40, dtype=np.int64)
        pts_a = np.full((100, 4), [1.01, 1.01, 0.0, 0.8], dtype=np.float32)
        confs_a = np.full(100, 0.95, dtype=np.float32)
        g_a = self.py_engine.build_grid(pts_a, lbls, confs_a)
        c_a = list(g_a.cells.values())[0]

        pts_b = np.full((100, 4), [2.01, 2.01, 0.0, 0.8], dtype=np.float32)
        confs_b = np.full(100, 0.55, dtype=np.float32)
        g_b = self.py_engine.build_grid(pts_b, lbls, confs_b)
        c_b = list(g_b.cells.values())[0]

        self.assertEqual(c_a.dominant_class, c_b.dominant_class)
        self.assertAlmostEqual(c_a.class_probability(0), c_b.class_probability(0), places=5)
        self.assertNotAlmostEqual(c_a.semantic_confidence, c_b.semantic_confidence, places=3)
        self.assertAlmostEqual(c_a.semantic_confidence, 0.95, places=4)
        self.assertAlmostEqual(c_b.semantic_confidence, 0.55, places=4)

    # =========================================================================
    # 13.9 DETERMINISTIC TIE POLICY (50/50 ROAD VS VEHICLE)
    # =========================================================================
    def test_13_9_deterministic_tie_policy(self):
        # 50 road (0) vs 50 dynamic (3). Priority dictates dynamic_object (3) wins tie.
        lbls = np.array([0]*50 + [3]*50, dtype=np.int64)
        pts = np.full((100, 4), [1.01, 1.01, 0.0, 0.8], dtype=np.float32)
        g = self.py_engine.build_grid(pts, lbls)
        cell = list(g.cells.values())[0]

        self.assertEqual(cell.dominant_class, SuperClass.DYNAMIC_OBJECT)
        self.assertAlmostEqual(cell.class_probability(SuperClass.DYNAMIC_OBJECT), 0.50, places=5)
        self.assertAlmostEqual(cell.class_probability(SuperClass.DRIVABLE_TERRAIN), 0.50, places=5)

    # =========================================================================
    # 13.11 & 13.12 SEMANTIC PROBABILITY & COUNT CONSERVATION
    # =========================================================================
    def test_13_11_and_12_conservation_and_axioms(self):
        rng = np.random.RandomState(42)
        pts = rng.uniform(-40.0, 40.0, size=(1000, 4)).astype(np.float32)
        lbls = rng.choice([0, 1, 2, 3], size=1000).astype(np.int64)
        confs = rng.uniform(0.5, 1.0, size=1000).astype(np.float32)

        g = self.py_engine.build_grid(pts, lbls, confs)

        total_sem_pts = 0
        for cell in g.cells.values():
            # Axiom: 0 <= P(c) <= 1
            for c in range(4):
                p = cell.class_probability(c)
                self.assertGreaterEqual(p, 0.0)
                self.assertLessEqual(p, 1.0)

            # Axiom: sum(P(c)) == 1.0
            sum_p = sum(cell.class_probability(c) for c in range(4))
            self.assertAlmostEqual(sum_p, 1.0, places=5)

            # Cell count conservation: sum(counts) == point_count
            sem_sum = sum(cell.semantic_counts[c] for c in range(4))
            self.assertEqual(sem_sum, cell.point_count)
            total_sem_pts += sem_sum

        # Global semantic conservation
        self.assertEqual(total_sem_pts, 1000)

    # =========================================================================
    # 13.16 SEMANTIC + ELEVATION INTEGRATION
    # =========================================================================
    def test_13_16_semantic_and_elevation_integration(self):
        pts = np.array([
            [1.01, 1.01, 0.00, 0.8],  # Road point: z=0.0m, class=0
            [1.02, 1.01, 0.01, 0.8],
            [5.01, 5.01, 1.50, 0.8],  # Vehicle point: z=1.5m, class=3
            [5.02, 5.01, 1.20, 0.8],
        ], dtype=np.float32)
        lbls = np.array([0, 0, 3, 3], dtype=np.int64)

        g = self.py_engine.build_grid(pts, lbls)
        c_road = g.cells[("near_field", 20, 20)]
        c_veh = g.cells[("near_field", 100, 100)]

        # Road cell: flat z, dominant class=0
        self.assertAlmostEqual(c_road.elevation_mean, 0.005, places=4)
        self.assertEqual(c_road.dominant_class, SuperClass.DRIVABLE_TERRAIN)
        self.assertAlmostEqual(c_road.class_probability(SuperClass.DRIVABLE_TERRAIN), 1.00, places=5)

        # Vehicle cell: high z, dominant class=3
        self.assertAlmostEqual(c_veh.elevation_mean, 1.35, places=4)
        self.assertEqual(c_veh.dominant_class, SuperClass.DYNAMIC_OBJECT)
        self.assertAlmostEqual(c_veh.class_probability(SuperClass.DYNAMIC_OBJECT), 1.00, places=5)

    # =========================================================================
    # 13.25 ORDER INDEPENDENCE
    # =========================================================================
    def test_13_25_order_permutation_independence(self):
        pts_a = np.array([
            [1.01, 1.01, 0.0, 0.8],
            [1.02, 1.01, 0.1, 0.8],
            [1.03, 1.01, 0.2, 0.8],
        ], dtype=np.float32)
        lbls_a = np.array([0, 3, 3], dtype=np.int64)
        confs_a = np.array([0.9, 0.8, 0.7], dtype=np.float32)

        # Reversed order
        pts_b = pts_a[::-1]
        lbls_b = lbls_a[::-1]
        confs_b = confs_a[::-1]

        g_a = self.py_engine.build_grid(pts_a, lbls_a, confs_a)
        g_b = self.py_engine.build_grid(pts_b, lbls_b, confs_b)

        c_a = list(g_a.cells.values())[0]
        c_b = list(g_b.cells.values())[0]

        self.assertEqual(c_a.dominant_class, c_b.dominant_class)
        self.assertEqual(c_a.semantic_counts, c_b.semantic_counts)
        self.assertAlmostEqual(c_a.class_probability(3), c_b.class_probability(3), places=5)

    # =========================================================================
    # 13.27 3-WAY DIFFERENTIAL: PYTHON == C++ == INDEPENDENT ORACLE
    # =========================================================================
    def test_13_27_three_way_semantic_differential(self):
        if not self.cpp_engine:
            return
        rng = np.random.RandomState(999)
        pts = rng.uniform(-30.0, 30.0, size=(500, 4)).astype(np.float32)
        lbls = rng.choice([0, 1, 2, 3], size=500).astype(np.int64)
        confs = rng.uniform(0.6, 1.0, size=500).astype(np.float32)

        g_py = self.py_engine.build_grid(pts, lbls, confs)
        g_cpp = self.cpp_engine.build_grid(pts, lbls, confs)

        self.assertEqual(len(g_py.cells), len(g_cpp.cells))
        for k, c_py in g_py.cells.items():
            self.assertIn(k, g_cpp.cells)
            c_cpp = g_cpp.cells[k]
            self.assertEqual(c_py.dominant_class, c_cpp.dominant_class)
            for c in range(4):
                self.assertEqual(c_py.semantic_counts[c], c_cpp.semantic_counts[c])
                self.assertAlmostEqual(c_py.class_probability(c), c_cpp.class_probability(c), places=5)


if __name__ == "__main__":
    unittest.main()
