"""
Phase 19.2 Native Grid Rasterizer Unit & Invariant Test Suite:
- Validates native module imports and initialization
- Tests boundary conditions: cell boundaries, 100m sensor limit, empty input, single point
- Tests numerical equivalence: elevation mean, min, max, confidence, traversability, and semantic voting
- Tests bitwise cell set equivalence against reference Python implementation
- Tests NaN/Inf coordinate rejection and memory safety
"""

import json
import unittest
from pathlib import Path
import numpy as np
import torch

from src.core.native_grid import NativeGridMapRasterizer, rasterize_grid_native_cpu
from src.core.foveated_grid import HierarchicalFoveatedGridEngine
from benchmarks.phase19_2.correctness_audit import compare_grid_maps
from ml.pipeline.production_pipeline import verify_file_sha256


class TestPhase19_2NativeGrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.ckpt_path = cls.repo_root / "experiments/phase12_full_semanticposs_spvcnn/best_checkpoint.pt"
        cls.engine = HierarchicalFoveatedGridEngine()
        cls.rasterizer = NativeGridMapRasterizer()

    def test_01_checkpoint_immutability(self):
        """Test 1: Checkpoint SHA256 matches production baseline."""
        expected_sha = "b15c6dfb2f20d1dce4febc47be67f9d50b86a0af72f1895176c6a6ee58bca142"
        self.assertTrue(self.ckpt_path.is_file())
        self.assertTrue(verify_file_sha256(self.ckpt_path, expected_sha))

    def test_02_native_module_import(self):
        """Test 2: Native grid engine imports and initializes successfully."""
        self.assertIsNotNone(self.rasterizer)
        self.assertEqual(self.rasterizer.grid_shape, (500, 500))
        self.assertEqual(self.rasterizer.resolution, 0.20)

    def test_03_empty_input(self):
        """Test 3: Empty point clouds produce all-NaN elevation and 0-occupancy."""
        empty_pts = np.zeros((0, 3), dtype=np.float32)
        empty_c = np.zeros(0, dtype=np.int64)
        empty_conf = np.zeros(0, dtype=np.float32)

        grid = self.rasterizer.rasterize(empty_pts, empty_c, empty_conf)
        self.assertEqual(np.count_nonzero(grid.point_count_layer > 0), 0)
        self.assertTrue(np.all(np.isnan(grid.elevation_mean)))
        self.assertTrue(np.all(grid.semantic_layer == 255))

    def test_04_single_point(self):
        """Test 4: Single point produces exactly one occupied cell with correct properties."""
        pts = np.array([[10.1, 20.1, 1.5]], dtype=np.float32)
        c = np.array([0], dtype=np.int64)
        conf = np.array([0.95], dtype=np.float32)

        grid = self.rasterizer.rasterize(pts, c, conf)
        self.assertEqual(np.count_nonzero(grid.point_count_layer > 0), 1)

        # ix = floor((10.1 - (-50)) / 0.2) = 300, iy = floor((20.1 - (-50)) / 0.2) = 350
        self.assertEqual(grid.point_count_layer[350, 300], 1)
        self.assertAlmostEqual(float(grid.elevation_mean[350, 300]), 1.5, places=5)
        self.assertEqual(grid.semantic_layer[350, 300], 0)
        self.assertAlmostEqual(float(grid.traversability_layer[350, 300]), 1.0, places=5)

    def test_05_cell_boundary_and_100m_limit(self):
        """Test 5: Boundary points and 100m limits are handled safely."""
        # Point inside grid
        pts_in = np.array([[49.9, 49.9, 0.0]], dtype=np.float32)
        grid_in = self.rasterizer.rasterize(pts_in, np.array([0]), np.array([1.0]))
        self.assertEqual(np.count_nonzero(grid_in.point_count_layer > 0), 1)

        # Point outside bounds (e.g. 50.1m)
        pts_out = np.array([[50.1, 50.1, 0.0]], dtype=np.float32)
        grid_out = self.rasterizer.rasterize(pts_out, np.array([0]), np.array([1.0]))
        self.assertEqual(np.count_nonzero(grid_out.point_count_layer > 0), 0)

    def test_06_multiple_semantic_classes_majority_voting(self):
        """Test 6: Multiple semantic classes in one cell follow majority voting rule."""
        # 3 points in cell (0, 0): two Drivable (0), one Static (2)
        pts = np.array([[0.05, 0.05, 1.0], [0.05, 0.05, 1.2], [0.05, 0.05, 1.4]], dtype=np.float32)
        c = np.array([0, 0, 2], dtype=np.int64)
        conf = np.array([0.9, 0.8, 0.95], dtype=np.float32)

        grid = self.rasterizer.rasterize(pts, c, conf)
        # Center cell is (250, 250)
        self.assertEqual(grid.semantic_layer[250, 250], 0)
        self.assertEqual(grid.point_count_layer[250, 250], 3)
        self.assertAlmostEqual(float(grid.elevation_mean[250, 250]), 1.2, places=5)

    def test_07_elevation_min_max_mean_aggregation(self):
        """Test 7: Elevation min, max, and mean calculate exact values."""
        pts = np.array([[0.05, 0.05, 1.0], [0.05, 0.05, 2.0], [0.05, 0.05, 3.0]], dtype=np.float32)
        c = np.array([0, 0, 0], dtype=np.int64)
        conf = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        grid = self.rasterizer.rasterize(pts, c, conf)
        self.assertAlmostEqual(float(grid.elevation_min[250, 250]), 1.0, places=5)
        self.assertAlmostEqual(float(grid.elevation_max[250, 250]), 3.0, places=5)
        self.assertAlmostEqual(float(grid.elevation_mean[250, 250]), 2.0, places=5)

    def test_08_python_cpp_equivalence(self):
        """Test 8: Native rasterizer matches Python reference across 50,000 randomized points."""
        np.random.seed(42)
        xyz = np.random.uniform(-45.0, 45.0, (50000, 3)).astype(np.float32)
        classes = np.random.randint(0, 4, 50000).astype(np.int64)
        conf = np.random.uniform(0.5, 1.0, 50000).astype(np.float32)

        grid_ref = self.engine.build_25d_grid_reference_python(xyz, classes, conf)
        grid_nat = self.engine.build_25d_grid(xyz, classes, conf, use_native=True)

        cmp = compare_grid_maps(grid_ref, grid_nat)
        self.assertTrue(cmp["passed"])
        self.assertTrue(cmp["cell_set_match"])
        self.assertTrue(cmp["point_count_match"])
        self.assertTrue(cmp["elevation_mean_match"])
        self.assertTrue(cmp["semantic_layer_match"])
        self.assertTrue(cmp["traversability_layer_match"])

    def test_09_nan_inf_rejection(self):
        """Test 9: NaN and Inf coordinates do not crash the engine."""
        pts = np.array([[np.nan, 0.0, 1.0], [np.inf, 2.0, 3.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        c = np.array([0, 1, 2], dtype=np.int64)
        conf = np.array([0.9, 0.9, 0.9], dtype=np.float32)

        grid = self.rasterizer.rasterize(pts, c, conf)
        self.assertEqual(np.count_nonzero(grid.point_count_layer > 0), 1)

    def test_10_reports_existence_and_speedup(self):
        """Test 10: Validates Phase 19.2 report payloads and measured speedup >= 2.0x."""
        summary_p = self.repo_root / "reports/phase19_2/phase19_2_summary.json"
        self.assertTrue(summary_p.is_file(), "phase19_2_summary.json missing!")

        with open(summary_p, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["phase"], "19.2")
        self.assertEqual(data["status"], "ACCELERATION_COMPLETE")
        self.assertGreaterEqual(data["isolated_grid_benchmark"]["speedup_native_cpu"], 2.0)
        self.assertLess(data["end_to_end_comparison"]["phase19_2_accelerated"]["mean_ms"], 75.0)


if __name__ == "__main__":
    unittest.main()
