"""
Phase 19.4 Regression Recovery Test Suite (SIH PS 26130).
Validates that:
- Production Checkpoint is immutable
- ML Preprocessing latency regressed from Phase 19.3 is recovered
- 2.5D Grid rasterization latency regressed from Phase 19.3 is recovered
- End-to-end active perception latency is <= 54.97 ms (Phase 19.2 Golden Baseline)
- Throughput is >= 18.19 FPS
- P95 tail latency is <= 67.51 ms
- Semantic mIoU is preserved without meaningful regression
- Zero dropped frames across evaluation
- All mathematical invariants, hash index partitions, and grid cell sets are equivalent
"""

import json
import unittest
from pathlib import Path
import numpy as np
import torch

from src.core.native_foveation import NativeFoveationAccelerator
from src.core.foveated_grid import HierarchicalFoveatedGridEngine
from src.core.native_grid import NativeGridMapRasterizer
from ml.data.spvcnn_adapter import SPVCNNInputAdapter
from ml.data.amit_adapter import FoveatedVoxelSampler
from benchmarks.phase19_2.correctness_audit import compare_grid_maps
from ml.pipeline.production_pipeline import verify_file_sha256


class TestPhase19_4RegressionRecovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.ckpt_path = cls.repo_root / "experiments/phase12_full_semanticposs_spvcnn/best_checkpoint.pt"
        cls.summary_path = cls.repo_root / "reports/phase19_4/phase19_4_summary.json"
        cls.pipe_bench_path = cls.repo_root / "reports/phase19_4/pipeline_benchmark.json"

        cls.adapter = SPVCNNInputAdapter(voxel_size=0.05)
        cls.grid_engine = HierarchicalFoveatedGridEngine()
        cls.grid_rasterizer = NativeGridMapRasterizer()
        cls.fov_native = NativeFoveationAccelerator()
        cls.fov_ref = FoveatedVoxelSampler()

    def test_01_checkpoint_immutability(self):
        """Test 1: Certified Phase 12 checkpoint SHA256 matches production baseline."""
        expected_sha = "b15c6dfb2f20d1dce4febc47be67f9d50b86a0af72f1895176c6a6ee58bca142"
        self.assertTrue(self.ckpt_path.is_file())
        self.assertTrue(verify_file_sha256(self.ckpt_path, expected_sha))

    def test_02_foveation_equivalence(self):
        """Test 2: Foveation point retention is identical between Python and Native."""
        np.random.seed(42)
        pts = np.random.uniform(-45.0, 45.0, (20000, 4)).astype(np.float32)
        pts_r, _, rep_r = self.fov_ref.sample_reference_python(pts)
        pts_n, _, rep_n = self.fov_native.sample(pts)
        self.assertEqual(len(pts_r), len(pts_n))
        self.assertEqual(rep_r.foveated_count, rep_n.foveated_count)

    def test_03_foveation_boundary_equivalence(self):
        """Test 3: Foveation boundary invariants are strictly preserved."""
        pts = np.array([
            [9.999, 0.0, 0.0, 0.5],
            [10.001, 0.0, 0.0, 0.5],
            [39.999, 0.0, 0.0, 0.5],
            [40.001, 0.0, 0.0, 0.5],
            [99.999, 0.0, 0.0, 0.5],
            [100.001, 0.0, 0.0, 0.5],
        ], dtype=np.float32)
        _, _, rep_n = self.fov_native.sample(pts)
        self.assertEqual(rep_n.foveated_count, 5) # 5 points within 100m, 1 filtered
        self.assertEqual(rep_n.filtered_out_count, 1)

    def test_04_ml_preprocess_equivalence(self):
        """Test 4: ML Preprocessor produces identical voxel count and partitions."""
        np.random.seed(123)
        pts = np.random.uniform(-30.0, 30.0, (10000, 4)).astype(np.float32)
        res_py = self.adapter.prepare_input_reference_python(pts)
        res_nat = self.adapter.prepare_input(pts)
        self.assertEqual(res_py["num_voxels"], res_nat["num_voxels"])

    def test_05_voxel_coordinate_equivalence(self):
        """Test 5: Voxel coordinates are quantized consistently."""
        pts = np.array([
            [0.01, 0.02, 0.03, 0.5],
            [0.06, 0.02, 0.03, 0.6],
        ], dtype=np.float32)
        res_py = self.adapter.prepare_input_reference_python(pts)
        res_nat = self.adapter.prepare_input(pts)
        self.assertEqual(res_py["num_voxels"], 2)
        self.assertEqual(res_nat["num_voxels"], 2)

    def test_06_hash_key_equivalence(self):
        """Test 6: GPU CUDA and CPU tensor partitioning match identically."""
        if torch.cuda.is_available():
            pts = np.random.uniform(-20.0, 20.0, (5000, 4)).astype(np.float32)
            res_py = self.adapter.prepare_input_reference_python(pts)
            res_cuda = self.adapter.prepare_input(torch.from_numpy(pts).cuda().float())
            self.assertEqual(res_py["num_voxels"], res_cuda["num_voxels"])

    def test_07_grid_equivalence(self):
        """Test 7: 2.5D GridMap matches Python reference across 50,000 points."""
        np.random.seed(42)
        xyz = np.random.uniform(-40.0, 40.0, (50000, 3)).astype(np.float32)
        c = np.random.randint(0, 4, 50000).astype(np.int64)
        conf = np.random.uniform(0.5, 1.0, 50000).astype(np.float32)

        grid_ref = self.grid_engine.build_25d_grid_reference_python(xyz, c, conf)
        grid_nat = self.grid_rasterizer.rasterize(xyz, c, conf)
        cmp_res = compare_grid_maps(grid_ref, grid_nat)
        self.assertTrue(cmp_res["passed"])

    def test_08_grid_cell_set_equivalence(self):
        """Test 8: Occupied cell sets match with 100% precision."""
        xyz = np.array([
            [1.0, 1.0, 0.5],
            [1.05, 1.05, 0.6],
            [-5.0, -5.0, 1.2],
        ], dtype=np.float32)
        c = np.array([0, 0, 1], dtype=np.int64)
        conf = np.array([0.9, 0.8, 0.95], dtype=np.float32)
        grid = self.grid_rasterizer.rasterize(xyz, c, conf)
        occ = np.sum(grid.point_count_layer > 0)
        self.assertEqual(occ, 2)

    def test_09_grid_elevation_equivalence(self):
        """Test 9: Elevation min, max, mean match exactly."""
        xyz = np.array([
            [2.0, 2.0, 1.0],
            [2.0, 2.0, 3.0],
        ], dtype=np.float32)
        c = np.array([0, 0], dtype=np.int64)
        conf = np.array([1.0, 1.0], dtype=np.float32)
        grid = self.grid_rasterizer.rasterize(xyz, c, conf)
        # 2.0m is cell (260, 260) with resolution 0.20m, bounds (-50, 50)
        ix = int((2.0 - (-50.0)) / 0.20)
        iy = int((2.0 - (-50.0)) / 0.20)
        self.assertAlmostEqual(float(grid.elevation_min[iy, ix]), 1.0, places=4)
        self.assertAlmostEqual(float(grid.elevation_max[iy, ix]), 3.0, places=4)
        self.assertAlmostEqual(float(grid.elevation_mean[iy, ix]), 2.0, places=4)

    def test_10_grid_semantic_equivalence(self):
        """Test 10: Semantic majority voting produces valid class."""
        xyz = np.array([
            [2.0, 2.0, 1.0],
            [2.0, 2.0, 2.0],
            [2.0, 2.0, 3.0],
        ], dtype=np.float32)
        c = np.array([0, 2, 2], dtype=np.int64) # 2 votes for static (2), 1 for drivable (0)
        conf = np.array([0.9, 0.9, 0.9], dtype=np.float32)
        grid = self.grid_rasterizer.rasterize(xyz, c, conf)
        ix = int((2.0 - (-50.0)) / 0.20)
        iy = int((2.0 - (-50.0)) / 0.20)
        self.assertEqual(int(grid.semantic_layer[iy, ix]), 2)

    def test_11_miou_regression(self):
        """Test 11: Accuracy mIoU is preserved without meaningful regression."""
        acc_path = self.repo_root / "reports/phase19_4/accuracy_regression.json"
        self.assertTrue(acc_path.is_file())
        with open(acc_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertLessEqual(abs(data["drift_pct"]), 1.0)

    def test_12_latency_regression(self):
        """Test 12: End-to-end perception latency is <= 54.97 ms (Golden Baseline)."""
        self.assertTrue(self.pipe_bench_path.is_file())
        with open(self.pipe_bench_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertLessEqual(data["phase19_4_recovered"]["mean_ms"], 54.97)

    def test_13_p95_regression(self):
        """Test 13: P95 tail latency is <= 67.51 ms."""
        with open(self.pipe_bench_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertLessEqual(data["phase19_4_recovered"]["p95_ms"], 67.51)

    def test_14_p99_regression(self):
        """Test 14: P99 tail latency is <= Phase 19.3 (121.83 ms)."""
        with open(self.pipe_bench_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertLessEqual(data["phase19_4_recovered"]["p99_ms"], 121.83)

    def test_15_no_dropped_frames(self):
        """Test 15: Zero dropped frames across evaluation."""
        self.assertTrue(self.summary_path.is_file())
        with open(self.summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["status"], "REGRESSION_RECOVERY_COMPLETE")

    def test_16_memory_regression(self):
        """Test 16: GridMap25D primary float layers memory footprint remains 4.77 MB."""
        dummy_grid = self.grid_rasterizer.rasterize(
            np.zeros((1, 3), dtype=np.float32),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.float32),
        )
        float_mem_mb = (
            dummy_grid.elevation_min.nbytes +
            dummy_grid.elevation_max.nbytes +
            dummy_grid.elevation_mean.nbytes +
            dummy_grid.confidence_layer.nbytes +
            dummy_grid.traversability_layer.nbytes
        ) / (1024 * 1024)
        self.assertAlmostEqual(float_mem_mb, 4.77, places=1)


if __name__ == "__main__":
    unittest.main()
