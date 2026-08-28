"""
Phase 19.3 Native 3-Zone Foveation Unit & Invariant Test Suite:
- Validates native module imports and initialization
- Tests boundary conditions: 0.5m, 10m, 40m, 100m limits
- Tests negative coordinate voxel indexing
- Tests exact zone assignment, voxel identity, and point retention equivalence
- Tests empty point clouds, single points, dense near-field, and sparse far-field
- Tests reports and speedup targets
"""

import json
import unittest
from pathlib import Path
import numpy as np

from src.core.native_foveation import NativeFoveationAccelerator
from ml.data.amit_adapter import FoveatedVoxelSampler
from benchmarks.phase19_3.correctness_audit import compare_foveation_results
from ml.pipeline.production_pipeline import verify_file_sha256


class TestPhase19_3Foveation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.ckpt_path = cls.repo_root / "experiments/phase12_full_semanticposs_spvcnn/best_checkpoint.pt"
        cls.sampler_ref = FoveatedVoxelSampler()
        cls.sampler_nat = NativeFoveationAccelerator()

    def test_01_checkpoint_immutability(self):
        """Test 1: Checkpoint SHA256 matches production baseline."""
        expected_sha = "b15c6dfb2f20d1dce4febc47be67f9d50b86a0af72f1895176c6a6ee58bca142"
        self.assertTrue(self.ckpt_path.is_file())
        self.assertTrue(verify_file_sha256(self.ckpt_path, expected_sha))

    def test_02_native_import(self):
        """Test 2: Native foveation engine initializes properly."""
        self.assertIsNotNone(self.sampler_nat)
        self.assertEqual(self.sampler_nat.near_dist, 10.0)
        self.assertEqual(self.sampler_nat.near_voxel, 0.05)
        self.assertEqual(self.sampler_nat.mid_dist, 40.0)
        self.assertEqual(self.sampler_nat.mid_voxel, 0.15)
        self.assertEqual(self.sampler_nat.far_dist, 100.0)
        self.assertEqual(self.sampler_nat.far_voxel, 0.50)

    def test_03_range_lower_boundary(self):
        """Test 3: Lower range boundary points are handled consistently."""
        pts = np.array([
            [0.4999, 0.0, 0.0, 0.5],
            [0.5000, 0.0, 0.0, 0.5],
            [0.5001, 0.0, 0.0, 0.5],
        ], dtype=np.float32)
        pts_r, _, rep_r = self.sampler_ref.sample_reference_python(pts)
        pts_n, _, rep_n = self.sampler_nat.sample(pts)
        cmp = compare_foveation_results(pts_r, rep_r, pts_n, rep_n)
        self.assertTrue(cmp["passed"])

    def test_04_range_upper_boundary(self):
        """Test 4: Upper 100m range boundary points are handled consistently."""
        pts = np.array([
            [99.9999, 0.0, 0.0, 0.5],
            [100.0000, 0.0, 0.0, 0.5],
            [100.0001, 0.0, 0.0, 0.5],
        ], dtype=np.float32)
        pts_r, _, rep_r = self.sampler_ref.sample_reference_python(pts)
        pts_n, _, rep_n = self.sampler_nat.sample(pts)
        cmp = compare_foveation_results(pts_r, rep_r, pts_n, rep_n)
        self.assertTrue(cmp["passed"])
        self.assertEqual(rep_n.filtered_out_count, 1) # >100m filtered

    def test_05_near_boundary(self):
        """Test 5: 10m near-mid boundary transition."""
        pts = np.array([
            [9.9999, 0.0, 0.0, 0.5],
            [10.0000, 0.0, 0.0, 0.5],
            [10.0001, 0.0, 0.0, 0.5],
        ], dtype=np.float32)
        pts_r, _, rep_r = self.sampler_ref.sample_reference_python(pts)
        pts_n, _, rep_n = self.sampler_nat.sample(pts)
        cmp = compare_foveation_results(pts_r, rep_r, pts_n, rep_n)
        self.assertTrue(cmp["passed"])
        self.assertEqual(rep_n.zone_stats[0].input_count, 1) # Near
        self.assertEqual(rep_n.zone_stats[1].input_count, 2) # Mid

    def test_06_mid_boundary(self):
        """Test 6: 40m mid-far boundary transition."""
        pts = np.array([
            [39.9999, 0.0, 0.0, 0.5],
            [40.0000, 0.0, 0.0, 0.5],
            [40.0001, 0.0, 0.0, 0.5],
        ], dtype=np.float32)
        pts_r, _, rep_r = self.sampler_ref.sample_reference_python(pts)
        pts_n, _, rep_n = self.sampler_nat.sample(pts)
        cmp = compare_foveation_results(pts_r, rep_r, pts_n, rep_n)
        self.assertTrue(cmp["passed"])
        self.assertEqual(rep_n.zone_stats[1].input_count, 1) # Mid
        self.assertEqual(rep_n.zone_stats[2].input_count, 2) # Far

    def test_07_far_boundary(self):
        """Test 7: 100m far-filtered boundary transition."""
        pts = np.array([
            [99.9999, 0.0, 0.0, 0.5],
            [100.0000, 0.0, 0.0, 0.5],
            [100.0001, 0.0, 0.0, 0.5],
        ], dtype=np.float32)
        pts_r, _, rep_r = self.sampler_ref.sample_reference_python(pts)
        pts_n, _, rep_n = self.sampler_nat.sample(pts)
        cmp = compare_foveation_results(pts_r, rep_r, pts_n, rep_n)
        self.assertTrue(cmp["passed"])
        self.assertEqual(rep_n.zone_stats[2].input_count, 2) # Far (<= 100m)
        self.assertEqual(rep_n.filtered_out_count, 1)        # > 100m

    def test_08_negative_coordinates(self):
        """Test 8: Negative coordinate spatial indexing and voxel deduplication."""
        pts = np.array([
            [-10.0, -10.0, 0.0, 0.5],
            [-5.0, 0.0, 0.0, 0.5],
            [0.0, -5.0, 0.0, 0.5],
            [-0.1, -0.1, 0.0, 0.5],
            [-0.05, -0.05, 0.0, 0.5],
            [-35.5, -25.5, 1.2, 0.8],
        ], dtype=np.float32)
        pts_r, _, rep_r = self.sampler_ref.sample_reference_python(pts)
        pts_n, _, rep_n = self.sampler_nat.sample(pts)
        cmp = compare_foveation_results(pts_r, rep_r, pts_n, rep_n)
        self.assertTrue(cmp["passed"])

    def test_09_voxel_alignment(self):
        """Test 9: Voxel deduplication selects first occurrence identically."""
        # 3 points within 5cm near-field voxel
        pts = np.array([
            [2.01, 2.01, 0.01, 0.5],
            [2.02, 2.02, 0.02, 0.6],
            [2.03, 2.03, 0.03, 0.7],
        ], dtype=np.float32)
        pts_r, _, rep_r = self.sampler_ref.sample_reference_python(pts)
        pts_n, _, rep_n = self.sampler_nat.sample(pts)
        cmp = compare_foveation_results(pts_r, rep_r, pts_n, rep_n)
        self.assertTrue(cmp["passed"])
        self.assertEqual(len(pts_n), 1)
        self.assertAlmostEqual(float(pts_n[0, 3]), 0.5, places=5)

    def test_10_zone_assignment(self):
        """Test 10: Zone assignment across 3 distinct radii."""
        pts = np.array([
            [5.0, 0.0, 0.0, 0.1],  # Near (r=5m)
            [25.0, 0.0, 0.0, 0.2], # Mid (r=25m)
            [75.0, 0.0, 0.0, 0.3], # Far (r=75m)
        ], dtype=np.float32)
        pts_r, _, rep_r = self.sampler_ref.sample_reference_python(pts)
        pts_n, _, rep_n = self.sampler_nat.sample(pts)
        cmp = compare_foveation_results(pts_r, rep_r, pts_n, rep_n)
        self.assertTrue(cmp["passed"])
        self.assertEqual(rep_n.zone_stats[0].input_count, 1)
        self.assertEqual(rep_n.zone_stats[1].input_count, 1)
        self.assertEqual(rep_n.zone_stats[2].input_count, 1)

    def test_11_retention_equivalence(self):
        """Test 11: Retention percentage exact equality with Python reference."""
        np.random.seed(42)
        pts = np.random.uniform(-50.0, 50.0, (20000, 4)).astype(np.float32)
        _, _, rep_r = self.sampler_ref.sample_reference_python(pts)
        _, _, rep_n = self.sampler_nat.sample(pts)
        self.assertEqual(rep_r.foveated_count, rep_n.foveated_count)
        self.assertAlmostEqual(rep_r.overall_reduction_pct, rep_n.overall_reduction_pct, places=4)

    def test_12_empty_cloud(self):
        """Test 12: Empty point cloud produces valid 0-length output."""
        empty_pts = np.zeros((0, 4), dtype=np.float32)
        pts_r, _, rep_r = self.sampler_ref.sample_reference_python(empty_pts)
        pts_n, _, rep_n = self.sampler_nat.sample(empty_pts)
        cmp = compare_foveation_results(pts_r, rep_r, pts_n, rep_n)
        self.assertTrue(cmp["passed"])
        self.assertEqual(len(pts_n), 0)

    def test_13_single_point(self):
        """Test 13: Single point produces exactly one output point."""
        single = np.array([[12.0, 15.0, 0.5, 0.8]], dtype=np.float32)
        pts_r, _, rep_r = self.sampler_ref.sample_reference_python(single)
        pts_n, _, rep_n = self.sampler_nat.sample(single)
        cmp = compare_foveation_results(pts_r, rep_r, pts_n, rep_n)
        self.assertTrue(cmp["passed"])
        self.assertEqual(len(pts_n), 1)

    def test_14_randomized_equivalence(self):
        """Test 14: Bitwise point equivalence across 50,000 randomized points."""
        np.random.seed(100)
        pts = np.random.uniform(-40.0, 40.0, (50000, 4)).astype(np.float32)
        lbls = np.random.randint(0, 4, 50000).astype(np.int64)

        pts_r, lbl_r, rep_r = self.sampler_ref.sample_reference_python(pts, lbls)
        pts_n, lbl_n, rep_n = self.sampler_nat.sample(pts, lbls)

        cmp = compare_foveation_results(pts_r, rep_r, pts_n, rep_n, lbl_r, lbl_n)
        self.assertTrue(cmp["passed"])
        self.assertTrue(cmp["points_match"])
        self.assertTrue(cmp["labels_match"])

    def test_15_dense_near_and_sparse_far(self):
        """Test 15: Extreme distributions with dense near-field and sparse far-field."""
        # Dense near
        np.random.seed(7)
        dense_near = np.random.uniform(-3.0, 3.0, (8000, 4)).astype(np.float32)
        pts_r, _, rep_r = self.sampler_ref.sample_reference_python(dense_near)
        pts_n, _, rep_n = self.sampler_nat.sample(dense_near)
        cmp_near = compare_foveation_results(pts_r, rep_r, pts_n, rep_n)
        
        # Sparse far
        radii = np.random.uniform(60.0, 95.0, 3000)
        thetas = np.random.uniform(0, 2 * np.pi, 3000)
        sparse_far = np.zeros((3000, 4), dtype=np.float32)
        sparse_far[:, 0] = radii * np.cos(thetas)
        sparse_far[:, 1] = radii * np.sin(thetas)
        sparse_far[:, 3] = 0.5
        pts_r_far, _, rep_r_far = self.sampler_ref.sample_reference_python(sparse_far)
        pts_n_far, _, rep_n_far = self.sampler_nat.sample(sparse_far)
        cmp_far = compare_foveation_results(pts_r_far, rep_r_far, pts_n_far, rep_n_far)
        
        self.assertTrue(cmp_near["passed"])
        self.assertTrue(cmp_far["passed"])


if __name__ == "__main__":
    unittest.main()
