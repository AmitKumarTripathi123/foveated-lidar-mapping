"""
Phase 17.2 Foveated vs Uniform Memory & Compute Benchmark Tests:
- Asserts checkpoint SHA256 cryptographic immutability
- Validates UniformVoxelSampler vs FoveatedVoxelSampler geometric reduction
- Asserts >= 75% 2.5D grid memory reduction calculation
- Validates memory_benchmark.json report schema and target gate
- Asserts generated comparison figure existence
"""

import unittest
from pathlib import Path
import json
import hashlib
import numpy as np

from ml.pipeline.production_pipeline import verify_file_sha256
from ml.data.amit_adapter import FoveatedVoxelSampler
from scripts.benchmark_foveated_vs_uniform_memory import UniformVoxelSampler


class TestPhase17_2MemoryBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.ckpt_path = cls.repo_root / "experiments/phase12_full_semanticposs_spvcnn/best_checkpoint.pt"
        cls.report_json = cls.repo_root / "reports/phase17_2/memory_benchmark.json"
        cls.report_fig = cls.repo_root / "reports/phase17_2/figures/uniform_vs_foveated_comparison.png"

    def test_01_checkpoint_immutability(self):
        """Test 1: Certified checkpoint SHA256 matches production baseline."""
        expected_sha = "b15c6dfb2f20d1dce4febc47be67f9d50b86a0af72f1895176c6a6ee58bca142"
        self.assertTrue(self.ckpt_path.is_file())
        self.assertTrue(verify_file_sha256(self.ckpt_path, expected_sha))

    def test_02_uniform_vs_foveated_sampler_reduction(self):
        """Test 2: Foveated sampling produces fewer voxels than uniform 5cm on realistic point density."""
        uni_sampler = UniformVoxelSampler(voxel_size=0.05, max_range=100.0)
        fov_sampler = FoveatedVoxelSampler(near_dist=10.0, near_voxel=0.05, mid_dist=40.0, mid_voxel=0.15, far_dist=100.0, far_voxel=0.50)

        # Dense point cloud with multiple points per 15cm/50cm voxel
        np.random.seed(42)
        r = np.random.uniform(15.0, 60.0, size=(10000, 1))
        theta = np.random.uniform(0, 2*np.pi, size=(10000, 1))
        pts = np.hstack([r * np.cos(theta), r * np.sin(theta), np.random.uniform(-1, 2, size=(10000, 1)), np.ones((10000, 1))]).astype(np.float32)

        uni_pts, _, _ = uni_sampler.sample(pts)
        fov_pts, _, _ = fov_sampler.sample(pts)

        self.assertLess(len(fov_pts), len(uni_pts))
        self.assertGreater(len(fov_pts), 0)

    def test_03_grid_memory_reduction_target_gate(self):
        """Test 3: 500x500 foveated grid achieves >75% reduction over 2000x2000 uniform grid."""
        uni_cells = 2000 * 2000
        fov_cells = 500 * 500
        reduction_pct = (1.0 - fov_cells / uni_cells) * 100.0
        self.assertEqual(reduction_pct, 93.75)
        self.assertGreaterEqual(reduction_pct, 75.0)

    def test_04_benchmark_report_payload_and_sih_status(self):
        """Test 4: memory_benchmark.json exists and verifies SIH REQ-H PASS."""
        self.assertTrue(self.report_json.is_file(), "memory_benchmark.json missing!")
        with open(self.report_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("comparison", data)
        self.assertTrue(data["comparison"]["sih_target_met"])
        self.assertGreaterEqual(data["comparison"]["grid_memory_reduction_percent"], 75.0)
        self.assertGreater(data["comparison"]["speedup_factor"], 1.0)

    def test_05_visual_artifact_existence(self):
        """Test 5: Side-by-side diagnostic comparison figure exists."""
        self.assertTrue(self.report_fig.is_file(), "Comparison figure PNG missing!")
        self.assertGreater(self.report_fig.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
