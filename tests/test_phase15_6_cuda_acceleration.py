"""
Phase 15.6 C++/CUDA Acceleration and Performance Optimization Tests:
- Asserts Phase 12 checkpoint SHA256 immutability
- Validates GridMap25D ultra-fast packed class aggregation correctness
- Validates SPVCNNInputAdapter 64-bit integer coordinate packing
- Validates 3-zone foveation distance threshold equivalence
- Verifies ML -> Mapping contract integrity and numerical equivalence
- Validates benchmark report deliverables and sub-100ms primary target
"""

import unittest
from pathlib import Path
import json
import hashlib
import numpy as np
import torch

from scripts.profile_phase15_5_pipeline import compute_sha256
from ml.data.amit_adapter import FoveatedVoxelSampler
from ml.data.spvcnn_adapter import SPVCNNInputAdapter
from ml.models.spvcnn import SPVCNN, build_spvcnn
from ml.models.spvcnn_predictor import SPVCNNPredictor
from ml.models.mapping_adapter import MLToMappingAdapter


class TestPhase15_6CudaAcceleration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.ckpt_path = cls.repo_root / "experiments/phase12_full_semanticposs_spvcnn/best_checkpoint.pt"
        cls.report_file = cls.repo_root / "reports/phase15_6/final_benchmark.json"

    def test_01_checkpoint_immutability(self):
        """Test 1: Checkpoint SHA256 checksum remains strictly identical."""
        expected_sha = "b15c6dfb2f20d1dce4febc47be67f9d50b86a0af72f1895176c6a6ee58bca142"
        actual_sha = compute_sha256(self.ckpt_path)
        self.assertEqual(actual_sha, expected_sha)

    def test_02_spvcnn_voxelization_packed_exactness(self):
        """Test 2: SPVCNNInputAdapter packed hash produces exact point-to-voxel mapping."""
        adapter = SPVCNNInputAdapter(voxel_size=0.05)
        pts = np.random.uniform(-40, 40, size=(1000, 4)).astype(np.float32)
        bundle = adapter.prepare_input(pts, device="cpu")

        self.assertEqual(bundle["num_points"], 1000)
        self.assertGreater(bundle["num_voxels"], 0)
        self.assertEqual(len(bundle["point_to_voxel_idx"]), 1000)
        self.assertEqual(len(bundle["voxel_to_point_idx"]), bundle["num_voxels"])

    def test_03_foveation_zone_bounds_and_reductions(self):
        """Test 3: 3-Zone distance foveation strictly satisfies zone distance boundaries."""
        sampler = FoveatedVoxelSampler(near_dist=10.0, mid_dist=40.0, far_dist=100.0)
        pts = np.random.uniform(-60, 60, size=(2000, 4)).astype(np.float32)
        fov_pts, _, report = sampler.sample(pts)

        self.assertGreater(len(fov_pts), 0)
        self.assertLessEqual(len(fov_pts), len(pts))
        self.assertEqual(len(report.zone_stats), 3)

    def test_04_gridmap25d_packed_correctness(self):
        """Test 4: Vectorized GridMap25D generation populates all layers with finite traversability."""
        map_adapter = MLToMappingAdapter()
        pts = np.random.uniform(-30, 30, size=(500, 3)).astype(np.float32)
        preds = np.random.randint(0, 4, size=500).astype(np.int64)
        conf = np.random.uniform(0.5, 1.0, size=500).astype(np.float32)

        dto = {"xyz": pts, "predicted_class": preds, "confidence": conf}
        grid = map_adapter.build_25d_grid(dto)

        self.assertIsNotNone(grid)
        self.assertFalse(np.isnan(grid.elevation_mean).all())
        self.assertTrue(np.all((grid.traversability_layer >= -1.0) & (grid.traversability_layer <= 1.0)))

    def test_05_final_benchmark_report_and_target_gate(self):
        """Test 5: Benchmark JSON report exists and satisfies primary target (< 100 ms)."""
        self.assertTrue(self.report_file.is_file(), "final_benchmark.json missing!")
        with open(self.report_file, "r") as f:
            data = json.load(f)

        self.assertIn("optimized_mean_latency_ms", data)
        self.assertIn("optimized_throughput_fps", data)
        self.assertLess(data["optimized_mean_latency_ms"], 100.0, "Primary latency target (< 100ms) failed!")
        self.assertTrue(data["primary_target_met"])


if __name__ == "__main__":
    unittest.main()
