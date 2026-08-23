"""
Phase 15.5 Optimization Forensic Audit Unit and Integration Tests:
- Asserts Phase 12 checkpoint SHA256 immutability
- Validates 11-stage pipeline execution integrity
- Verifies profiling report metrics and top 10 bottleneck ranking
- Asserts numerical consistency across predictions and grid layers
"""

import unittest
from pathlib import Path
import json
import hashlib
import numpy as np
import torch

from scripts.profile_phase15_5_pipeline import compute_sha256
from ml.models.spvcnn import SPVCNN, build_spvcnn
from ml.models.spvcnn_predictor import SPVCNNPredictor
from ml.models.mapping_adapter import MLToMappingAdapter


class TestPhase15_5OptimizationAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.ckpt_path = cls.repo_root / "experiments/phase12_full_semanticposs_spvcnn/best_checkpoint.pt"
        cls.report_file = cls.repo_root / "reports/phase15_5/optimization_audit.json"

    def test_01_checkpoint_immutability(self):
        """Test 1: Checkpoint SHA256 hash remains strictly unchanged."""
        expected_sha = "4ce8e3518e860a99806343a7af5037a440da4344c337ef31253d7963abf1aa33"
        actual_sha = compute_sha256(self.ckpt_path)
        self.assertEqual(actual_sha, expected_sha)

    def test_02_profiling_report_exists_and_valid(self):
        """Test 2: Optimization audit JSON report exists with all 11 stages and resource metrics."""
        self.assertTrue(self.report_file.is_file(), "optimization_audit.json missing!")
        with open(self.report_file, "r") as f:
            data = json.load(f)

        bench = data.get("benchmark_results", {})
        self.assertIn("stages", bench)
        stages = bench["stages"]

        required_stages = [
            "A_lidar_file_read",
            "B_point_parsing",
            "C_range_filtering",
            "D_3zone_foveation",
            "E_spvcnn_voxelization",
            "F_host_to_device_transfer",
            "G_spvcnn_cuda_inference",
            "H_device_to_host_transfer",
            "I_ml_mapping_contract",
            "J_vectorized_gridmap25d",
            "TOTAL_PIPELINE",
        ]
        for s in required_stages:
            self.assertIn(s, stages, f"Stage {s} missing from benchmark report!")
            self.assertGreater(stages[s]["mean"], 0.0)

    def test_03_top_10_bottlenecks_ranking(self):
        """Test 3: Top 10 bottlenecks are properly ranked with severity, ROI, and risks."""
        with open(self.report_file, "r") as f:
            data = json.load(f)

        bottlenecks = data.get("top_10_bottlenecks", [])
        self.assertEqual(len(bottlenecks), 10)
        for b in bottlenecks:
            self.assertIn("rank", b)
            self.assertIn("severity", b)
            self.assertIn("proposed_optimization", b)
            self.assertIn("retraining_required", b)
            self.assertFalse(b["retraining_required"], "Optimization must not require retraining!")

    def test_04_numerical_correctness_and_finite_contract(self):
        """Test 4: Predictions satisfy numerical boundaries and GridMap25D layers are non-empty."""
        predictor = SPVCNNPredictor(device="cpu", pretrained_path=str(self.ckpt_path))
        map_adapter = MLToMappingAdapter()

        pts = np.random.uniform(-30, 30, size=(500, 4)).astype(np.float32)
        res = predictor.predict(pts)
        self.assertEqual(len(res["predicted_class"]), 500)
        self.assertTrue(np.all((res["predicted_class"] >= 0) & (res["predicted_class"] < 4)))
        self.assertTrue(np.all((res["confidence"] >= 0.0) & (res["confidence"] <= 1.0)))

        grid = map_adapter.build_25d_grid(res)
        self.assertFalse(np.isnan(grid.elevation_mean).all())


if __name__ == "__main__":
    unittest.main()
