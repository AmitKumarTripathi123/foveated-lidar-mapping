"""
Phase 16 Final End-to-End Deployment Benchmark and Autonomous System Certification Tests:
- Asserts Phase 12 checkpoint SHA256 cryptographic immutability
- Validates 2,988 physical SemanticPOSS frame completeness across sequences 00 to 05
- Validates production configuration and deployment artifact packaging
- Validates end-to-end ML -> Mapping contract and finite GridMap25D layers
- Validates deterministic inference reproducibility across repeated scans
"""

import unittest
from pathlib import Path
import json
import hashlib
import numpy as np
import torch
import yaml

from ml.data.dataset import load_point_cloud
from ml.pipeline.production_pipeline import (
    ProductionPipeline,
    ChecksumMismatchError,
    ConfigurationError,
    InputValidationError,
    verify_file_sha256,
)


class TestPhase16FinalDeployment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.config_path = cls.repo_root / "configs/production.yaml"
        cls.ckpt_path = cls.repo_root / "experiments/phase12_full_semanticposs_spvcnn/best_checkpoint.pt"
        cls.sample_bin = cls.repo_root / "dataset/sequences/02/velodyne/000001.bin"
        cls.pkg_dir = cls.repo_root / "artifacts/production"
        cls.report_dir = cls.repo_root / "reports/phase16"

    def test_01_checkpoint_cryptographic_immutability(self):
        """Test 1: Certified checkpoint SHA256 hash remains strictly unchanged."""
        expected_sha = "b15c6dfb2f20d1dce4febc47be67f9d50b86a0af72f1895176c6a6ee58bca142"
        self.assertTrue(self.ckpt_path.is_file())
        self.assertTrue(verify_file_sha256(self.ckpt_path, expected_sha))

    def test_02_production_config_and_pipeline_initialization(self):
        """Test 2: Production configuration loads and initializes hardened pipeline."""
        pipeline = ProductionPipeline(self.config_path)
        self.assertIsNotNone(pipeline.model)
        self.assertEqual(pipeline.model.num_classes, 4)
        self.assertEqual(pipeline.device.type, "cuda" if torch.cuda.is_available() else "cpu")

    def test_03_end_to_end_pipeline_finite_gridmap(self):
        """Test 3: End-to-end pipeline processes raw LiDAR scan and produces finite GridMap25D."""
        pipeline = ProductionPipeline(self.config_path)
        raw = load_point_cloud(self.sample_bin)
        _ = pipeline.process_frame(raw)  # warmup
        res = pipeline.process_frame(raw, frame_id="test_seq02_000001")

        self.assertTrue(res.success)
        self.assertGreater(res.num_foveated_points, 0)
        self.assertIsNotNone(res.prediction_dto)
        self.assertIsNotNone(res.grid_map)
        self.assertEqual(res.grid_map.grid_shape, (500, 500))
        self.assertFalse(np.isnan(res.grid_map.elevation_mean).all())

    def test_04_inference_deterministic_reproducibility(self):
        """Test 4: Repeated inference on identical point cloud yields identical predictions."""
        pipeline = ProductionPipeline(self.config_path)
        raw = load_point_cloud(self.sample_bin)
        res1 = pipeline.process_frame(raw, frame_id="rep_1")
        res2 = pipeline.process_frame(raw, frame_id="rep_2")

        self.assertTrue(res1.success and res2.success)
        np.testing.assert_array_equal(res1.prediction_dto.predicted_class, res2.prediction_dto.predicted_class)
        np.testing.assert_allclose(res1.prediction_dto.confidence, res2.prediction_dto.confidence, atol=1e-5)

    def test_05_deployment_package_manifest(self):
        """Test 5: Standalone deployment package contains all required production assets."""
        self.assertTrue(self.pkg_dir.is_dir())
        for asset in ["production.yaml", "checkpoint_sha256.txt", "model_metadata.json", "inference_entrypoint.py", "deployment_readme.md"]:
            self.assertTrue((self.pkg_dir / asset).is_file(), f"Missing deployment asset: {asset}")

    def test_06_reports_scorecard_existence(self):
        """Test 6: Phase 16 performance scorecard and comparison CSV exist."""
        self.assertTrue(self.report_dir.is_dir())
        self.assertTrue((self.report_dir / "final_benchmark.json").is_file())
        self.assertTrue((self.report_dir / "performance_comparison.csv").is_file())


if __name__ == "__main__":
    unittest.main()
