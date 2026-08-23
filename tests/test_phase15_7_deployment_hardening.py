"""
Phase 15.7 Production Hardening & Deployment Verification Tests:
- Asserts checkpoint SHA256 cryptographic verification
- Validates fail-fast configuration schema enforcement
- Validates input point sanitizer against NaNs, Infs, and corrupt shapes
- Validates output PredictionBatch and GridMap25D contract compliance
- Validates failure recovery and artifact package assembly
"""

import unittest
from pathlib import Path
import json
import hashlib
import numpy as np
import torch
import yaml

from ml.pipeline.production_pipeline import (
    ProductionPipeline,
    ChecksumMismatchError,
    ConfigurationError,
    InputValidationError,
    verify_file_sha256,
)


class TestPhase15_7DeploymentHardening(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.config_path = cls.repo_root / "configs/production.yaml"
        cls.sample_bin = cls.repo_root / "dataset/sequences/02/velodyne/000001.bin"
        cls.pkg_dir = cls.repo_root / "artifacts/production"

    def test_01_config_and_checksum_verification(self):
        """Test 1: Checkpoint SHA256 matches production manifest and pipeline initializes."""
        pipeline = ProductionPipeline(self.config_path)
        self.assertIsNotNone(pipeline.model)
        self.assertEqual(pipeline.model.num_classes, 4)

    def test_02_checksum_mismatch_fails_fast(self):
        """Test 2: Pipeline aborts with ChecksumMismatchError on corrupted checksum."""
        corrupt_cfg = Path("configs/test_corrupt_temp.yaml")
        with open(self.config_path, "r", encoding="utf-8") as f:
            c = yaml.safe_load(f)
        c["checkpoint"]["expected_sha256"] = "1111111111111111111111111111111111111111111111111111111111111111"
        with open(corrupt_cfg, "w", encoding="utf-8") as f:
            yaml.dump(c, f)

        try:
            with self.assertRaises(ChecksumMismatchError):
                _ = ProductionPipeline(corrupt_cfg)
        finally:
            if corrupt_cfg.is_file():
                corrupt_cfg.unlink()

    def test_03_invalid_config_schema_fails_fast(self):
        """Test 3: Pipeline aborts with ConfigurationError on missing required fields."""
        bad_cfg = Path("configs/test_bad_temp.yaml")
        with open(bad_cfg, "w", encoding="utf-8") as f:
            yaml.dump({"pipeline": {"name": "incomplete"}}, f)

        try:
            with self.assertRaises(ConfigurationError):
                _ = ProductionPipeline(bad_cfg)
        finally:
            if bad_cfg.is_file():
                bad_cfg.unlink()

    def test_04_input_sanitizer_nan_inf_filtering(self):
        """Test 4: Input sanitizer cleans NaNs/Infs without raising uncaught exceptions."""
        pipeline = ProductionPipeline(self.config_path)
        pts = np.array([
            [1.0, 2.0, 3.0, 0.5],
            [np.nan, 2.0, 3.0, 0.5],
            [1.0, np.inf, 3.0, 0.5],
            [5.0, 5.0, 0.0, 0.8],
        ], dtype=np.float32)
        clean = pipeline.validate_raw_points(pts)
        self.assertEqual(len(clean), 2)
        self.assertTrue(np.all(np.isfinite(clean)))

    def test_05_malformed_shape_rejection(self):
        """Test 5: Reject malformed shape arrays gracefully with InputValidationError."""
        pipeline = ProductionPipeline(self.config_path)
        bad_shape = np.random.uniform(-10, 10, size=(50, 2)).astype(np.float32)
        with self.assertRaises(InputValidationError):
            _ = pipeline.validate_raw_points(bad_shape)

    def test_06_end_to_end_real_frame_processing(self):
        """Test 6: End-to-end processing yields successful PredictionBatch and GridMap25D."""
        pipeline = ProductionPipeline(self.config_path)
        from ml.data.dataset import load_point_cloud
        raw = load_point_cloud(self.sample_bin)
        _ = pipeline.process_frame(raw)  # Warmup
        res = pipeline.process_frame(raw, frame_id="test_000001")

        self.assertTrue(res.success)
        self.assertGreater(res.num_foveated_points, 0)
        self.assertIsNotNone(res.prediction_dto)
        self.assertIsNotNone(res.grid_map)
        self.assertFalse(np.isnan(res.grid_map.elevation_mean).all())
        self.assertLess(res.latency_ms, 300.0)

    def test_07_production_artifact_package_completeness(self):
        """Test 7: Production artifact package directory contains all required deployment manifests."""
        self.assertTrue(self.pkg_dir.is_dir(), "artifacts/production/ directory missing!")
        required_files = [
            "production.yaml",
            "checkpoint_sha256.txt",
            "model_metadata.json",
            "inference_entrypoint.py",
            "deployment_readme.md",
            "benchmark_report.json",
        ]
        for f in required_files:
            self.assertTrue((self.pkg_dir / f).is_file(), f"Missing artifact: {f}")


if __name__ == "__main__":
    unittest.main()
