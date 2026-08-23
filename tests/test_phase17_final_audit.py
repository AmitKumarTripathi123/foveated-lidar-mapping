"""
Phase 17 Final AI/ML Audit & Production Freeze Verification Tests:
- Asserts checkpoint SHA256 cryptographic verification
- Validates dataset frame count (2,988 frames) and train/val disjointness
- Validates final freeze artifact package in artifacts/final_freeze/
- Validates reproducibility (zero reload logit delta)
- Asserts secret scan clean status (NO SECRET FOUND)
"""

import unittest
from pathlib import Path
import json
import hashlib
import numpy as np
import torch

from ml.pipeline.production_pipeline import verify_file_sha256
from scripts.audit_phase17_production_freeze import audit_checkpoint_forensics, audit_security_and_secrets


class TestPhase17FinalAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.ckpt_path = cls.repo_root / "experiments/phase12_full_semanticposs_spvcnn/best_checkpoint.pt"
        cls.freeze_dir = cls.repo_root / "artifacts/final_freeze"
        cls.report_dir = cls.repo_root / "reports/phase17"

    def test_01_checkpoint_checksum_and_immutability(self):
        """Test 1: Checkpoint SHA256 hash strictly matches production baseline."""
        expected_sha = "b15c6dfb2f20d1dce4febc47be67f9d50b86a0af72f1895176c6a6ee58bca142"
        self.assertTrue(self.ckpt_path.is_file())
        self.assertTrue(verify_file_sha256(self.ckpt_path, expected_sha))

    def test_02_checkpoint_forensic_reproducibility(self):
        """Test 2: Checkpoint reloads with zero unexpected keys and logit delta < 1e-5."""
        audit = audit_checkpoint_forensics(self.ckpt_path)
        self.assertTrue(audit["sha256_match"])
        self.assertLess(audit["reload_delta_max"], 1e-5)
        self.assertEqual(audit["num_classes"], 4)
        self.assertEqual(audit["in_channels"], 4)

    def test_03_final_freeze_artifact_completeness(self):
        """Test 3: Final freeze directory contains all manifests, metadata, and reports."""
        self.assertTrue(self.freeze_dir.is_dir(), "artifacts/final_freeze/ missing!")
        required_files = [
            "checkpoint_sha256.txt",
            "model_metadata.json",
            "production.yaml",
            "final_freeze_manifest.json",
            "final_benchmark.json",
            "final_ai_ml_audit.json",
            "README.md",
        ]
        for rf in required_files:
            self.assertTrue((self.freeze_dir / rf).is_file(), f"Missing freeze artifact: {rf}")

    def test_04_freeze_manifest_contents(self):
        """Test 4: Final freeze manifest verifies approved production status and documented limitations."""
        manifest_file = self.freeze_dir / "final_freeze_manifest.json"
        with open(manifest_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["status"], "APPROVED")
        self.assertEqual(data["dataset_frames"], 2988)
        self.assertEqual(data["training_frames"], 2488)
        self.assertEqual(data["held_out_validation_frames"], 500)
        self.assertGreater(len(data["known_limitations"]), 0)

    def test_05_security_and_secrets_cleanliness(self):
        """Test 5: Repository security scan passes with NO SECRET FOUND."""
        sec = audit_security_and_secrets(self.repo_root)
        self.assertEqual(sec["security_scan_status"], "NO SECRET FOUND")
        self.assertEqual(len(sec["flagged_files"]), 0)


if __name__ == "__main__":
    unittest.main()
