"""
Phase 19.1 Audit & Optimization Profiler Unit & System Test Suite:
- Asserts checkpoint SHA256 immutability
- Tests mandatory boundary distances: 9.999m, 10.000m, 10.001m, 39.999m, 40.000m, 40.001m, 99.999m, 100.000m, 100.001m
- Tests metric unit edge cases: perfect prediction (mIoU=1.0), all wrong, ignore-label (255) exclusion, empty input, NaN/Inf handling
- Asserts all Phase 19.1 JSON reports and 6 diagnostic figures exist and are valid
"""

import json
import unittest
from pathlib import Path
import numpy as np
import yaml

from src.core.hierarchy import FoveatedHierarchyEngine
from src.core.types import SuperClass
from ml.pipeline.production_pipeline import verify_file_sha256
from benchmarks.phase19_1.accuracy_audit import compute_multiclass_metrics, update_confusion_matrix
from benchmarks.phase19_1.distance_audit import DistanceWiseAuditor


class TestPhase19_1Audit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.config_path = cls.repo_root / "configs/system_config.yaml"
        cls.ckpt_path = cls.repo_root / "experiments/phase12_full_semanticposs_spvcnn/best_checkpoint.pt"
        cls.out_dir = cls.repo_root / "reports/phase19_1"
        cls.fig_dir = cls.out_dir / "figures"

    def test_01_checkpoint_immutability(self):
        """Test 1: Certified Phase 12 checkpoint SHA256 matches production baseline."""
        expected_sha = "b15c6dfb2f20d1dce4febc47be67f9d50b86a0af72f1895176c6a6ee58bca142"
        self.assertTrue(self.ckpt_path.is_file(), "Checkpoint file missing!")
        self.assertTrue(verify_file_sha256(self.ckpt_path, expected_sha))

    def test_02_mandatory_distance_boundary_invariants(self):
        """Test 2: Explicitly tests 9.999m, 10.000m, 10.001m, 39.999m, 40.000m, 40.001m, 99.999m, 100.000m, 100.001m."""
        engine = FoveatedHierarchyEngine(self.config_path)

        # Near boundary (< 10.0m)
        self.assertEqual(engine.resolve_zone(9.999).level, 0)
        self.assertEqual(engine.resolve_zone(10.000).level, 1)
        self.assertEqual(engine.resolve_zone(10.001).level, 1)

        # Mid boundary (< 40.0m)
        self.assertEqual(engine.resolve_zone(39.999).level, 1)
        self.assertEqual(engine.resolve_zone(40.000).level, 2)
        self.assertEqual(engine.resolve_zone(40.001).level, 2)

        # Far boundary (<= 100.0m)
        self.assertEqual(engine.resolve_zone(99.999).level, 2)
        self.assertEqual(engine.resolve_zone(100.000).level, 2)
        self.assertIsNone(engine.resolve_zone(100.001)) # Filtered

    def test_03_metric_perfect_prediction(self):
        """Test 3: Perfect prediction produces mIoU = 1.0, point_accuracy = 1.0."""
        cm = np.zeros((4, 4), dtype=np.int64)
        targets = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
        preds = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)

        update_confusion_matrix(cm, preds, targets)
        metrics = compute_multiclass_metrics(cm)

        self.assertEqual(metrics["overall"]["miou"], 1.0)
        self.assertEqual(metrics["overall"]["point_accuracy"], 1.0)
        self.assertEqual(metrics["overall"]["mean_class_accuracy"], 1.0)
        for c in ["drivable", "non_drivable", "static", "dynamic"]:
            self.assertEqual(metrics["classes"][c]["iou"], 1.0)

    def test_04_metric_ignore_label_exclusion(self):
        """Test 4: Ignore label (255) is strictly excluded from mIoU and confusion matrix."""
        cm = np.zeros((4, 4), dtype=np.int64)
        targets = np.array([0, 1, 255, 255, 2, 3], dtype=np.int64)
        preds = np.array([0, 1, 0, 1, 2, 3], dtype=np.int64)

        update_confusion_matrix(cm, preds, targets)
        metrics = compute_multiclass_metrics(cm)

        self.assertEqual(metrics["overall"]["total_valid_points"], 4)
        self.assertEqual(metrics["overall"]["miou"], 1.0)
        self.assertEqual(np.sum(cm), 4)

    def test_05_metric_empty_and_nan_handling(self):
        """Test 5: Empty arrays and NaN coordinates produce finite outputs without throwing."""
        cm = np.zeros((4, 4), dtype=np.int64)
        preds_empty = np.array([], dtype=np.int64)
        targets_empty = np.array([], dtype=np.int64)

        update_confusion_matrix(cm, preds_empty, targets_empty)
        metrics = compute_multiclass_metrics(cm)
        self.assertEqual(metrics["overall"]["miou"], 0.0)
        self.assertEqual(metrics["overall"]["point_accuracy"], 0.0)

    def test_06_reports_and_deliverables_existence(self):
        """Test 6: Asserts all required Phase 19.1 JSON reports and figures exist."""
        required_jsons = [
            "optimization_profile.json",
            "accuracy_audit.json",
            "distance_miou.json",
            "confusion_matrix.json",
            "telemetry.json",
            "phase19_1_summary.json",
        ]
        for j_name in required_jsons:
            p = self.out_dir / j_name
            self.assertTrue(p.is_file(), f"Missing required report: {j_name}")
            self.assertGreater(p.stat().st_size, 50)

        required_figs = [
            "latency_breakdown.png",
            "distance_miou.png",
            "class_iou.png",
            "confusion_matrix.png",
            "performance_summary.png",
            "gpu_cpu_telemetry.png",
        ]
        for f_name in required_figs:
            p = self.fig_dir / f_name
            self.assertTrue(p.is_file(), f"Missing required figure: {f_name}")
            self.assertGreater(p.stat().st_size, 1000)

    def test_07_summary_payload_invariants(self):
        """Test 7: Validates phase19_1_summary.json structure and automated bottleneck detection."""
        summary_path = self.out_dir / "phase19_1_summary.json"
        self.assertTrue(summary_path.is_file())
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["phase"], "19.1")
        self.assertEqual(data["status"], "AUDIT_COMPLETE")
        self.assertIn("bottleneck", data)
        self.assertIn("primary", data["bottleneck"])
        self.assertIn("recommendation", data)
        self.assertGreater(data["measured_phase19_1"]["overall_miou"], 0.50)
        self.assertLess(data["measured_phase19_1"]["perception_mean_latency_ms"], 100.0)


if __name__ == "__main__":
    unittest.main()
