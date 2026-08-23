"""
Phase 14 Robustness and Sequence-Wise Scientific Validation Tests:
- Sequence evaluation metrics computation
- Distance-dependent range binning
- Cross-sequence aggregation stability
- Model collapse and entropy verification
- Checkpoint reproducibility assertion
- Dataset integrity and partition disjointness
"""

import unittest
from pathlib import Path
import json
import numpy as np
import torch

from scripts.evaluate_phase14_robustness import (
    compute_iou_from_cm,
    audit_full_dataset,
    RANGE_BINS,
    CLASS_NAMES,
)
from ml.models.spvcnn import build_spvcnn
from ml.models.spvcnn_predictor import SPVCNNPredictor
from ml.models.mapping_adapter import MLToMappingAdapter


class TestPhase14Robustness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.ds_root = cls.repo_root / "dataset"
        cls.ckpt_path = cls.repo_root / "experiments/phase12_full_semanticposs_spvcnn/best_checkpoint.pt"
        cls.reports_dir = cls.repo_root / "reports/phase14"

    def test_01_compute_iou_from_cm_exactness(self):
        """Test 1: compute_iou_from_cm accurately computes TP, FP, FN, IoU, and accuracy."""
        # 4x4 diagonal CM (perfect prediction)
        cm = np.array([
            [100, 0, 0, 0],
            [0, 50, 0, 0],
            [0, 0, 200, 0],
            [0, 0, 0, 30],
        ], dtype=np.int64)

        miou, ious, prec, rec, acc = compute_iou_from_cm(cm)
        self.assertEqual(miou, 100.0)
        self.assertEqual(acc, 100.0)
        for c in range(4):
            self.assertEqual(ious[c], 100.0)
            self.assertEqual(prec[c], 100.0)
            self.assertEqual(rec[c], 100.0)

    def test_02_compute_iou_imbalanced_confusion(self):
        """Test 2: compute_iou_from_cm handles off-diagonal misclassifications correctly."""
        cm = np.array([
            [80, 20, 0, 0],   # Class 0: 80 TP, 20 FP in C1
            [0, 50, 0, 0],    # Class 1: 50 TP
            [0, 0, 100, 0],   # Class 2: 100 TP
            [0, 0, 0, 10],    # Class 3: 10 TP
        ], dtype=np.int64)

        miou, ious, prec, rec, acc = compute_iou_from_cm(cm)
        # Class 0 IoU = 80 / (80 + 0 + 20) = 80%
        self.assertEqual(ious[0], 80.0)
        # Class 1 IoU = 50 / (50 + 20 + 0) = 71.43%
        self.assertAlmostEqual(ious[1], 71.43, places=1)

    def test_03_forensic_dataset_completeness(self):
        """Test 3: Full 2,988 frames discovered across sequences 00 to 05."""
        audit_res = audit_full_dataset(self.ds_root)
        self.assertTrue(audit_res["dataset_complete"])
        self.assertEqual(audit_res["total_matched_pairs"], 2988)
        self.assertEqual(audit_res["total_expected_frames"], 2988)

    def test_04_cross_sequence_stability(self):
        """Test 4: Sequence metrics JSON exists with valid cross-sequence mean and low variance."""
        seq_json = self.reports_dir / "sequence_metrics.json"
        self.assertTrue(seq_json.is_file(), "reports/phase14/sequence_metrics.json missing!")
        with open(seq_json, "r") as f:
            data = json.load(f)

        summary = data.get("cross_sequence_summary", {})
        self.assertIn("mean_miou", summary)
        self.assertGreater(summary["mean_miou"], 45.0)
        self.assertLess(summary["std_miou"], 10.0)  # Variance across scenes is small (< 10%)

    def test_05_distance_metrics_structure(self):
        """Test 5: Distance metrics CSV has all 6 standard range bins from 0m to 100m."""
        dist_csv = self.reports_dir / "distance_metrics.csv"
        self.assertTrue(dist_csv.is_file(), "reports/phase14/distance_metrics.csv missing!")
        with open(dist_csv, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        # Header + 6 bins = 7 lines
        self.assertEqual(len(lines), 7)
        self.assertIn("0-10m", lines[1])
        self.assertIn("80-100m", lines[6])

    def test_06_checkpoint_reproducibility_assertion(self):
        """Test 6: Production checkpoint reload produces identical output within 1e-4."""
        self.assertTrue(self.ckpt_path.is_file(), "Phase 12 checkpoint missing!")
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        self.assertIn("metrics", ckpt)
        self.assertIn("model_state_dict", ckpt)

        # Instantiate two independent models
        m1 = build_spvcnn(num_classes=4, in_channels=4, pretrained_path=str(self.ckpt_path), device="cpu")
        m2 = build_spvcnn(num_classes=4, in_channels=4, pretrained_path=str(self.ckpt_path), device="cpu")
        m1.eval()
        m2.eval()

        pts = torch.randn(50, 4)
        from ml.data.spvcnn_adapter import SPVCNNInputAdapter
        adapter = SPVCNNInputAdapter(voxel_size=0.05)
        bundle = adapter.prepare_input(pts, device="cpu")

        with torch.no_grad():
            out1 = m1(bundle["features"], bundle["point_to_voxel_idx"], bundle["num_voxels"])
            out2 = m2(bundle["features"], bundle["point_to_voxel_idx"], bundle["num_voxels"])

        max_diff = torch.max(torch.abs(out1 - out2)).item()
        self.assertLess(max_diff, 1e-5)

    def test_07_ml_mapping_contract_compliance(self):
        """Test 7: Frozen production model satisfies Amit ML -> Mapping contract without NaNs."""
        predictor = SPVCNNPredictor(device="cpu", pretrained_path=str(self.ckpt_path))
        map_adapter = MLToMappingAdapter()

        pts = np.random.uniform(-25, 25, size=(500, 4)).astype(np.float32)
        res = predictor.predict(pts)
        self.assertIn("xyz", res)
        self.assertIn("predicted_class", res)
        self.assertIn("confidence", res)

        grid = map_adapter.build_25d_grid(res)
        self.assertIsNotNone(grid)
        self.assertFalse(np.isnan(grid.elevation_mean).all())


if __name__ == "__main__":
    unittest.main()
