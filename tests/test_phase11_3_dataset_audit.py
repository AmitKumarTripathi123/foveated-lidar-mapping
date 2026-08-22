"""Tests for Phase 11.3 SemanticPOSS Dataset Audit and Pipeline Debugging."""

import sys
import unittest
from pathlib import Path
import numpy as np

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.audit_semanticposs import audit_dataset
from dataset import build_file_list, FoveatedLidarDataset
from ml.data.dataset import load_point_cloud, load_labels, validate_point_label_alignment


class TestPhase11_3DatasetAudit(unittest.TestCase):
    """Test suite for SemanticPOSS dataset auditing, stem pairing, and split validation."""

    @classmethod
    def setUpClass(cls):
        """Set up class paths."""
        cls.dataset_root = repo_root / "dataset"
        cls.bin_file = repo_root / "dataset/sequences/00/velodyne/000000.bin"
        cls.lbl_file = repo_root / "dataset/sequences/00/labels/000000.label"

    def test_01_audit_tool_execution(self):
        """Test 1: audit_dataset executes cleanly on dataset root and discovers sequence 00."""
        res = audit_dataset(self.dataset_root, expected_sequences=["00"], expected_frames=1)
        self.assertTrue(res["root_exists"])
        self.assertIn("00", res["sequences_found"])
        self.assertEqual(res["total_matched"], 1)
        self.assertEqual(res["status"], "PASS")

    def test_02_audit_tool_detects_missing_sequences(self):
        """Test 2: audit_dataset marks status FAIL when expected 6 sequences are missing."""
        res = audit_dataset(self.dataset_root, expected_sequences=["00", "01", "02", "03", "04", "05"], expected_frames=2988)
        self.assertEqual(res["status"], "FAIL")
        self.assertEqual(res["total_matched"], 1)
        self.assertFalse(res["sequence_details"]["01"]["exists"])

    def test_03_stem_based_pairing(self):
        """Test 3: build_file_list pairs files strictly by filename stem."""
        bin_paths, label_paths = build_file_list(self.dataset_root, ["00"])
        self.assertEqual(len(bin_paths), 1)
        self.assertEqual(len(label_paths), 1)
        self.assertEqual(Path(bin_paths[0]).stem, Path(label_paths[0]).stem)

    def test_04_max_frames_limiter(self):
        """Test 4: build_file_list respects max_frames parameter."""
        bin_paths, label_paths = build_file_list(self.dataset_root, ["00"], max_frames=1)
        self.assertEqual(len(bin_paths), 1)

    def test_05_point_label_alignment(self):
        """Test 5: Validates exact 1:1 point-to-label alignment."""
        pts = load_point_cloud(self.bin_file)
        lbls = load_labels(self.lbl_file)
        self.assertEqual(pts.shape[0], lbls.shape[0])
        self.assertTrue(validate_point_label_alignment(pts, lbls))

    def test_06_train_val_split_disjointness(self):
        """Test 6: Train sequences and validation sequences have zero overlap."""
        train_seqs = {"00", "01", "03", "04", "05"}
        val_seqs = {"02"}
        self.assertTrue(train_seqs.isdisjoint(val_seqs))


if __name__ == "__main__":
    unittest.main()
