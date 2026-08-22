"""Tests for Phase 11.1 Real Dataset Discovery, Inventory, and Pairing Validation."""

import hashlib
import sys
import unittest
from pathlib import Path
import numpy as np

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.dataset import load_point_cloud, load_labels, validate_point_label_alignment
from ml.data.frame_discovery import FrameRecord, discover_frames


class TestPhase11_1RealDataset(unittest.TestCase):
    """Test suite for dataset discovery, frame pairing, and inventory auditing."""

    @classmethod
    def setUpClass(cls):
        """Set up dataset paths."""
        cls.dataset_root = repo_root / "dataset"
        cls.bin_file = repo_root / "dataset/sequences/00/velodyne/000000.bin"
        cls.lbl_file = repo_root / "dataset/sequences/00/labels/000000.label"

    def test_01_frame_discovery(self):
        """Test 1: Frame discovery finds at least 1 valid scan pair."""
        records = discover_frames(self.dataset_root)
        self.assertGreaterEqual(len(records), 1)

    def test_02_bin_label_pairing(self):
        """Test 2: Discovered frame records have matched .bin and .label paths."""
        records = discover_frames(self.dataset_root)
        for rec in records:
            self.assertTrue(rec.is_matched)
            self.assertTrue(Path(rec.point_cloud_path).is_file())
            self.assertTrue(Path(rec.label_path).is_file())

    def test_03_raw_data_validity(self):
        """Test 3: Raw scan points are float32 (N, 4) and labels are uint32 (N,)."""
        pts = load_point_cloud(self.bin_file)
        lbls = load_labels(self.lbl_file)
        self.assertEqual(pts.ndim, 2)
        self.assertEqual(pts.shape[1], 4)
        self.assertEqual(lbls.ndim, 1)
        self.assertEqual(pts.shape[0], lbls.shape[0])
        self.assertFalse(np.isnan(pts).any())
        self.assertFalse(np.isinf(pts).any())

    def test_04_point_label_alignment(self):
        """Test 4: Raw points count exactly equals raw labels count."""
        pts = load_point_cloud(self.bin_file)
        lbls = load_labels(self.lbl_file)
        self.assertTrue(validate_point_label_alignment(pts, lbls))


if __name__ == "__main__":
    unittest.main()
