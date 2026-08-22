"""Tests for Phase 11.2 Real Dataset Discovery, Inventory Auditing, and Verification."""

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


class TestPhase11_2RealDataset(unittest.TestCase):
    """Test suite verifying dataset discovery, frame pairing, and inventory integrity."""

    @classmethod
    def setUpClass(cls):
        """Set up class paths."""
        cls.dataset_root = repo_root / "dataset"
        cls.bin_file = repo_root / "dataset/sequences/00/velodyne/000000.bin"
        cls.lbl_file = repo_root / "dataset/sequences/00/labels/000000.label"

    def test_01_dataset_discovery_returns_records(self):
        """Test 1: Frame discovery finds physical scan pair."""
        records = discover_frames(self.dataset_root)
        self.assertGreaterEqual(len(records), 1)

    def test_02_frame_pairing_exact(self):
        """Test 2: Raw .bin and .label are paired without missing files."""
        records = discover_frames(self.dataset_root)
        for rec in records:
            self.assertTrue(rec.is_matched)
            self.assertTrue(Path(rec.point_cloud_path).is_file())
            self.assertTrue(Path(rec.label_path).is_file())

    def test_03_point_label_alignment(self):
        """Test 3: Raw points count exactly matches raw labels count (66,658)."""
        pts = load_point_cloud(self.bin_file)
        lbls = load_labels(self.lbl_file)
        self.assertEqual(pts.shape[0], lbls.shape[0])
        self.assertTrue(validate_point_label_alignment(pts, lbls))

    def test_04_raw_float32_finite(self):
        """Test 4: Raw points are finite float32 coordinates without NaNs or Infs."""
        pts = load_point_cloud(self.bin_file)
        self.assertEqual(pts.dtype, np.float32)
        self.assertFalse(np.isnan(pts).any())
        self.assertFalse(np.isinf(pts).any())


if __name__ == "__main__":
    unittest.main()
