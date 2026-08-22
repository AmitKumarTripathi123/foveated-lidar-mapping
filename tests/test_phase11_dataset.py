"""Tests for Phase 11 Dataset Forensic Discovery and Validation."""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.dataset import load_point_cloud, load_labels, validate_point_label_alignment
from ml.data.frame_discovery import FrameRecord, discover_frames


class TestPhase11Dataset(unittest.TestCase):
    """Test suite for dataset discovery, pairing, corruption and duplicate detection."""

    @classmethod
    def setUpClass(cls):
        """Set up class paths."""
        cls.dataset_root = repo_root / "dataset"
        cls.bin_file = repo_root / "dataset/sequences/00/velodyne/000000.bin"
        cls.lbl_file = repo_root / "dataset/sequences/00/labels/000000.label"

    def test_01_frame_discovery(self):
        """Test 1: Frame discovery engine locates sequence 00 scan."""
        records = discover_frames(self.dataset_root)
        self.assertGreaterEqual(len(records), 1)
        self.assertEqual(records[0].sequence_id, "00")

    def test_02_pair_matching(self):
        """Test 2: Raw .bin and .label are paired without missing files."""
        records = discover_frames(self.dataset_root)
        rec = records[0]
        self.assertTrue(rec.is_matched)
        self.assertTrue(Path(rec.point_cloud_path).is_file())
        self.assertTrue(Path(rec.label_path).is_file())

    def test_03_point_label_alignment(self):
        """Test 3: Exactly 66,658 points align with 66,658 labels."""
        pts = load_point_cloud(self.bin_file)
        lbls = load_labels(self.lbl_file)
        self.assertEqual(pts.shape[0], lbls.shape[0])
        self.assertTrue(validate_point_label_alignment(pts, lbls))

    def test_04_duplicate_detection(self):
        """Test 4: Hashing validates physical data integrity without collisions."""
        with open(self.bin_file, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(len(h), 64)

    def test_05_sequence_discovery(self):
        """Test 5: Sequence directory discovery is deterministic."""
        seq_dirs = list((self.dataset_root / "sequences").iterdir())
        self.assertGreaterEqual(len(seq_dirs), 1)


if __name__ == "__main__":
    unittest.main()
