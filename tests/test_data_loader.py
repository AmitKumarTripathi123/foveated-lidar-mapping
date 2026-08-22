"""Unit tests for LiDARDataLoader."""
import unittest
from pathlib import Path
import numpy as np

from src.types import PointCloudFrame, ValidationPolicy
from src.data_loader import LiDARDataLoader


class TestLiDARDataLoader(unittest.TestCase):
    def setUp(self):
        self.dataset_path = Path("data/synthetic_sequence")
        self.loader = LiDARDataLoader(
            dataset_path=self.dataset_path,
            sequence_id="00",
            validation_policy=ValidationPolicy.SKIP_AND_WARN
        )

    def test_discover_frames(self):
        frames = self.loader.discover_frames()
        self.assertGreaterEqual(len(frames), 5)
        scan_p, label_p = frames[0]
        self.assertTrue(scan_p.exists())
        self.assertTrue(label_p.exists())

    def test_load_valid_frame(self):
        frames = self.loader.discover_frames()
        scan_p, label_p = frames[0]
        frame = self.loader.load_frame(scan_p, label_p)
        self.assertTrue(frame.is_valid)
        self.assertGreater(frame.num_points, 0)
        self.assertEqual(frame.points.shape[1], 4)
        self.assertEqual(len(frame.points), len(frame.labels))

    def test_mismatched_label_handling(self):
        corrupt_loader = LiDARDataLoader(
            dataset_path=self.dataset_path,
            sequence_id="99",
            validation_policy=ValidationPolicy.SKIP_AND_WARN
        )
        frames = corrupt_loader.discover_frames()
        self.assertGreaterEqual(len(frames), 1)
        scan_p, label_p = frames[0]
        frame = corrupt_loader.load_frame(scan_p, label_p)
        self.assertFalse(frame.is_valid)
        self.assertGreater(len(corrupt_loader.invalid_frames), 0)

    def test_strict_stop_policy(self):
        strict_loader = LiDARDataLoader(
            dataset_path=self.dataset_path,
            sequence_id="99",
            validation_policy=ValidationPolicy.STRICT_STOP
        )
        frames = strict_loader.discover_frames()
        scan_p, label_p = frames[0]
        with self.assertRaises(ValueError):
            strict_loader.load_frame(scan_p, label_p)


if __name__ == "__main__":
    unittest.main()
