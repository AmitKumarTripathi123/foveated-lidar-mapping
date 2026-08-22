"""Tests for Phase 11.2 Amit Foveated Downsampling and Point-Label Alignment."""

import sys
import unittest
from pathlib import Path
import numpy as np

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.dataset import load_point_cloud, load_labels, validate_point_label_alignment
from ml.data.preprocessing import filter_invalid_points
from ml.data.amit_adapter import FoveatedVoxelSampler


class TestPhase11_2Foveation(unittest.TestCase):
    """Test suite verifying Amit's foveated voxelizer reduction and point-label alignment."""

    @classmethod
    def setUpClass(cls):
        """Set up class paths."""
        cls.bin_file = repo_root / "dataset/sequences/00/velodyne/000000.bin"
        cls.lbl_file = repo_root / "dataset/sequences/00/labels/000000.label"

    def test_01_foveated_alignment(self):
        """Test 1: Foveated voxelization preserves exact point-label correspondence."""
        raw_pts = load_point_cloud(self.bin_file)
        raw_lbl = load_labels(self.lbl_file)
        v_pts, v_lbl, _ = filter_invalid_points(raw_pts, raw_lbl)

        sampler = FoveatedVoxelSampler()
        fov_pts, fov_lbl, rep = sampler.sample(v_pts, v_lbl)

        self.assertEqual(fov_pts.shape[0], fov_lbl.shape[0])
        self.assertGreater(fov_pts.shape[0], 40000)
        self.assertLess(fov_pts.shape[0], raw_pts.shape[0])
        self.assertTrue(rep.alignment_pass)
        self.assertTrue(validate_point_label_alignment(fov_pts, fov_lbl))

    def test_02_point_reduction_measurement(self):
        """Test 2: Foveation achieves measurable point reduction (> 20%)."""
        raw_pts = load_point_cloud(self.bin_file)
        v_pts, _, _ = filter_invalid_points(raw_pts)
        sampler = FoveatedVoxelSampler()
        fov_pts, _, rep = sampler.sample(v_pts)
        self.assertGreater(rep.overall_reduction_pct, 20.0)


if __name__ == "__main__":
    unittest.main()
