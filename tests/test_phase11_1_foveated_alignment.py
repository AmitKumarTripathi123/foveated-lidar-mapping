"""Tests for Phase 11.1 Amit Foveated Pipeline Alignment."""

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


class TestPhase11_1FoveatedAlignment(unittest.TestCase):
    """Test suite ensuring 100% point-label alignment through Amit's foveated voxelizer."""

    @classmethod
    def setUpClass(cls):
        """Set up representative dataset scan."""
        cls.bin_file = repo_root / "dataset/sequences/00/velodyne/000000.bin"
        cls.lbl_file = repo_root / "dataset/sequences/00/labels/000000.label"

    def test_01_foveated_alignment_real(self):
        """Test 1: Foveated voxelization preserves exact point-label correspondence on real scan."""
        raw_pts = load_point_cloud(self.bin_file)
        raw_lbl = load_labels(self.lbl_file)
        v_pts, v_lbl, _ = filter_invalid_points(raw_pts, raw_lbl)

        sampler = FoveatedVoxelSampler()
        f_pts, f_lbl, rep = sampler.sample(v_pts, v_lbl)

        self.assertEqual(f_pts.shape[0], f_lbl.shape[0])
        self.assertEqual(f_pts.shape[0], 50571)
        self.assertTrue(rep.alignment_pass)
        self.assertTrue(validate_point_label_alignment(f_pts, f_lbl))

    def test_02_reduction_percentage(self):
        """Test 2: Foveated downsampler reduces raw points by > 20%."""
        raw_pts = load_point_cloud(self.bin_file)
        v_pts, _, _ = filter_invalid_points(raw_pts)
        sampler = FoveatedVoxelSampler()
        f_pts, _, rep = sampler.sample(v_pts)
        self.assertGreater(rep.overall_reduction_pct, 20.0)


if __name__ == "__main__":
    unittest.main()
