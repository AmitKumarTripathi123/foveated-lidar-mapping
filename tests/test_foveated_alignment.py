"""Tests for Foveated Downsampling Point-Label Alignment (Phase 11 Part D1)."""

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
from ml.data.amit_adapter import FoveatedVoxelSampler, voxel_grid_downsample


class TestFoveatedAlignment(unittest.TestCase):
    """Test suite ensuring strict point-label alignment across all foveated downsampling operations."""

    @classmethod
    def setUpClass(cls):
        """Set up dataset paths."""
        cls.bin_file = repo_root / "dataset/sequences/00/velodyne/000000.bin"
        cls.lbl_file = repo_root / "dataset/sequences/00/labels/000000.label"

    def test_01_uniform_voxel_alignment(self):
        """Test 1: Uniform voxel downsampling preserves strict 1-to-1 point-label alignment."""
        pts = np.random.uniform(-10, 10, size=(1000, 4)).astype(np.float32)
        lbls = np.random.randint(0, 50, size=(1000,), dtype=np.uint32)
        down_pts, down_lbls = voxel_grid_downsample(pts, lbls, voxel_size=0.5)
        self.assertEqual(down_pts.shape[0], down_lbls.shape[0])
        self.assertTrue(validate_point_label_alignment(down_pts, down_lbls))

    def test_02_3zone_foveated_alignment_synthetic(self):
        """Test 2: 3-Zone foveated voxel sampler maintains point-label alignment on synthetic data."""
        pts = np.random.uniform(-60, 60, size=(5000, 4)).astype(np.float32)
        lbls = np.random.randint(0, 50, size=(5000,), dtype=np.uint32)
        sampler = FoveatedVoxelSampler()
        f_pts, f_lbls, rep = sampler.sample(pts, lbls)
        self.assertEqual(f_pts.shape[0], f_lbls.shape[0])
        self.assertTrue(rep.alignment_pass)
        self.assertTrue(validate_point_label_alignment(f_pts, f_lbls))

    def test_03_real_frame_foveated_alignment(self):
        """Test 3: Real frame 000000 foveated downsampling maintains 100% point-label alignment."""
        raw_pts = load_point_cloud(self.bin_file)
        raw_lbl = load_labels(self.lbl_file)
        v_pts, v_lbl, _ = filter_invalid_points(raw_pts, raw_lbl)

        sampler = FoveatedVoxelSampler()
        fov_pts, fov_lbl, rep = sampler.sample(v_pts, v_lbl)

        self.assertEqual(fov_pts.shape[0], fov_lbl.shape[0])
        self.assertEqual(fov_pts.shape[0], 50571)
        self.assertTrue(rep.alignment_pass)
        self.assertTrue(validate_point_label_alignment(fov_pts, fov_lbl))


if __name__ == "__main__":
    unittest.main()
