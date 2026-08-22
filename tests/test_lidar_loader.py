"""Automated Test Suite for LiDAR Loader and Validation Pipeline (Phase 1).

Covers:
  - Test 1: Valid .bin loading
  - Test 2: Point cloud shape (N, 4)
  - Test 3: Point cloud dtype float32
  - Test 4: Valid .label loading
  - Test 5: Semantic label extraction via & 0xFFFF
  - Test 6: Point-label alignment verification
  - Test 7: Malformed point data rejection (non-divisible by 4)
  - Test 8: Point-label count mismatch raises LiDARAlignmentError
  - Test 9: NaN detection
  - Test 10: Inf detection
  - Test 11: Real/Verified sample scan verification
"""

import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.dataset import (
    load_point_cloud,
    load_labels,
    validate_point_label_alignment,
    validate_data_integrity,
    compute_point_cloud_stats,
    compute_label_distribution,
    validate_dataset_pair,
    LiDARFileNotFoundError,
    LiDARFormatError,
    LiDARAlignmentError,
)


class TestLiDARLoader(unittest.TestCase):
    """Unit and integration tests for LiDAR loader engine."""

    def setUp(self):
        """Set up temporary directory for synthetic test files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_01_valid_bin_file_loads(self):
        """Test 1: Valid .bin file loads successfully."""
        num_points = 100
        synthetic_points = np.random.randn(num_points, 4).astype(np.float32)
        bin_file = self.temp_path / "valid.bin"
        synthetic_points.tofile(bin_file)

        loaded = load_point_cloud(bin_file)
        self.assertIsInstance(loaded, np.ndarray)
        self.assertEqual(len(loaded), num_points)

    def test_02_shape_is_n_by_4(self):
        """Test 2: Point cloud shape is strictly (N, 4)."""
        num_points = 250
        synthetic_points = np.random.randn(num_points, 4).astype(np.float32)
        bin_file = self.temp_path / "shape_test.bin"
        synthetic_points.tofile(bin_file)

        loaded = load_point_cloud(bin_file)
        self.assertEqual(loaded.shape, (num_points, 4))

    def test_03_dtype_is_float32(self):
        """Test 3: Point cloud dtype is float32."""
        num_points = 50
        synthetic_points = np.random.randn(num_points, 4).astype(np.float32)
        bin_file = self.temp_path / "dtype_test.bin"
        synthetic_points.tofile(bin_file)

        loaded = load_point_cloud(bin_file)
        self.assertEqual(loaded.dtype, np.float32)

    def test_04_valid_label_loads(self):
        """Test 4: Valid .label file loads successfully."""
        num_points = 100
        synthetic_labels = np.random.randint(0, 50, size=num_points, dtype=np.uint32)
        label_file = self.temp_path / "valid.label"
        synthetic_labels.tofile(label_file)

        loaded = load_labels(label_file)
        self.assertIsInstance(loaded, np.ndarray)
        self.assertEqual(loaded.shape, (num_points,))

    def test_05_semantic_labels_extracted_with_mask(self):
        """Test 5: Semantic labels correctly extracted using raw_labels & 0xFFFF."""
        # Create labels with higher 16 bits (instance ID) and lower 16 bits (semantic ID)
        instance_ids = np.array([1, 2, 5, 10], dtype=np.uint32)
        semantic_ids = np.array([40, 10, 70, 48], dtype=np.uint32)
        raw_values = (instance_ids << 16) | semantic_ids

        label_file = self.temp_path / "masked.label"
        raw_values.tofile(label_file)

        extracted = load_labels(label_file)
        np.testing.assert_array_equal(extracted, semantic_ids)

    def test_06_point_label_alignment_passes(self):
        """Test 6: Point-label alignment passes when counts match."""
        num_points = 500
        points = np.zeros((num_points, 4), dtype=np.float32)
        labels = np.zeros(num_points, dtype=np.uint32)

        self.assertTrue(validate_point_label_alignment(points, labels))

    def test_07_malformed_point_data_rejected(self):
        """Test 7: Point file with invalid byte length (not divisible by 4 floats) raises LiDARFormatError."""
        # 10 floats = 2 points + 2 floats left over (not divisible by 4)
        malformed = np.random.randn(10).astype(np.float32)
        bin_file = self.temp_path / "malformed.bin"
        malformed.tofile(bin_file)

        with self.assertRaises(LiDARFormatError):
            load_point_cloud(bin_file)

    def test_08_point_label_mismatch_raises_error(self):
        """Test 8: Point-label count mismatch raises loud LiDARAlignmentError."""
        points = np.zeros((100, 4), dtype=np.float32)
        labels = np.zeros(90, dtype=np.uint32)

        with self.assertRaises(LiDARAlignmentError):
            validate_point_label_alignment(points, labels)

    def test_09_nan_detection(self):
        """Test 9: NaN values in points are detected properly."""
        clean_points = np.zeros((50, 4), dtype=np.float32)
        self.assertTrue(validate_data_integrity(clean_points)["valid"])

        corrupt_points = clean_points.copy()
        corrupt_points[10, 2] = np.nan
        res = validate_data_integrity(corrupt_points)
        self.assertTrue(res["has_nan"])
        self.assertFalse(res["valid"])

    def test_10_inf_detection(self):
        """Test 10: Inf values in points are detected properly."""
        clean_points = np.zeros((50, 4), dtype=np.float32)
        corrupt_points = clean_points.copy()
        corrupt_points[5, 0] = np.inf
        res = validate_data_integrity(corrupt_points)
        self.assertTrue(res["has_inf"])
        self.assertFalse(res["valid"])

    def test_11_real_sample_scan_verification(self):
        """Test 11: Real representative sample scan loads and passes all validation checks."""
        bin_path = repo_root / "dataset/sequences/00/velodyne/000000.bin"
        label_path = repo_root / "dataset/sequences/00/labels/000000.label"

        if not bin_path.is_file() or not label_path.is_file():
            self.skipTest("Sample files not present at default path.")

        report = validate_dataset_pair(bin_path, label_path)

        self.assertTrue(report.passed)
        self.assertEqual(report.points_shape, (66658, 4))
        self.assertEqual(report.labels_shape, (66658,))
        self.assertEqual(report.num_points, 66658)
        self.assertEqual(report.num_labels, 66658)
        self.assertEqual(report.point_dtype, "float32")
        self.assertTrue(report.alignment_pass)
        self.assertTrue(report.nan_check_pass)
        self.assertTrue(report.inf_check_pass)
        self.assertGreater(len(report.label_distribution), 0)


if __name__ == "__main__":
    unittest.main()
