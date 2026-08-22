"""Unit tests for PointCloudValidator."""
import unittest
import numpy as np

from src.types import PointCloudFrame
from src.validator import PointCloudValidator


class TestPointCloudValidator(unittest.TestCase):
    def setUp(self):
        self.validator = PointCloudValidator(max_allowed_range=100.0)

    def test_coordinate_validity_clean(self):
        pts = np.random.uniform(-10, 10, size=(100, 4)).astype(np.float32)
        res = self.validator.validate_coordinates(pts)
        self.assertTrue(res.is_clean)
        self.assertEqual(res.invalid_point_count, 0)
        self.assertEqual(res.invalid_point_percentage, 0.0)

    def test_coordinate_validity_nan_inf(self):
        pts = np.random.uniform(-10, 10, size=(100, 4)).astype(np.float32)
        pts[0, 0] = np.nan
        pts[1, 1] = np.inf
        pts[2, 2] = -np.inf
        res = self.validator.validate_coordinates(pts)
        self.assertFalse(res.is_clean)
        self.assertEqual(res.nan_count, 1)
        self.assertEqual(res.pos_inf_count, 1)
        self.assertEqual(res.neg_inf_count, 1)
        self.assertEqual(res.invalid_point_count, 3)
        self.assertEqual(res.invalid_point_percentage, 3.0)

    def test_range_validation(self):
        pts = np.array([
            [3.0, 4.0, 0.0, 0.5],    # r = 5
            [6.0, 8.0, 0.0, 0.5],    # r = 10
            [60.0, 80.0, 0.0, 0.5],  # r = 100
            [120.0, 50.0, 0.0, 0.5]  # r = 130 (>100m)
        ], dtype=np.float32)
        res = self.validator.validate_ranges(pts)
        self.assertEqual(res.min_range, 5.0)
        self.assertEqual(res.max_range, 130.0)
        self.assertEqual(res.points_within_100m, 3)
        self.assertEqual(res.points_beyond_100m, 1)

    def test_intensity_validation_detection(self):
        pts_norm = np.column_stack([np.zeros((10, 3)), np.linspace(0, 1, 10)])
        res_norm = self.validator.validate_intensity(pts_norm)
        self.assertEqual(res_norm.detected_format, "normalized_0_1")
        self.assertTrue(res_norm.is_normalized)

        pts_255 = np.column_stack([np.zeros((10, 3)), np.array([0, 50, 100, 255, 20, 30, 40, 50, 60, 70])])
        res_255 = self.validator.validate_intensity(pts_255)
        self.assertEqual(res_255.detected_format, "integer_0_255")
        self.assertFalse(res_255.is_normalized)

        # Test non-destructive normalization
        norm_pts = self.validator.normalize_intensity(pts_255, detected_format="integer_0_255")
        self.assertAlmostEqual(float(np.max(norm_pts[:, 3])), 1.0)
        self.assertEqual(pts_255[3, 3], 255)  # original untouched

    def test_coordinate_distribution(self):
        pts = np.array([
            [10.0, -2.0, -1.7, 0.5],
            [20.0, 2.0, -1.7, 0.5],
            [-5.0, 0.0, -1.7, 0.5]
        ], dtype=np.float32)
        res = self.validator.validate_coordinate_distribution(pts)
        self.assertIn("Human Confirmation Required", res.coordinate_convention_status)
        self.assertAlmostEqual(res.forward_x_percentage, 66.67, places=1)


if __name__ == "__main__":
    unittest.main()
