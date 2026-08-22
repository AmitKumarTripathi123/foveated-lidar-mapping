"""Unit tests for RangeFilter."""
import unittest
import numpy as np

from src.types import PointCloudFrame
from src.range_filter import RangeFilter


class TestRangeFilter(unittest.TestCase):
    def test_range_filter_100m(self):
        rf = RangeFilter(min_range=0.0, max_range=100.0)
        pts = np.array([
            [5.0, 0.0, 0.0, 0.5],     # r = 5 (keep)
            [50.0, 50.0, 0.0, 0.5],   # r = 70.7 (keep)
            [80.0, 60.0, 0.0, 0.5],   # r = 100 (keep)
            [80.0, 70.0, 0.0, 0.5],   # r = 106.3 (>100, remove)
            [np.nan, 0.0, 0.0, 0.5],  # NaN (remove)
        ], dtype=np.float32)
        lbls = np.array([0, 1, 2, 3, 0], dtype=np.uint32)

        frame = PointCloudFrame(points=pts, labels=lbls)
        filtered_frame, report = rf.filter_frame(frame)

        self.assertEqual(report.input_points, 5)
        self.assertEqual(report.removed_invalid_points, 1)
        self.assertEqual(report.removed_out_of_range_points, 1)
        self.assertEqual(report.output_points, 3)
        self.assertEqual(filtered_frame.num_points, 3)
        self.assertEqual(frame.num_points, 5)  # non-destructive


if __name__ == "__main__":
    unittest.main()
