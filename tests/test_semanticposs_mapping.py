"""Unit and Regression tests for SemanticPOSS label mapping."""
import unittest
import numpy as np

from src.types import SuperClass, PointCloudFrame
from src.label_mapper import LabelMapper


class TestSemanticPOSSMapping(unittest.TestCase):
    def setUp(self):
        self.mapper = LabelMapper(mapping_config_path="configs/semanticposs_mapping.yaml")

    def test_known_drivable_mapping(self):
        """Raw label 21 (ground/road) maps to SuperClass.DRIVABLE_TERRAIN (0)."""
        self.assertEqual(self.mapper.map_single_label(21), SuperClass.DRIVABLE_TERRAIN)

    def test_known_nondrivable_mapping(self):
        """Raw label 19 (terrain) maps to SuperClass.NON_DRIVABLE_TERRAIN (1)."""
        self.assertEqual(self.mapper.map_single_label(19), SuperClass.NON_DRIVABLE_TERRAIN)

    def test_known_static_obstacles(self):
        """Raw labels 9, 10, 11, 13, 14, 15, 16, 17, 18 map to STATIC_OBSTACLE (2)."""
        for raw_id in [9, 10, 11, 13, 14, 15, 16, 17, 18]:
            self.assertEqual(self.mapper.map_single_label(raw_id), SuperClass.STATIC_OBSTACLE)

    def test_known_dynamic_objects(self):
        """Raw labels 4 (person), 5 (two-wheelers), 6 (rider), 7 (car), 8 (other-vehicle) map to DYNAMIC_OBJECT (3)."""
        for raw_id in [4, 5, 6, 7, 8]:
            self.assertEqual(self.mapper.map_single_label(raw_id), SuperClass.DYNAMIC_OBJECT)

    def test_ignore_labels(self):
        """Raw labels 0 (unlabeled) and 22 (outlier) map to IGNORE_LABEL (255)."""
        self.assertEqual(self.mapper.map_single_label(0), SuperClass.IGNORE_LABEL)
        self.assertEqual(self.mapper.map_single_label(22), SuperClass.IGNORE_LABEL)

    def test_unknown_label_fallback(self):
        """Unmapped label falls back to IGNORE_LABEL (255)."""
        self.assertEqual(self.mapper.map_single_label(999), SuperClass.IGNORE_LABEL)

    def test_human_warning_flag(self):
        """Incomplete/provisional mapping issues human warning."""
        self.assertFalse(self.mapper.mapping_complete)
        self.assertIsNotNone(self.mapper.warning_message)


if __name__ == "__main__":
    unittest.main()
