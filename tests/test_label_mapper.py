"""Unit tests for LabelMapper."""
import unittest
import numpy as np

from src.types import SuperClass
from src.label_mapper import LabelMapper


class TestLabelMapper(unittest.TestCase):
    def test_semantickitti_mapping(self):
        mapper = LabelMapper(mapping_config_path="configs/semantickitti_mapping.yaml")
        raw = np.array([40, 48, 50, 10, 30, 0, 999], dtype=np.uint32)
        mapped = mapper.map_labels(raw)

        self.assertEqual(mapped[0], SuperClass.DRIVABLE_TERRAIN)      # Road 40 -> 0
        self.assertEqual(mapped[1], SuperClass.NON_DRIVABLE_TERRAIN)  # Sidewalk 48 -> 1
        self.assertEqual(mapped[2], SuperClass.STATIC_OBSTACLE)       # Building 50 -> 2
        self.assertEqual(mapped[3], SuperClass.DYNAMIC_OBJECT)        # Car 10 -> 3
        self.assertEqual(mapped[4], SuperClass.DYNAMIC_OBJECT)        # Person 30 -> 3
        self.assertEqual(mapped[5], SuperClass.IGNORE_LABEL)          # Unlabeled 0 -> 255
        self.assertEqual(mapped[6], SuperClass.IGNORE_LABEL)          # Unknown 999 -> 255

    def test_semanticposs_incomplete_mapping_warning(self):
        mapper = LabelMapper(mapping_config_path="configs/semanticposs_mapping.yaml", dataset_type="semanticposs")
        self.assertTrue(mapper.has_mapping_warning)
        self.assertIn("non_drivable_terrain mapping is undefined/incomplete", mapper.mapping_warning_msg)

    def test_histogram_and_imbalance(self):
        mapper = LabelMapper()
        # 1000 road, 10 cars
        raw = np.concatenate([np.full(1000, 40, dtype=np.uint32), np.full(10, 10, dtype=np.uint32)])
        report = mapper.analyze_and_validate(raw)
        self.assertEqual(report.total_labeled_points, 1010)
        self.assertEqual(len(report.raw_label_histogram), 2)
        self.assertGreater(report.class_imbalance_ratio, 50.0)


if __name__ == "__main__":
    unittest.main()
