"""Unit tests for Phase2Dataset."""
import unittest
import torch
import numpy as np
from pathlib import Path

from phase2.dataset import Phase2Dataset, remap_poss_labels, SEMANTICPOSS_TO_PROJECT
from src.types import SuperClass


class TestPhase2Dataset(unittest.TestCase):
    def test_single_authoritative_mapping(self):
        self.assertEqual(SEMANTICPOSS_TO_PROJECT[21], SuperClass.DRIVABLE_TERRAIN)
        self.assertEqual(SEMANTICPOSS_TO_PROJECT[20], SuperClass.NON_DRIVABLE_TERRAIN)
        self.assertEqual(SEMANTICPOSS_TO_PROJECT[19], SuperClass.NON_DRIVABLE_TERRAIN)
        self.assertEqual(SEMANTICPOSS_TO_PROJECT[22], SuperClass.IGNORE_LABEL)
        self.assertEqual(SEMANTICPOSS_TO_PROJECT[4], SuperClass.DYNAMIC_OBJECT)
        self.assertEqual(SEMANTICPOSS_TO_PROJECT[9], SuperClass.STATIC_OBSTACLE)

    def test_remap_poss_labels_vectorized(self):
        raw_arr = np.array([21, 20, 19, 22, 4, 9, 0, 1], dtype=np.uint32)
        remapped = remap_poss_labels(raw_arr)
        expected = np.array([0, 1, 1, 255, 3, 2, 255, 255], dtype=np.int64)
        np.testing.assert_array_equal(remapped, expected)

    def test_dataset_loading(self):
        ds = Phase2Dataset(dataset_root="data/semanticposs_sequence", sequences=["01"], split="train")
        if len(ds) > 0:
            sample = ds[0]
            self.assertIn("points", sample)
            self.assertIn("labels", sample)
            self.assertEqual(sample["points"].shape[1], 4)
            self.assertEqual(sample["points"].shape[0], sample["labels"].shape[0])
            self.assertTrue((sample["labels"] >= 0).all())


if __name__ == "__main__":
    unittest.main()
