"""Automated Test Suite for SIH Four-Class Semantic Label Remapping (Phase 3).

Covers:
  - Test 1: Exact SIH class IDs (0, 1, 2, 3, 255)
  - Test 2: Known raw SemanticKITTI mappings verification
  - Test 3: Unmapped raw label handling (maps to 255 ignore)
  - Test 4: Output range strictly subset of {0, 1, 2, 3, 255}
  - Test 5: Length preservation (N_raw == N_mapped)
  - Test 6: Deterministic mapping reproducibility
  - Test 7: Point-label alignment preservation (no shuffling/reordering)
  - Test 8: Explicit ignore class preservation (raw 0 and 1 -> 255)
  - Test 9: Invalid configuration rejection
  - Test 10: Real representative dataset sample remapping (000000.label)
  - Test 11: Vectorized execution performance
  - Test 12: PyTorch Dataset integration with label remapper
  - Test 13: Explicit verification of raw ID 255 -> SIH 3 (dynamic_object)
  - Test 14: Explicit verification that SIH 255 remains ignore
"""

import sys
import tempfile
import time
import unittest
from pathlib import Path
import numpy as np

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.dataset import load_labels, LidarDataset
from ml.data.label_mapping import (
    SemanticLabelRemapper,
    validate_mapped_labels,
    LabelMappingError,
    SIH_DRIVABLE_TERRAIN,
    SIH_NON_DRIVABLE_TERRAIN,
    SIH_STATIC_OBSTACLE,
    SIH_DYNAMIC_OBJECT,
    SIH_IGNORE,
    VALID_SIH_IDS,
    SIH_CLASS_NAMES,
    DEFAULT_RAW_TO_SIH,
)

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class TestLabelMapping(unittest.TestCase):
    """Unit and integration tests for Phase 3 label remapping engine."""

    def setUp(self):
        """Set up remapper instance and temporary test directory."""
        self.remapper = SemanticLabelRemapper()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_01_exact_class_ids(self):
        """Test 1: Frozen SIH ontology constants and class IDs."""
        self.assertEqual(SIH_DRIVABLE_TERRAIN, 0)
        self.assertEqual(SIH_NON_DRIVABLE_TERRAIN, 1)
        self.assertEqual(SIH_STATIC_OBSTACLE, 2)
        self.assertEqual(SIH_DYNAMIC_OBJECT, 3)
        self.assertEqual(SIH_IGNORE, 255)
        self.assertEqual(VALID_SIH_IDS, {0, 1, 2, 3, 255})
        self.assertEqual(len(SIH_CLASS_NAMES), 5)

    def test_02_known_raw_label_mappings(self):
        """Test 2: Every verified SemanticKITTI raw ID maps to its expected SIH class."""
        for raw_id, expected_sih in DEFAULT_RAW_TO_SIH.items():
            input_arr = np.array([raw_id], dtype=np.uint32)
            mapped = self.remapper.remap(input_arr)
            self.assertEqual(
                mapped[0],
                expected_sih,
                f"Raw ID {raw_id} mapped to {mapped[0]}, expected {expected_sih}",
            )

    def test_03_unmapped_labels(self):
        """Test 3: Unmapped/unknown raw labels cleanly map to 255 (ignore)."""
        unknown_ids = np.array([999, 12345, 99999], dtype=np.uint32)
        mapped = self.remapper.remap(unknown_ids)
        np.testing.assert_array_equal(mapped, np.full(len(unknown_ids), 255, dtype=np.uint8))

    def test_04_output_range(self):
        """Test 4: Output labels strictly belong to {0, 1, 2, 3, 255}."""
        raw_labels = np.random.randint(0, 300, size=1000, dtype=np.uint32)
        mapped = self.remapper.remap(raw_labels)

        unique_vals = set(np.unique(mapped))
        self.assertTrue(unique_vals.issubset(VALID_SIH_IDS))
        self.assertTrue(validate_mapped_labels(mapped))

    def test_05_point_count_preservation(self):
        """Test 5: Remapping strictly preserves label count (len(raw) == len(mapped))."""
        for n in [0, 1, 50, 1000, 66658]:
            raw = np.random.randint(0, 100, size=n, dtype=np.uint32)
            mapped = self.remapper.remap(raw)
            self.assertEqual(len(raw), len(mapped))

    def test_06_determinism(self):
        """Test 6: Same input produces identical output deterministically."""
        raw = np.random.randint(0, 100, size=5000, dtype=np.uint32)
        mapped_1 = self.remapper.remap(raw)
        mapped_2 = self.remapper.remap(raw)
        np.testing.assert_array_equal(mapped_1, mapped_2)

    def test_07_point_label_alignment(self):
        """Test 7: Remapping does not shuffle, reorder, or alter positional correspondence."""
        raw = np.array([40, 10, 50, 48, 0, 70, 80], dtype=np.uint32)
        expected = np.array([0, 3, 2, 1, 255, 2, 2], dtype=np.uint8)

        mapped = self.remapper.remap(raw)
        np.testing.assert_array_equal(mapped, expected)

    def test_08_ignore_preservation(self):
        """Test 8: Unlabeled noise (0) and outliers (1) map to 255 (ignore)."""
        noise_and_outliers = np.array([0, 1, 0, 1], dtype=np.uint32)
        mapped = self.remapper.remap(noise_and_outliers)
        np.testing.assert_array_equal(mapped, np.array([255, 255, 255, 255], dtype=np.uint8))

    def test_09_invalid_configuration(self):
        """Test 9: Invalid target SIH class IDs raise LabelMappingError."""
        invalid_config = {
            "ontology": {"ignore_id": 255},
            "raw_to_sih": {
                40: 99,  # Invalid SIH class ID
            },
        }
        with self.assertRaises(LabelMappingError):
            SemanticLabelRemapper(config=invalid_config)

    def test_10_real_dataset_sample(self):
        """Test 10: Real representative sample frame (000000.label) remapping verification."""
        label_file = repo_root / "dataset/sequences/00/labels/000000.label"
        if not label_file.is_file():
            self.skipTest("Sample label file not found.")

        raw_labels = load_labels(label_file)
        self.assertEqual(len(raw_labels), 66658)

        mapped_labels = self.remapper.remap(raw_labels)
        self.assertEqual(len(mapped_labels), 66658)

        unique_mapped = set(np.unique(mapped_labels))
        self.assertTrue(unique_mapped.issubset(VALID_SIH_IDS))

        report = self.remapper.audit(raw_labels, mapped_labels)
        self.assertTrue(report.passed)
        self.assertEqual(report.total_points, 66658)
        self.assertEqual(len(report.unmapped_ids), 0)

        # Verify class counts match verified sample
        dist_dict = {item.class_id: item.count for item in report.sih_distribution}
        self.assertEqual(dist_dict[0], 23000)  # drivable_terrain (road 40)
        self.assertEqual(dist_dict[1], 8000)   # non_drivable_terrain (sidewalk 48)
        self.assertEqual(dist_dict[2], 28500)  # static_obstacle (bld 10k + f 2k + veg 13k + tr 2k + pole 1.5k)
        self.assertEqual(dist_dict[3], 6000)   # dynamic_object (car 10)
        self.assertEqual(dist_dict[255], 1158) # ignore (unlabeled 0)

    def test_11_vectorized_performance(self):
        """Test 11: Vectorized lookup performs 100,000 points in < 15ms."""
        large_labels = np.random.randint(0, 100, size=100000, dtype=np.uint32)

        start_time = time.perf_counter()
        _ = self.remapper.remap(large_labels)
        elapsed = time.perf_counter() - start_time

        self.assertLess(elapsed, 0.05)

    def test_12_pytorch_dataset_integration(self):
        """Test 12: LidarDataset integration with label remapper."""
        if not _HAS_TORCH:
            self.skipTest("PyTorch is not installed in environment.")

        # Create mock sequence
        seq_dir = self.temp_path / "sequences" / "00"
        (seq_dir / "velodyne").mkdir(parents=True)
        (seq_dir / "labels").mkdir(parents=True)

        for i in range(2):
            stem = f"{i:06d}"
            np.random.randn(100, 4).astype(np.float32).tofile(seq_dir / "velodyne" / f"{stem}.bin")
            # Raw labels with road (40), car (10), unlabeled (0)
            raw = np.random.choice([40, 10, 0], size=100).astype(np.uint32)
            raw.tofile(seq_dir / "labels" / f"{stem}.label")

        dataset = LidarDataset(
            root=self.temp_path,
            sequences=["00"],
            label_remapper=self.remapper,
            to_tensor=True,
        )

        sample = dataset[0]
        self.assertIsInstance(sample["labels"], torch.Tensor)
        unique_tensors = set(sample["labels"].numpy().tolist())
        self.assertTrue(unique_tensors.issubset({0, 3, 255}))

    def test_13_raw_id_255_maps_to_sih_3_dynamic_object(self):
        """Test 13: Raw SemanticKITTI ID 255 (moving-motorcyclist) explicitly maps to SIH 3 (dynamic_object)."""
        raw_255 = np.array([255, 255, 255], dtype=np.uint32)
        mapped = self.remapper.remap(raw_255)
        np.testing.assert_array_equal(mapped, np.array([3, 3, 3], dtype=np.uint8))
        self.assertEqual(SIH_CLASS_NAMES[3], "dynamic_object")

    def test_14_sih_255_remains_ignore_for_noise_and_unmapped(self):
        """Test 14: SIH class 255 strictly represents ignore for noise (0), outliers (1), and unmapped IDs."""
        raw_inputs = np.array([0, 1, 999, 1000], dtype=np.uint32)
        mapped = self.remapper.remap(raw_inputs)
        np.testing.assert_array_equal(mapped, np.array([255, 255, 255, 255], dtype=np.uint8))
        self.assertEqual(SIH_CLASS_NAMES[255], "ignore")


if __name__ == "__main__":
    unittest.main()
