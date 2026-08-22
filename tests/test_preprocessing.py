"""Automated Test Suite for LiDAR Preprocessing and PyTorch Dataset Pipeline (Phase 2).

Covers:
  - Test 1: Identity / keep-all preservation
  - Test 2: Invalid point removal (NaN/Inf) with shared boolean mask
  - Test 3: Range filtering with exact bounding box
  - Test 4: Random sampling point-count verification
  - Test 5: Sampling alignment correspondence check
  - Test 6: Deterministic reproducibility with seed
  - Test 7: Different seeds producing different samples
  - Test 8: Invalid sampling request (num_points > N)
  - Test 9: Label integer classification-compatible dtype
  - Test 10: Dataset lazy indexing across sequences
  - Test 11: Missing label file handling
  - Test 12: Missing point cloud file handling
  - Test 13: PyTorch Dataset & DataLoader collation
  - Test 14: Real representative frame preprocessing (keep-all & 16,384 sampled)
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
    LidarDataset,
    lidar_collate_fn,
    LiDARFileNotFoundError,
    LiDARAlignmentError,
)
from ml.data.preprocessing import (
    LidarPreprocessor,
    PreprocessingConfig,
    SamplingConfig,
    InvalidPointsConfig,
    RangeFilterConfig,
    CoordinatesConfig,
    IntensityConfig,
    filter_invalid_points,
    apply_range_filter,
    sample_points,
    handle_coordinates,
    handle_intensity,
    PreprocessingError,
)

try:
    import torch
    from torch.utils.data import DataLoader
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class TestPreprocessing(unittest.TestCase):
    """Unit and integration tests for Phase 2 Preprocessing pipeline."""

    def setUp(self):
        """Set up temporary directory for synthetic datasets."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_01_identity_keep_all(self):
        """Test 1: keep_all preserves original (N, 4) points and (N,) labels."""
        num_points = 500
        points = np.random.randn(num_points, 4).astype(np.float32)
        labels = np.random.randint(0, 50, size=num_points, dtype=np.uint32)

        preprocessor = LidarPreprocessor(PreprocessingConfig())
        processed = preprocessor(points, labels)

        self.assertEqual(processed.points.shape, (num_points, 4))
        self.assertEqual(processed.labels.shape, (num_points,))
        np.testing.assert_array_equal(processed.points, points)
        np.testing.assert_array_equal(processed.labels, labels)
        self.assertTrue(processed.report.alignment_pass)

    def test_02_invalid_point_removal(self):
        """Test 2: NaN/Inf points and corresponding labels removed with shared mask."""
        num_points = 100
        points = np.ones((num_points, 4), dtype=np.float32) * 10.0
        labels = np.arange(num_points, dtype=np.uint32)

        # Inject NaNs and Infs at specific indices
        points[10, 0] = np.nan
        points[25, 2] = np.inf
        points[50, 3] = -np.inf

        filtered_points, filtered_labels, num_removed = filter_invalid_points(points, labels)

        self.assertEqual(num_removed, 3)
        self.assertEqual(filtered_points.shape, (97, 4))
        self.assertEqual(filtered_labels.shape, (97,))

        # Verify that invalid indices were removed from labels as well
        self.assertNotIn(10, filtered_labels)
        self.assertNotIn(25, filtered_labels)
        self.assertNotIn(50, filtered_labels)
        self.assertTrue(validate_point_label_alignment(filtered_points, filtered_labels))

    def test_03_range_filtering(self):
        """Test 3: Spatial range filter extracts exact bounding box."""
        # Points from x = -50 to +50
        x = np.linspace(-50, 50, 101, dtype=np.float32)
        y = np.zeros(101, dtype=np.float32)
        z = np.zeros(101, dtype=np.float32)
        intensity = np.ones(101, dtype=np.float32) * 0.5

        points = np.column_stack([x, y, z, intensity])
        labels = np.arange(101, dtype=np.uint32)

        # Filter range: x in [0.0, 20.0] -> values 0, 1, 2, ..., 20 (21 points)
        filtered_points, filtered_labels, num_filtered = apply_range_filter(
            points, labels, min_x=0.0, max_x=20.0
        )

        self.assertEqual(len(filtered_points), 21)
        self.assertEqual(len(filtered_labels), 21)
        self.assertTrue((filtered_points[:, 0] >= 0.0).all())
        self.assertTrue((filtered_points[:, 0] <= 20.0).all())
        self.assertTrue(validate_point_label_alignment(filtered_points, filtered_labels))

    def test_04_random_sampling_count(self):
        """Test 4: Random sampling produces exact requested point count."""
        num_points = 1000
        points = np.random.randn(num_points, 4).astype(np.float32)
        labels = np.random.randint(0, 50, size=num_points, dtype=np.uint32)

        target_count = 256
        sampled_points, sampled_labels, meta = sample_points(
            points, labels, num_points=target_count, strategy="random", seed=42
        )

        self.assertEqual(sampled_points.shape, (target_count, 4))
        self.assertEqual(sampled_labels.shape, (target_count,))
        self.assertTrue(validate_point_label_alignment(sampled_points, sampled_labels))

    def test_05_sampling_alignment_correspondence(self):
        """Test 5: Sampling retains exact correspondence between point coordinates and labels."""
        num_points = 100
        # Point i has coordinate [i, i, i, i] and label i * 10
        points = np.repeat(np.arange(num_points, dtype=np.float32)[:, None], 4, axis=1)
        labels = np.arange(num_points, dtype=np.uint32) * 10

        sampled_points, sampled_labels, _ = sample_points(
            points, labels, num_points=30, strategy="random", seed=123
        )

        for i in range(len(sampled_points)):
            expected_val = sampled_points[i, 0]
            expected_label = int(expected_val * 10)
            self.assertEqual(sampled_labels[i], expected_label)

    def test_06_reproducibility(self):
        """Test 6: Same input + config + seed produces identical output."""
        points = np.random.randn(500, 4).astype(np.float32)
        labels = np.random.randint(0, 50, size=500, dtype=np.uint32)

        p1, l1, _ = sample_points(points, labels, num_points=128, strategy="random", seed=999)
        p2, l2, _ = sample_points(points, labels, num_points=128, strategy="random", seed=999)

        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(l1, l2)

    def test_07_different_seeds_produce_different_samples(self):
        """Test 7: Different random seeds produce distinct selections."""
        points = np.random.randn(500, 4).astype(np.float32)
        labels = np.random.randint(0, 50, size=500, dtype=np.uint32)

        p1, _, _ = sample_points(points, labels, num_points=128, strategy="random", seed=42)
        p2, _, _ = sample_points(points, labels, num_points=128, strategy="random", seed=43)

        self.assertFalse(np.array_equal(p1, p2))

    def test_08_invalid_sampling_request(self):
        """Test 8: num_points > N without replacement raises PreprocessingError."""
        points = np.random.randn(50, 4).astype(np.float32)
        labels = np.random.randint(0, 50, size=50, dtype=np.uint32)

        with self.assertRaises(PreprocessingError):
            sample_points(points, labels, num_points=100, strategy="random", seed=42)

    def test_09_label_dtype_integrity(self):
        """Test 9: Labels maintain integer classification-compatible dtype."""
        points = np.random.randn(100, 4).astype(np.float32)
        labels = np.random.randint(0, 50, size=100, dtype=np.uint32)

        preprocessor = LidarPreprocessor(
            PreprocessingConfig(
                sampling=SamplingConfig(strategy="random", num_points=50, seed=42)
            )
        )
        processed = preprocessor(points, labels)

        self.assertTrue(np.issubdtype(processed.labels.dtype, np.integer))

    def test_10_dataset_lazy_indexing(self):
        """Test 10: LidarDataset correctly discovers and indexes sequence scan pairs."""
        # Create mock sequence directory structure
        seq_dir = self.temp_path / "sequences" / "00"
        (seq_dir / "velodyne").mkdir(parents=True)
        (seq_dir / "labels").mkdir(parents=True)

        for i in range(3):
            stem = f"{i:06d}"
            np.zeros((50, 4), dtype=np.float32).tofile(seq_dir / "velodyne" / f"{stem}.bin")
            np.zeros(50, dtype=np.uint32).tofile(seq_dir / "labels" / f"{stem}.label")

        dataset = LidarDataset(root=self.temp_path, sequences=["00"])
        self.assertEqual(len(dataset), 3)

        sample = dataset[0]
        self.assertEqual(sample["points"].shape, (50, 4))
        self.assertEqual(sample["labels"].shape, (50,))
        self.assertEqual(sample["metadata"]["frame"], "000000")

    def test_11_missing_label_file(self):
        """Test 11: Missing label file is handled clearly."""
        non_existent = self.temp_path / "missing.label"
        with self.assertRaises(LiDARFileNotFoundError):
            load_labels(non_existent)

    def test_12_missing_point_cloud_file(self):
        """Test 12: Missing point cloud file is handled clearly."""
        non_existent = self.temp_path / "missing.bin"
        with self.assertRaises(LiDARFileNotFoundError):
            load_point_cloud(non_existent)

    def test_13_pytorch_dataset_and_dataloader(self):
        """Test 13: PyTorch Dataset conversion and DataLoader batch collation."""
        if not _HAS_TORCH:
            self.skipTest("PyTorch is not installed in environment.")

        # Create mock sequence
        seq_dir = self.temp_path / "sequences" / "00"
        (seq_dir / "velodyne").mkdir(parents=True)
        (seq_dir / "labels").mkdir(parents=True)

        for i in range(4):
            stem = f"{i:06d}"
            np.random.randn(200, 4).astype(np.float32).tofile(seq_dir / "velodyne" / f"{stem}.bin")
            np.random.randint(0, 20, size=200, dtype=np.uint32).tofile(seq_dir / "labels" / f"{stem}.label")

        preprocessor = LidarPreprocessor(
            PreprocessingConfig(
                sampling=SamplingConfig(strategy="random", num_points=64, seed=42)
            )
        )

        dataset = LidarDataset(
            root=self.temp_path,
            sequences=["00"],
            preprocessor=preprocessor,
            to_tensor=True,
        )

        self.assertEqual(len(dataset), 4)

        sample = dataset[0]
        self.assertIsInstance(sample["points"], torch.Tensor)
        self.assertEqual(sample["points"].shape, (64, 4))
        self.assertEqual(sample["labels"].shape, (64,))
        self.assertEqual(sample["points"].dtype, torch.float32)
        self.assertEqual(sample["labels"].dtype, torch.int64)

        # DataLoader batch collation
        loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=lidar_collate_fn)
        batch = next(iter(loader))

        self.assertEqual(batch["points"].shape, (2, 64, 4))
        self.assertEqual(batch["labels"].shape, (2, 64))

    def test_14_real_sample_scan_preprocessing(self):
        """Test 14: Real representative sample scan preprocessing (keep-all & 16,384 sampled)."""
        bin_path = repo_root / "dataset/sequences/00/velodyne/000000.bin"
        label_path = repo_root / "dataset/sequences/00/labels/000000.label"

        if not bin_path.is_file() or not label_path.is_file():
            self.skipTest("Sample files not present at default path.")

        points = load_point_cloud(bin_path)
        labels = load_labels(label_path)

        # 1. Keep-all test
        prep_keep = LidarPreprocessor(PreprocessingConfig())
        out_keep = prep_keep(points, labels)

        self.assertEqual(out_keep.points.shape, (66658, 4))
        self.assertEqual(out_keep.labels.shape, (66658,))
        self.assertTrue(out_keep.report.alignment_pass)

        # 2. Controlled 16,384 sampling test
        config_sampled = PreprocessingConfig(
            sampling=SamplingConfig(strategy="random", num_points=16384, seed=42)
        )
        prep_sampled = LidarPreprocessor(config_sampled)
        out_sampled = prep_sampled(points, labels)

        self.assertEqual(out_sampled.points.shape, (16384, 4))
        self.assertEqual(out_sampled.labels.shape, (16384,))
        self.assertTrue(out_sampled.report.alignment_pass)


if __name__ == "__main__":
    unittest.main()
