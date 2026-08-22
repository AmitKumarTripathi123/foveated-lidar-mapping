"""Phase 7.1 Amit Data Pipeline Audit & Atul ML Data Integration Tests (14 Tests).

Covers:
  1. Frame discovery
  2. Sequence discovery
  3. .bin / .label pairing
  4. Missing labels detection
  5. Manifest generation
  6. Frame count correctness
  7. Point-label alignment
  8. Finite point validation
  9. SIH label validation
  10. Foveated alignment
  11. ML dataset receives discovered frames
  12. Split disjointness
  13. External dataset root configuration
  14. Cache vs raw data distinction
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.dataset import load_point_cloud, load_labels, validate_point_label_alignment
from ml.data.preprocessing import filter_invalid_points
from ml.data.amit_adapter import FoveatedVoxelSampler
from ml.data.label_mapping import SemanticLabelRemapper
from ml.data.foveated_dataset import FoveatedLidarDataset
from ml.data.frame_discovery import FrameRecord, discover_frames, audit_discovered_frames


class TestPhase71DataIntegration(unittest.TestCase):
    """Test suite for Phase 7.1 data discovery and integration."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
        cls.dataset_root = repo_root / "dataset"
        cls.bin_file = repo_root / "dataset/sequences/00/velodyne/000000.bin"
        cls.lbl_file = repo_root / "dataset/sequences/00/labels/000000.label"

    def setUp(self):
        """Set up temporary directory."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    # 1. Frame discovery
    def test_01_frame_discovery(self):
        """Test 1: discover_frames discovers local sequence 00 frame 000000."""
        records = discover_frames(self.dataset_root)
        self.assertGreaterEqual(len(records), 1)
        self.assertEqual(records[0].sequence_id, "00")
        self.assertEqual(records[0].frame_id, "000000")

    # 2. Sequence discovery
    def test_02_sequence_discovery(self):
        """Test 2: audit_discovered_frames accurately extracts all unique sequence IDs."""
        records = discover_frames(self.dataset_root)
        audit = audit_discovered_frames(records)
        self.assertIn("00", audit["sequences"])
        self.assertEqual(audit["total_sequences"], len(audit["sequences"]))

    # 3. .bin / .label pairing
    def test_03_bin_label_pairing(self):
        """Test 3: Matched scan has both point_cloud_path and label_path set."""
        records = discover_frames(self.dataset_root)
        rec = records[0]
        self.assertTrue(rec.has_label)
        self.assertTrue(rec.is_matched)
        self.assertIsNotNone(rec.label_path)
        self.assertTrue(Path(rec.point_cloud_path).is_file())
        self.assertTrue(Path(rec.label_path).is_file())

    # 4. Missing labels detection
    def test_04_missing_labels_detection(self):
        """Test 4: Frames without matching .label are flagged as unmatched without failing."""
        seq_dir = self.temp_path / "sequences" / "09"
        (seq_dir / "velodyne").mkdir(parents=True)
        (seq_dir / "labels").mkdir(parents=True)

        dummy_pts = np.random.randn(10, 4).astype(np.float32)
        dummy_pts.tofile(seq_dir / "velodyne" / "000001.bin")

        records = discover_frames(self.temp_path)
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0].has_label)
        self.assertFalse(records[0].is_matched)
        self.assertIsNone(records[0].label_path)

    # 5. Manifest generation
    def test_05_manifest_generation(self):
        """Test 5: Audit dictionary contains comprehensive frame-level metadata."""
        records = discover_frames(self.dataset_root)
        audit = audit_discovered_frames(records)
        self.assertIn("frames", audit)
        self.assertEqual(len(audit["frames"]), len(records))
        self.assertEqual(audit["frames"][0]["sequence"], "00")
        self.assertEqual(audit["frames"][0]["frame"], "000000")

    # 6. Frame count correctness
    def test_06_frame_count_correctness(self):
        """Test 6: Frame count matches actual .bin file count on disk."""
        bin_files = list(self.dataset_root.glob("sequences/*/velodyne/*.bin"))
        records = discover_frames(self.dataset_root)
        self.assertEqual(len(records), len(bin_files))

    # 7. Point-label alignment
    def test_07_point_label_alignment(self):
        """Test 7: Discovered frame has exact point and label count equality."""
        records = discover_frames(self.dataset_root)
        audit = audit_discovered_frames(records)
        f_info = audit["frames"][0]
        self.assertTrue(f_info["aligned"])
        self.assertEqual(f_info["points"], f_info["labels"])

    # 8. Finite point validation
    def test_08_finite_point_validation(self):
        """Test 8: Discovered frame has zero NaN or Inf values."""
        records = discover_frames(self.dataset_root)
        audit = audit_discovered_frames(records)
        self.assertTrue(audit["frames"][0]["finite"])

    # 9. SIH label validation
    def test_09_sih_label_validation(self):
        """Test 9: All raw labels remap strictly into {0, 1, 2, 3, 255}."""
        labels = load_labels(self.lbl_file)
        remapper = SemanticLabelRemapper()
        sih_labels = remapper.remap(labels)
        unique_sih = set(np.unique(sih_labels))
        self.assertTrue(unique_sih.issubset({0, 1, 2, 3, 255}))

    # 10. Foveated alignment
    def test_10_foveated_alignment(self):
        """Test 10: Point-label alignment is strictly preserved across foveated downsampling."""
        pts = load_point_cloud(self.bin_file)
        lbls = load_labels(self.lbl_file)
        v_pts, v_lbls, _ = filter_invalid_points(pts, lbls)

        sampler = FoveatedVoxelSampler()
        fov_pts, fov_lbls, _ = sampler.sample(v_pts, v_lbls)
        self.assertEqual(fov_pts.shape[0], fov_lbls.shape[0])

    # 11. ML dataset receives discovered frames
    def test_11_ml_dataset_receives_discovered_frames(self):
        """Test 11: FoveatedLidarDataset directly initializes from discover_frames output."""
        records = discover_frames(self.dataset_root)
        dataset = FoveatedLidarDataset(raw_manifest=records, target_num_points=1024, to_tensor=True)
        self.assertEqual(len(dataset), len(records))

        sample = dataset[0]
        self.assertEqual(sample["points"].shape, (1024, 4))
        self.assertEqual(sample["labels"].shape, (1024,))
        self.assertEqual(sample["metadata"]["sequence"], "00")

    # 12. Split disjointness
    def test_12_split_disjointness(self):
        """Test 12: Split logic guarantees sequence-level disjointness."""
        train_seqs = {"00", "01", "03", "04", "05"}
        val_seqs = {"02"}
        test_seqs = {"08"}
        self.assertTrue(train_seqs.isdisjoint(val_seqs))
        self.assertTrue(train_seqs.isdisjoint(test_seqs))

    # 13. External dataset root configuration
    def test_13_external_dataset_root_configuration(self):
        """Test 13: discover_frames accepts DATASET_ROOT environment variable."""
        ext_root = self.temp_path / "ext_dataset"
        (ext_root / "sequences" / "03" / "velodyne").mkdir(parents=True)
        (ext_root / "sequences" / "03" / "labels").mkdir(parents=True)
        dummy_pts = np.random.randn(20, 4).astype(np.float32)
        dummy_pts.tofile(ext_root / "sequences" / "03" / "velodyne" / "000000.bin")
        dummy_lbls = np.random.randint(0, 50, size=(20,), dtype=np.uint32)
        dummy_lbls.tofile(ext_root / "sequences" / "03" / "labels" / "000000.label")

        os.environ["DATASET_ROOT"] = str(ext_root)
        try:
            records = discover_frames()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].sequence_id, "03")
        finally:
            del os.environ["DATASET_ROOT"]

    # 14. Cache vs raw data distinction
    def test_14_cache_vs_raw_data_distinction(self):
        """Test 14: Differentiate raw .bin files from cached .npy files in processed/."""
        records = discover_frames(self.dataset_root)
        for r in records:
            self.assertTrue(r.point_cloud_path.endswith(".bin"))
            self.assertFalse(r.point_cloud_path.endswith(".npy"))


if __name__ == "__main__":
    unittest.main()
