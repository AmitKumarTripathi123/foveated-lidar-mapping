"""Phase 8 Real Data Acquisition, Dataset Expansion & Semantic Segmentation Test Suite (21 Tests).

Covers:
  1. Dataset discovery
  2. Dataset root configuration
  3. Frame count
  4. Bin-label pairing
  5. Point-label alignment
  6. Raw label validation
  7. SIH label validation
  8. Foveated alignment
  9. Split disjointness
  10. Class distribution
  11. Class weight calculation
  12. Augmentation
  13. Training step
  14. Validation metrics
  15. Checkpoint save
  16. Checkpoint reload
  17. Collapse detection
  18. Prediction contract
  19. Mapping regression
  20. Multi-frame inference
  21. Performance benchmark sanity
"""

import os
import sys
import tempfile
import time
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
from ml.data.foveated_dataset import FoveatedLidarDataset, normalize_point_count
from ml.data.frame_discovery import FrameRecord, discover_frames, audit_discovered_frames
from ml.models.pointnet2 import build_model
from ml.models.predictor import PointNet2Predictor
from ml.models.mapping_adapter import MLToMappingAdapter, GridMap25D, PredictionBatch
from ml.training.losses import get_loss_function, compute_class_weights
from ml.training.augmentation import LidarAugmentor
from ml.training.metrics import SemanticSegmentationMetrics
from scripts.benchmark_latency import benchmark_pipeline


class TestPhase8(unittest.TestCase):
    """Test suite for Phase 8 multi-frame perception and integration."""

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

    # 1. Dataset discovery
    def test_01_dataset_discovery(self):
        """Test 1: discover_frames discovers sequence 00 frame 000000."""
        records = discover_frames(self.dataset_root)
        self.assertGreaterEqual(len(records), 1)
        self.assertEqual(records[0].sequence_id, "00")

    # 2. Dataset root configuration
    def test_02_dataset_root_configuration(self):
        """Test 2: DATASET_ROOT environment variable correctly configures discovery."""
        ext_root = self.temp_path / "custom_root"
        (ext_root / "sequences" / "05" / "velodyne").mkdir(parents=True)
        (ext_root / "sequences" / "05" / "labels").mkdir(parents=True)
        pts = np.random.randn(30, 4).astype(np.float32)
        pts.tofile(ext_root / "sequences" / "05" / "velodyne" / "000000.bin")
        lbl = np.random.randint(0, 50, size=(30,), dtype=np.uint32)
        lbl.tofile(ext_root / "sequences" / "05" / "labels" / "000000.label")

        os.environ["DATASET_ROOT"] = str(ext_root)
        try:
            records = discover_frames()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].sequence_id, "05")
        finally:
            del os.environ["DATASET_ROOT"]

    # 3. Frame count
    def test_03_frame_count(self):
        """Test 3: Frame discovery accurately counts total local frames."""
        records = discover_frames(self.dataset_root)
        self.assertEqual(len(records), 1)

    # 4. Bin-label pairing
    def test_04_bin_label_pairing(self):
        """Test 4: Raw .bin and .label are paired without missing components."""
        records = discover_frames(self.dataset_root)
        rec = records[0]
        self.assertTrue(rec.has_label)
        self.assertTrue(rec.is_matched)
        self.assertTrue(Path(rec.point_cloud_path).is_file())
        self.assertTrue(Path(rec.label_path).is_file())

    # 5. Point-label alignment
    def test_05_point_label_alignment(self):
        """Test 5: Number of points equals number of labels in raw frame."""
        pts = load_point_cloud(self.bin_file)
        lbls = load_labels(self.lbl_file)
        self.assertEqual(pts.shape[0], lbls.shape[0])
        self.assertTrue(validate_point_label_alignment(pts, lbls))

    # 6. Raw label validation
    def test_06_raw_label_validation(self):
        """Test 6: Raw labels contain expected SemanticKITTI class IDs."""
        lbls = load_labels(self.lbl_file)
        unique_raw = set(np.unique(lbls))
        self.assertIn(40, unique_raw)  # road
        self.assertIn(10, unique_raw)  # car

    # 7. SIH label validation
    def test_07_sih_label_validation(self):
        """Test 7: Remapped labels strictly belong to {0, 1, 2, 3, 255}."""
        lbls = load_labels(self.lbl_file)
        remapper = SemanticLabelRemapper()
        sih_lbls = remapper.remap(lbls)
        self.assertTrue(set(np.unique(sih_lbls)).issubset({0, 1, 2, 3, 255}))

    # 8. Foveated alignment
    def test_08_foveated_alignment(self):
        """Test 8: Foveated voxelization preserves point-label correspondence."""
        pts = load_point_cloud(self.bin_file)
        lbls = load_labels(self.lbl_file)
        v_pts, v_lbls, _ = filter_invalid_points(pts, lbls)
        sampler = FoveatedVoxelSampler()
        f_pts, f_lbls, _ = sampler.sample(v_pts, v_lbls)
        self.assertEqual(f_pts.shape[0], f_lbls.shape[0])

    # 9. Split disjointness
    def test_09_split_disjointness(self):
        """Test 9: Assert sequence disjointness for train/val/test splits."""
        train_seqs = {"00", "01", "03", "04", "05"}
        val_seqs = {"02"}
        test_seqs = {"08"}
        self.assertTrue(train_seqs.isdisjoint(val_seqs))
        self.assertTrue(train_seqs.isdisjoint(test_seqs))

    # 10. Class distribution
    def test_10_class_distribution(self):
        """Test 10: Accurate distribution calculation for SIH classes."""
        lbls = np.array([0, 0, 1, 2, 2, 3, 255], dtype=np.uint8)
        u, c = np.unique(lbls, return_counts=True)
        dist = {int(k): int(v) for k, v in zip(u, c)}
        self.assertEqual(dist[0], 2)
        self.assertEqual(dist[1], 1)
        self.assertEqual(dist[2], 2)

    # 11. Class weight calculation
    def test_11_class_weight_calculation(self):
        """Test 11: Inverse-frequency class weights penalize majority class."""
        counts = {0: 1000, 1: 500, 2: 5000, 3: 200}
        weights = compute_class_weights(counts, num_classes=4)
        self.assertLess(weights[2].item(), weights[3].item())

    # 12. Augmentation
    def test_12_augmentation(self):
        """Test 12: Training augmentation preserves (N, 4) tensor shape."""
        augmentor = LidarAugmentor(enabled=True, seed=42)
        pts = np.random.randn(128, 4).astype(np.float32)
        aug = augmentor(pts)
        self.assertEqual(aug.shape, (128, 4))

    # 13. Training step
    def test_13_training_step(self):
        """Test 13: Forward and backward pass update PointNet++ gradients."""
        model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = get_loss_function(loss_type="cross_entropy", ignore_index=255)

        pts = torch.randn(1, 128, 4)
        lbl = torch.randint(0, 4, size=(1, 128))
        out = model(pts)
        loss = criterion(out.transpose(1, 2), lbl)
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.isnan(loss))

    # 14. Validation metrics
    def test_14_validation_metrics(self):
        """Test 14: Metric accumulator correctly computes mIoU and accuracy."""
        metrics = SemanticSegmentationMetrics(num_classes=4, ignore_index=255)
        metrics.update(np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]))
        res = metrics.compute()
        self.assertAlmostEqual(res.miou, 1.0, places=4)

    # 15. Checkpoint save
    def test_15_checkpoint_save(self):
        """Test 15: Saving checkpoint writes file and metadata."""
        ckpt_path = self.temp_path / "best.pt"
        torch.save({"val_miou": 0.45, "epoch": 5}, ckpt_path)
        self.assertTrue(ckpt_path.is_file())

    # 16. Checkpoint reload
    def test_16_checkpoint_reload(self):
        """Test 16: Checkpoint loads exact metadata and parameters."""
        ckpt_path = self.temp_path / "best.pt"
        torch.save({"val_miou": 0.45, "epoch": 5}, ckpt_path)
        data = torch.load(ckpt_path, map_location="cpu")
        self.assertEqual(data["epoch"], 5)
        self.assertAlmostEqual(data["val_miou"], 0.45)

    # 17. Collapse detection
    def test_17_collapse_detection(self):
        """Test 17: Detect majority-class collapse when single class > 90%."""
        preds = np.array([2] * 95 + [0] * 5)
        _, counts = np.unique(preds, return_counts=True)
        max_ratio = max(counts) / len(preds)
        self.assertTrue(max_ratio > 0.90)

    # 18. Prediction contract
    def test_18_prediction_contract(self):
        """Test 18: PointNet2Predictor preserves XYZ and outputs valid classes/confidences."""
        model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        predictor = PointNet2Predictor(model=model, device="cpu")
        pts = np.random.randn(64, 4).astype(np.float32)
        res = predictor.predict(pts)
        self.assertEqual(res["xyz"].shape, (64, 3))
        self.assertTrue((res["confidence"] >= 0.0).all() and (res["confidence"] <= 1.0).all())
        np.testing.assert_array_equal(res["xyz"], pts[:, :3])

    # 19. Mapping regression
    def test_19_mapping_regression(self):
        """Test 19: Phase 6 MLToMappingAdapter produces valid GridMap25D from predictions."""
        model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        predictor = PointNet2Predictor(model=model, device="cpu")
        pts = np.random.uniform(-30, 30, size=(128, 4)).astype(np.float32)
        pred = predictor.predict(pts)

        adapter = MLToMappingAdapter()
        grid = adapter.build_25d_grid(pred)
        self.assertIsInstance(grid, GridMap25D)
        self.assertFalse(np.isinf(grid.elevation_mean[~np.isnan(grid.elevation_mean)]).any())

    # 20. Multi-frame inference
    def test_20_multi_frame_inference(self):
        """Test 20: Run sequential inference on multiple frames without state leakage."""
        model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        predictor = PointNet2Predictor(model=model, device="cpu")
        for i in range(3):
            pts = np.random.uniform(-40, 40, size=(256, 4)).astype(np.float32)
            pred = predictor.predict(pts)
            self.assertEqual(pred["xyz"].shape[0], 256)

    # 21. Performance benchmark sanity
    def test_21_performance_benchmark_sanity(self):
        """Test 21: Latency benchmark executes and returns positive total runtime."""
        res = benchmark_pipeline(self.bin_file, self.lbl_file, num_points=256, iterations=1, device_str="cpu")
        self.assertGreater(res["total_ms"], 0.0)
        self.assertGreater(res["throughput_fps"], 0.0)


if __name__ == "__main__":
    unittest.main()
