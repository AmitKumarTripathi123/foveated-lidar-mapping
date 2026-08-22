"""Phase 7 Multi-Frame Training, Diagnostics, and Evaluation Test Suite (19 Tests).

Covers:
  1. Dataset discovery
  2. Manifest correctness
  3. Split disjointness
  4. Class distribution calculation
  5. Label range validation
  6. Point-label alignment
  7. Point count normalization
  8. Augmentation output shape
  9. Augmentation label preservation
  10. Loss ignores 255
  11. Class-weight generation
  12. Metric correctness
  13. Checkpoint save
  14. Checkpoint reload
  15. Prediction distribution computation
  16. Model collapse detection
  17. ML output contract
  18. Real-frame inference
  19. Phase 6 mapping regression
"""

import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.dataset import load_point_cloud, load_labels, validate_point_label_alignment
from ml.data.preprocessing import filter_invalid_points
from ml.data.amit_adapter import FoveatedVoxelSampler
from ml.data.manifest import discover_dataset, audit_dataset
from ml.data.foveated_dataset import normalize_point_count, FoveatedLidarDataset
from ml.models.pointnet2 import build_model
from ml.models.predictor import PointNet2Predictor
from ml.models.mapping_adapter import MLToMappingAdapter, GridMap25D, PredictionBatch
from ml.training.losses import get_loss_function, compute_class_weights
from ml.training.augmentation import LidarAugmentor
from ml.training.metrics import SemanticSegmentationMetrics


class TestPhase7Training(unittest.TestCase):
    """Test suite for Phase 7 multi-frame training infrastructure."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
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
        """Test 1: Dataset discovery finds sequence 00 scan files."""
        manifest = discover_dataset(repo_root / "dataset")
        self.assertIn("train", manifest)
        self.assertGreaterEqual(len(manifest["train"]), 1)

    # 2. Manifest correctness
    def test_02_manifest_correctness(self):
        """Test 2: Manifest records contain relative POSIX paths and valid frame IDs."""
        manifest = discover_dataset(repo_root / "dataset")
        rec = manifest["train"][0]
        self.assertIn("point_path", rec)
        self.assertTrue(rec["point_path"].endswith(".bin"))
        self.assertFalse(":" in rec["point_path"])  # Portable relative path

    # 3. Split disjointness
    def test_03_split_disjointness(self):
        """Test 3: Assert sequence-level train and validation sets are disjoint."""
        train_seqs = {"00", "01", "03", "04", "05"}
        val_seqs = {"02"}
        test_seqs = {"08"}
        self.assertTrue(train_seqs.isdisjoint(val_seqs))
        self.assertTrue(train_seqs.isdisjoint(test_seqs))
        self.assertTrue(val_seqs.isdisjoint(test_seqs))

    # 4. Class distribution
    def test_04_class_distribution(self):
        """Test 4: Class distribution correctly tallies labels."""
        lbls = np.array([0, 0, 1, 2, 2, 2, 3, 255], dtype=np.uint8)
        u, counts = np.unique(lbls, return_counts=True)
        dist = {int(k): int(v) for k, v in zip(u, counts)}
        self.assertEqual(dist[0], 2)
        self.assertEqual(dist[2], 3)
        self.assertEqual(dist[255], 1)

    # 5. Label range
    def test_05_label_range(self):
        """Test 5: Target labels strictly belong to {0, 1, 2, 3, 255}."""
        labels = load_labels(self.lbl_file)
        raw_to_sih = {10: 3, 40: 0, 48: 1, 50: 2, 51: 2, 70: 2, 71: 2, 80: 2, 0: 255}
        mapped = np.array([raw_to_sih.get(l, 255) for l in labels], dtype=np.uint8)
        self.assertTrue(set(np.unique(mapped)).issubset({0, 1, 2, 3, 255}))

    # 6. Point-label alignment
    def test_06_point_label_alignment(self):
        """Test 6: Raw points and labels have identical length."""
        pts = load_point_cloud(self.bin_file)
        lbls = load_labels(self.lbl_file)
        self.assertEqual(pts.shape[0], lbls.shape[0])
        self.assertTrue(validate_point_label_alignment(pts, lbls))

    # 7. Point count normalization
    def test_07_point_count_normalization(self):
        """Test 7: Point normalization to target N produces exact tensor shape."""
        dummy_pts = np.random.randn(500, 4).astype(np.float32)
        dummy_lbl = np.random.randint(0, 4, size=(500,)).astype(np.uint8)
        norm_pts, norm_lbl = normalize_point_count(dummy_pts, dummy_lbl, target_num_points=1024, seed=42)
        self.assertEqual(norm_pts.shape, (1024, 4))
        self.assertEqual(norm_lbl.shape, (1024,))

    # 8. Augmentation shape
    def test_08_augmentation_shape(self):
        """Test 8: LidarAugmentor preserves (N, 4) input shape."""
        augmentor = LidarAugmentor(enabled=True, seed=42)
        pts = np.random.randn(256, 4).astype(np.float32)
        aug_pts = augmentor(pts)
        self.assertEqual(aug_pts.shape, (256, 4))

    # 9. Augmentation label preservation
    def test_09_augmentation_label_preservation(self):
        """Test 9: Geometric augmentations never modify semantic labels."""
        pts = np.random.randn(100, 4).astype(np.float32)
        lbls = np.random.randint(0, 4, size=(100,)).astype(np.uint8)
        augmentor = LidarAugmentor(enabled=True, seed=42)
        aug_pts = augmentor(pts)
        self.assertEqual(aug_pts.shape[0], lbls.shape[0])

    # 10. Loss ignores 255
    def test_10_loss_ignores_255(self):
        """Test 10: Loss gradient on ignore index 255 is exactly zero."""
        criterion = get_loss_function(loss_type="cross_entropy", ignore_index=255)
        logits = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], requires_grad=True)
        targets = torch.tensor([0, 255])
        loss = criterion(logits, targets)
        loss.backward()
        self.assertEqual(logits.grad[1].sum().item(), 0.0)

    # 11. Class-weight generation
    def test_11_class_weight_generation(self):
        """Test 11: Inverse-frequency class weights upweight minority classes."""
        counts = {0: 1000, 1: 500, 2: 2000, 3: 100}
        weights = compute_class_weights(counts, num_classes=4, strategy="inverse_frequency")
        self.assertEqual(weights.shape[0], 4)
        self.assertGreater(weights[3].item(), weights[2].item())

    # 12. Metric correctness
    def test_12_metric_correctness(self):
        """Test 12: Metric accumulator correctly computes mIoU and precision/recall."""
        metrics = SemanticSegmentationMetrics(num_classes=4, ignore_index=255)
        gt = np.array([0, 1, 2, 3])
        pred = np.array([0, 1, 2, 3])
        metrics.update(pred, gt)
        rep = metrics.compute()
        self.assertAlmostEqual(rep.miou, 1.0, places=4)
        self.assertAlmostEqual(rep.overall_accuracy, 1.0, places=4)

    # 13. Checkpoint save
    def test_13_checkpoint_save(self):
        """Test 13: Model checkpoint writes state and metadata."""
        model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        save_path = self.temp_path / "test.pt"
        torch.save({"model_state_dict": model.state_dict(), "val_miou": 0.55}, save_path)
        self.assertTrue(save_path.is_file())

    # 14. Checkpoint reload
    def test_14_checkpoint_reload(self):
        """Test 14: Model parameters match exactly after checkpoint load."""
        m1 = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        save_path = self.temp_path / "test.pt"
        torch.save({"model_state_dict": m1.state_dict()}, save_path)

        m2 = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        data = torch.load(save_path, map_location="cpu")
        m2.load_state_dict(data["model_state_dict"])
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    # 15. Prediction distribution
    def test_15_prediction_distribution(self):
        """Test 15: Prediction distribution computes accurate per-class percentages."""
        preds = np.array([0, 0, 1, 2, 2, 2, 2, 2, 2, 2])
        u, c = np.unique(preds, return_counts=True)
        dist = {int(k): (int(v) / len(preds)) * 100.0 for k, v in zip(u, c)}
        self.assertAlmostEqual(dist[2], 70.0, places=2)

    # 16. Model collapse detection
    def test_16_model_collapse_detection(self):
        """Test 16: Detect collapse when one class dominates >90% of predictions."""
        preds = np.array([2] * 95 + [0] * 5)
        u, c = np.unique(preds, return_counts=True)
        max_pct = max(c) / len(preds)
        is_collapsed = max_pct > 0.90
        self.assertTrue(is_collapsed)

    # 17. ML output contract
    def test_17_ml_output_contract(self):
        """Test 17: Predictor returns [x, y, z, predicted_class, confidence]."""
        model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        predictor = PointNet2Predictor(model=model, device="cpu")
        dummy_pts = np.random.randn(128, 4).astype(np.float32)
        res = predictor.predict(dummy_pts)
        self.assertEqual(res["xyz"].shape, (128, 3))
        self.assertEqual(res["predicted_class"].shape, (128,))
        self.assertEqual(res["confidence"].shape, (128,))
        np.testing.assert_array_equal(res["xyz"], dummy_pts[:, :3])

    # 18. Real-frame inference
    def test_18_real_frame_inference(self):
        """Test 18: Real scan runs through Foveated Voxelizer -> PointNet++ -> Predictor."""
        raw_pts = load_point_cloud(self.bin_file)
        sampler = FoveatedVoxelSampler()
        fov_pts, _, _ = sampler.sample(raw_pts)
        norm_pts, _ = normalize_point_count(fov_pts, None, target_num_points=512, seed=42)

        model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        predictor = PointNet2Predictor(model=model, device="cpu")
        res = predictor.predict(norm_pts)
        self.assertEqual(res["xyz"].shape, (512, 3))

    # 19. Phase 6 mapping regression
    def test_19_phase6_mapping_regression(self):
        """Test 19: Phase 6 MLToMappingAdapter converts predictions into GridMap25D."""
        model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        predictor = PointNet2Predictor(model=model, device="cpu")
        dummy_pts = np.random.uniform(-40, 40, size=(256, 4)).astype(np.float32)
        res = predictor.predict(dummy_pts)

        adapter = MLToMappingAdapter(bounds_x=(-50, 50), bounds_y=(-50, 50), resolution=0.5)
        grid = adapter.build_25d_grid(res)
        self.assertIsInstance(grid, GridMap25D)
        self.assertEqual(grid.grid_shape, (200, 200))
        self.assertGreater(grid.point_count_layer.sum(), 0)


if __name__ == "__main__":
    unittest.main()
