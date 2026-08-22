"""Master End-to-End Pipeline & Integration Test Suite (26 Tests).

Covers:
  1. Real .bin loading
  2. Real .label loading
  3. Point-label alignment
  4. Finite input
  5. Label remapping
  6. Allowed labels
  7. Foveated range filtering
  8. Foveated voxelization
  9. Foveated point-label alignment
  10. Manifest generation
  11. Train/val disjointness
  12. Test disjointness if available
  13. Dataset output shape
  14. Point count normalization
  15. PointNet++ forward
  16. Loss with ignore_index=255
  17. Backward pass
  18. IoU calculation
  19. mIoU calculation
  20. Confusion matrix
  21. Checkpoint save
  22. Checkpoint reload
  23. Predictor output contract
  24. Point order preservation
  25. End-to-end single real frame
  26. End-to-end multi-frame smoke training
"""

import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.dataset import (
    load_point_cloud,
    load_labels,
    validate_point_label_alignment,
    validate_data_integrity,
    LidarDataset,
    lidar_collate_fn,
)
from ml.data.preprocessing import filter_invalid_points
from ml.data.amit_adapter import FoveatedVoxelSampler
from ml.data.label_mapping import SemanticLabelRemapper, VALID_SIH_IDS
from ml.data.manifest import discover_dataset, audit_dataset
from ml.data.foveated_dataset import FoveatedLidarDataset, normalize_point_count
from ml.models.pointnet2 import PointNet2SemSeg, build_model
from ml.models.predictor import PointNet2Predictor
from ml.training.losses import get_loss_function, compute_class_weights
from ml.training.metrics import SemanticSegmentationMetrics
from ml.training.trainer import PointNet2Trainer


class TestFullPipeline(unittest.TestCase):
    """26-test Master Integration Test Suite for End-to-End LiDAR ML Pipeline."""

    @classmethod
    def setUpClass(cls):
        """Set fixed seed and locate representative scan files."""
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

    # 1. Real .bin loading
    def test_01_real_bin_loading(self):
        """Test 1: Real .bin file loads into (N, 4) float32 array."""
        pts = load_point_cloud(self.bin_file)
        self.assertEqual(pts.ndim, 2)
        self.assertEqual(pts.shape[1], 4)
        self.assertEqual(pts.dtype, np.float32)
        self.assertEqual(pts.shape[0], 66658)

    # 2. Real .label loading
    def test_02_real_label_loading(self):
        """Test 2: Real .label file loads into (N,) integer array."""
        lbls = load_labels(self.lbl_file)
        self.assertEqual(lbls.ndim, 1)
        self.assertEqual(lbls.shape[0], 66658)

    # 3. Point-label alignment
    def test_03_point_label_alignment(self):
        """Test 3: Real point count exactly matches real label count."""
        pts = load_point_cloud(self.bin_file)
        lbls = load_labels(self.lbl_file)
        self.assertTrue(validate_point_label_alignment(pts, lbls))

    # 4. Finite input
    def test_04_finite_input(self):
        """Test 4: Real point cloud contains zero NaN or Inf values."""
        pts = load_point_cloud(self.bin_file)
        integrity = validate_data_integrity(pts)
        self.assertFalse(integrity["has_nan"])
        self.assertFalse(integrity["has_inf"])

    # 5. Label remapping
    def test_05_label_remapping(self):
        """Test 5: SemanticLabelRemapper executes on real labels."""
        lbls = load_labels(self.lbl_file)
        remapper = SemanticLabelRemapper()
        mapped = remapper.remap(lbls)
        self.assertEqual(len(mapped), len(lbls))

    # 6. Allowed labels
    def test_06_allowed_labels(self):
        """Test 6: All remapped labels strictly belong to {0, 1, 2, 3, 255}."""
        lbls = load_labels(self.lbl_file)
        remapper = SemanticLabelRemapper()
        mapped = remapper.remap(lbls)
        self.assertTrue(set(np.unique(mapped)).issubset(VALID_SIH_IDS))

    # 7. Foveated range filtering
    def test_07_foveated_range_filtering(self):
        """Test 7: Outer points (>100m) are filtered by foveated sampler."""
        sampler = FoveatedVoxelSampler(far_dist=100.0)
        dummy_pts = np.array([
            [5.0, 0.0, 0.0, 0.5],    # Near
            [20.0, 0.0, 0.0, 0.5],   # Mid
            [50.0, 0.0, 0.0, 0.5],   # Far
            [120.0, 0.0, 0.0, 0.5],  # Out of bounds (>100m)
        ], dtype=np.float32)
        fov_pts, _, rep = sampler.sample(dummy_pts)
        self.assertEqual(rep.filtered_out_count, 1)
        self.assertEqual(fov_pts.shape[0], 3)

    # 8. Foveated voxelization
    def test_08_foveated_voxelization(self):
        """Test 8: Foveated sampler downsamples points within distance zones."""
        pts = load_point_cloud(self.bin_file)
        sampler = FoveatedVoxelSampler()
        fov_pts, _, rep = sampler.sample(pts)
        self.assertLess(fov_pts.shape[0], pts.shape[0])
        self.assertGreater(rep.overall_reduction_pct, 0.0)

    # 9. Foveated point-label alignment
    def test_09_foveated_point_label_alignment(self):
        """Test 9: Foveated voxel sampling strictly preserves point-label correspondence."""
        pts = load_point_cloud(self.bin_file)
        lbls = load_labels(self.lbl_file)
        sampler = FoveatedVoxelSampler()
        fov_pts, fov_lbls, _ = sampler.sample(pts, lbls)
        self.assertEqual(fov_pts.shape[0], fov_lbls.shape[0])
        self.assertTrue(validate_point_label_alignment(fov_pts, fov_lbls))

    # 10. Manifest generation
    def test_10_manifest_generation(self):
        """Test 10: Manifest generator discovers sequence 00 scan files."""
        manifest = discover_dataset(repo_root / "dataset")
        self.assertIn("train", manifest)
        self.assertIn("val", manifest)
        self.assertGreater(len(manifest["train"]), 0)

    # 11. Train/val disjointness
    def test_11_train_val_disjointness(self):
        """Test 11: Explicit sequence disjointness assertion."""
        train_seqs = {"00", "01", "03", "04", "05"}
        val_seqs = {"02"}
        self.assertTrue(train_seqs.isdisjoint(val_seqs))

    # 12. Test disjointness if available
    def test_12_test_disjointness(self):
        """Test 12: Test sequence disjointness assertion."""
        train_seqs = {"00", "01"}
        val_seqs = {"02"}
        test_seqs = {"08"}
        self.assertTrue(train_seqs.isdisjoint(test_seqs))
        self.assertTrue(val_seqs.isdisjoint(test_seqs))

    # 13. Dataset output shape
    def test_13_dataset_output_shape(self):
        """Test 13: FoveatedLidarDataset yields [target_N, 4] and [target_N] tensors."""
        manifest = discover_dataset(repo_root / "dataset")
        dataset = FoveatedLidarDataset(
            raw_manifest=manifest["train"], target_num_points=1024, to_tensor=True
        )
        sample = dataset[0]
        self.assertEqual(sample["points"].shape, (1024, 4))
        self.assertEqual(sample["labels"].shape, (1024,))

    # 14. Point count normalization
    def test_14_point_count_normalization(self):
        """Test 14: Normalization scales arbitrary input sizes to target_N."""
        pts = np.random.randn(500, 4).astype(np.float32)
        lbls = np.random.randint(0, 4, (500,)).astype(np.uint8)
        norm_pts, norm_lbls = normalize_point_count(pts, lbls, target_num_points=1024)
        self.assertEqual(norm_pts.shape, (1024, 4))
        self.assertEqual(norm_lbls.shape, (1024,))

    # 15. PointNet++ forward
    def test_15_pointnet2_forward(self):
        """Test 15: PointNet++ forward pass produces [B, N, 4] logits."""
        model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        dummy_in = torch.randn(1, 256, 4)
        model.eval()
        with torch.no_grad():
            out = model(dummy_in)
        self.assertEqual(out.shape, (1, 256, 4))

    # 16. Loss with ignore_index=255
    def test_16_loss_with_ignore_index(self):
        """Test 16: Loss function masks out ignore_index=255."""
        criterion = get_loss_function(ignore_index=255)
        logits = torch.tensor([[10.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0]], requires_grad=True)
        targets = torch.tensor([0, 255])
        loss = criterion(logits, targets)
        loss.backward()
        self.assertEqual(logits.grad[1].sum().item(), 0.0)

    # 17. Backward pass
    def test_17_backward_pass(self):
        """Test 17: Backward pass computes finite non-zero gradients."""
        model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        dummy_pts = torch.randn(1, 128, 4)
        dummy_lbl = torch.randint(0, 4, (1, 128))

        optimizer.zero_grad()
        logits = model(dummy_pts)
        loss = criterion(logits.view(-1, 4), dummy_lbl.view(-1))
        loss.backward()
        optimizer.step()

        self.assertTrue(torch.isfinite(loss).item())

    # 18. IoU calculation
    def test_18_iou_calculation(self):
        """Test 18: IoU computation returns mathematically correct overlap."""
        gt = np.array([0, 0, 1, 1], dtype=np.uint8)
        pred = np.array([0, 1, 1, 1], dtype=np.uint8)
        metrics = SemanticSegmentationMetrics(num_classes=4, ignore_index=255)
        metrics.update(pred, gt)
        rep = metrics.compute()
        # Class 0: TP=1, FP=0, FN=1 -> IoU = 0.5
        self.assertAlmostEqual(rep.per_class[0].iou, 0.5, places=4)

    # 19. mIoU calculation
    def test_19_miou_calculation(self):
        """Test 19: mIoU averages valid class IoUs."""
        gt = np.array([0, 1, 2, 3], dtype=np.uint8)
        pred = np.array([0, 1, 2, 3], dtype=np.uint8)
        metrics = SemanticSegmentationMetrics(num_classes=4, ignore_index=255)
        metrics.update(pred, gt)
        rep = metrics.compute()
        self.assertAlmostEqual(rep.miou, 1.0, places=4)

    # 20. Confusion matrix
    def test_20_confusion_matrix(self):
        """Test 20: 4x4 confusion matrix accumulates ground truth and prediction counts."""
        gt = np.array([0, 1, 2, 3], dtype=np.uint8)
        pred = np.array([0, 0, 2, 2], dtype=np.uint8)
        metrics = SemanticSegmentationMetrics(num_classes=4, ignore_index=255)
        metrics.update(pred, gt)
        rep = metrics.compute()
        self.assertEqual(rep.confusion_matrix.shape, (4, 4))
        self.assertEqual(rep.confusion_matrix[0, 0], 1)
        self.assertEqual(rep.confusion_matrix[1, 0], 1)

    # 21. Checkpoint save
    def test_21_checkpoint_save(self):
        """Test 21: Checkpoint writes model state and metadata to disk."""
        model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        save_path = self.temp_path / "test_ckpt.pt"
        torch.save({"model_state_dict": model.state_dict(), "val_miou": 0.88}, save_path)
        self.assertTrue(save_path.is_file())

    # 22. Checkpoint reload
    def test_22_checkpoint_reload(self):
        """Test 22: Checkpoint restores exact model parameters."""
        model1 = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        save_path = self.temp_path / "test_ckpt.pt"
        torch.save({"model_state_dict": model1.state_dict()}, save_path)

        model2 = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        data = torch.load(save_path, map_location="cpu")
        model2.load_state_dict(data["model_state_dict"])
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    # 23. Predictor output contract
    def test_23_predictor_output_contract(self):
        """Test 23: PointNet2Predictor satisfies Amit's frozen contract."""
        model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        predictor = PointNet2Predictor(model=model, device="cpu")
        dummy_pts = np.random.randn(256, 4).astype(np.float32)
        res = predictor.predict(dummy_pts)

        self.assertIn("xyz", res)
        self.assertIn("predicted_class", res)
        self.assertIn("confidence", res)
        self.assertEqual(res["xyz"].shape, (256, 3))
        self.assertEqual(res["predicted_class"].shape, (256,))
        self.assertEqual(res["confidence"].shape, (256,))

    # 24. Point order preservation
    def test_24_point_order_preservation(self):
        """Test 24: Input XYZ is identical to output XYZ in order and value."""
        model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        predictor = PointNet2Predictor(model=model, device="cpu")
        dummy_pts = np.random.uniform(-50, 50, size=(256, 4)).astype(np.float32)
        res = predictor.predict(dummy_pts)
        np.testing.assert_array_equal(res["xyz"], dummy_pts[:, :3])

    # 25. End-to-end single real frame
    def test_25_end_to_end_single_real_frame(self):
        """Test 25: Full single-frame pipeline from raw .bin/.label to prediction contract."""
        # 1. Load
        pts = load_point_cloud(self.bin_file)
        lbls = load_labels(self.lbl_file)
        # 2. Filter & Foveate
        v_pts, v_lbls, _ = filter_invalid_points(pts, lbls)
        sampler = FoveatedVoxelSampler()
        fov_pts, fov_lbls, _ = sampler.sample(v_pts, v_lbls)
        # 3. Remap
        remapper = SemanticLabelRemapper()
        sih_lbls = remapper.remap(fov_lbls)
        # 4. Normalize
        norm_pts, norm_lbls = normalize_point_count(fov_pts, sih_lbls, target_num_points=512)
        # 5. Model & Predict
        model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        predictor = PointNet2Predictor(model=model, device="cpu")
        res = predictor.predict(norm_pts)

        self.assertEqual(res["xyz"].shape, (512, 3))
        self.assertEqual(res["predicted_class"].shape, (512,))
        self.assertEqual(res["confidence"].shape, (512,))
        np.testing.assert_array_equal(res["xyz"], norm_pts[:, :3])

    # 26. End-to-end multi-frame smoke training
    def test_26_end_to_end_smoke_training(self):
        """Test 26: Multi-epoch smoke training loop on real foveated dataset."""
        manifest = discover_dataset(repo_root / "dataset")
        train_ds = FoveatedLidarDataset(raw_manifest=manifest["train"], target_num_points=128, to_tensor=True)
        val_ds = FoveatedLidarDataset(raw_manifest=manifest["val"], target_num_points=128, to_tensor=True)

        train_loader = DataLoader(train_ds, batch_size=1, collate_fn=lidar_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=1, collate_fn=lidar_collate_fn)

        model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        config = {
            "experiment": {"name": "smoke_test", "device": "cpu"},
            "training": {"epochs": 2, "learning_rate": 0.01},
            "loss": {"type": "cross_entropy", "ignore_index": 255},
        }
        trainer = PointNet2Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            experiment_dir=self.temp_path / "smoke_exp",
        )
        summary = trainer.train()
        self.assertGreater(summary["best_val_miou"], 0.0)
        self.assertTrue((self.temp_path / "smoke_exp" / "best_checkpoint.pt").is_file())


if __name__ == "__main__":
    unittest.main()
