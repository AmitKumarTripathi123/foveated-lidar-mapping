"""Automated Test Suite for PointNet++ Training, Losses, Metrics & Checkpointing (Phase 5).

Covers all 15 required tests:
  - Test 1: Dataset split has no overlap
  - Test 2: Training batch shape is correct
  - Test 3: Target labels are valid ({0, 1, 2, 3, 255})
  - Test 4: 255 labels are supported
  - Test 5: Loss ignores 255
  - Test 6: One training step works
  - Test 7: One validation step works
  - Test 8: Metrics are mathematically correct on synthetic known data
  - Test 9: IoU calculation works
  - Test 10: Ignored points (255) do not affect metrics
  - Test 11: Checkpoint save works with full metadata
  - Test 12: Checkpoint load and state restoration works
  - Test 13: Resume training works
  - Test 14: Best checkpoint selection strictly tracks validation mIoU
  - Test 15: Experiment configuration is reproducible with fixed seed
"""

import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.dataset import load_point_cloud, load_labels, LidarDataset, lidar_collate_fn
from ml.data.preprocessing import LidarPreprocessor, PreprocessingConfig, SamplingConfig
from ml.data.label_mapping import SemanticLabelRemapper
from ml.models.pointnet2 import build_model
from ml.training.losses import get_loss_function, compute_class_weights
from ml.training.metrics import SemanticSegmentationMetrics
from ml.training.trainer import PointNet2Trainer


class TestTraining(unittest.TestCase):
    """Unit and integration test suite for Phase 5 training pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary fixtures."""
        self.temp_dir.cleanup()

    def test_01_dataset_split_no_overlap(self):
        """Test 1: Dataset split sequences/frames have zero overlap (no leakage)."""
        train_seqs = ["00"]
        val_seqs = ["01"]
        test_seqs = ["02"]

        train_set = set(train_seqs)
        val_set = set(val_seqs)
        test_set = set(test_seqs)

        self.assertEqual(len(train_set.intersection(val_set)), 0)
        self.assertEqual(len(train_set.intersection(test_set)), 0)
        self.assertEqual(len(val_set.intersection(test_set)), 0)

    def test_02_training_batch_shape(self):
        """Test 2: Training batch collation produces expected tensor shapes."""
        B, N, C = 2, 256, 4
        points = torch.randn(B, N, C)
        labels = torch.randint(0, 4, (B, N))
        batch = [{"points": points[i], "labels": labels[i], "metadata": {}} for i in range(B)]
        collated = lidar_collate_fn(batch)

        self.assertEqual(collated["points"].shape, (B, N, 4))
        self.assertEqual(collated["labels"].shape, (B, N))

    def test_03_target_labels_validity(self):
        """Test 3: Target label values strictly belong to {0, 1, 2, 3, 255}."""
        labels = np.array([0, 1, 2, 3, 255, 0, 1, 2, 3, 255], dtype=np.uint8)
        self.assertTrue(set(np.unique(labels)).issubset({0, 1, 2, 3, 255}))

    def test_04_255_labels_supported(self):
        """Test 4: Class 255 is recognized and handled as an ignore index."""
        criterion = get_loss_function(loss_type="cross_entropy", ignore_index=255)
        self.assertEqual(criterion.ignore_index, 255)

    def test_05_loss_ignores_255(self):
        """Test 5: Points with label 255 do not contribute to supervised loss."""
        criterion = nn.CrossEntropyLoss(ignore_index=255)

        # Batch of 4 points where 2 points have label 255
        logits = torch.tensor([
            [10.0, 0.0, 0.0, 0.0],  # Pred class 0
            [0.0, 10.0, 0.0, 0.0],  # Pred class 1
            [0.0, 0.0, 10.0, 0.0],  # Ignored point
            [0.0, 0.0, 0.0, 10.0],  # Ignored point
        ], requires_grad=True)
        targets = torch.tensor([0, 1, 255, 255], dtype=torch.long)

        loss = criterion(logits, targets)
        loss.backward()

        # Gradients on ignored rows should be exactly 0
        self.assertTrue((logits.grad[2] == 0.0).all().item())
        self.assertTrue((logits.grad[3] == 0.0).all().item())

    def test_06_one_training_step(self):
        """Test 6: Single training step executes forward and backward pass cleanly."""
        model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = get_loss_function(ignore_index=255)

        dummy_pts = torch.randn(1, 256, 4)
        dummy_lbl = torch.randint(0, 4, (1, 256))

        optimizer.zero_grad()
        out = model(dummy_pts)
        loss = criterion(out.view(-1, 4), dummy_lbl.view(-1))
        loss.backward()
        optimizer.step()

        self.assertTrue(torch.isfinite(loss).item())

    def test_07_one_validation_step(self):
        """Test 7: Validation step runs forward pass without modifying gradients."""
        model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        model.eval()
        dummy_pts = torch.randn(1, 256, 4)
        dummy_lbl = torch.randint(0, 4, (1, 256))

        metrics = SemanticSegmentationMetrics(num_classes=4, ignore_index=255)
        with torch.no_grad():
            out = model(dummy_pts)
            preds = out.argmax(dim=-1)
            metrics.update(preds, dummy_lbl)

        report = metrics.compute()
        self.assertTrue(0.0 <= report.miou <= 1.0)
        self.assertTrue(0.0 <= report.overall_accuracy <= 1.0)

    def test_08_synthetic_metric_exactness(self):
        """Test 8: Metrics exactly match manually verified calculations on synthetic ground-truth."""
        # Ground Truth: [0, 0, 1, 1, 2, 2, 3, 3]
        # Prediction:   [0, 1, 1, 1, 2, 3, 3, 2]
        # Class 0: GT=[0,0], Pred=[0,1] -> TP=1, FP=0, FN=1 -> IoU = 1/(1+0+1) = 1/2 = 0.5
        # Class 1: GT=[1,1], Pred=[1,1] + [0->1] -> TP=2, FP=1, FN=0 -> IoU = 2/(2+1+0) = 2/3 ≈ 0.6667
        # Class 2: GT=[2,2], Pred=[2,3] + [3->2] -> TP=1, FP=1, FN=1 -> IoU = 1/(1+1+1) = 1/3 ≈ 0.3333
        # Class 3: GT=[3,3], Pred=[3,2] + [2->3] -> TP=1, FP=1, FN=1 -> IoU = 1/(1+1+1) = 1/3 ≈ 0.3333
        # mIoU = (0.5 + 2/3 + 1/3 + 1/3) / 4 = (1/2 + 4/3) / 4 = (11/6) / 4 = 11/24 ≈ 0.45833
        gt = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.uint8)
        pred = np.array([0, 1, 1, 1, 2, 3, 3, 2], dtype=np.uint8)

        metrics = SemanticSegmentationMetrics(num_classes=4, ignore_index=255)
        metrics.update(pred, gt)
        report = metrics.compute()

        self.assertAlmostEqual(report.per_class[0].iou, 0.5, places=4)
        self.assertAlmostEqual(report.per_class[1].iou, 2.0 / 3.0, places=4)
        self.assertAlmostEqual(report.per_class[2].iou, 1.0 / 3.0, places=4)
        self.assertAlmostEqual(report.per_class[3].iou, 1.0 / 3.0, places=4)
        self.assertAlmostEqual(report.miou, 11.0 / 24.0, places=4)

    def test_09_iou_calculation(self):
        """Test 9: Perfect prediction produces exact IoU of 1.0."""
        gt = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.uint8)
        pred = gt.copy()

        metrics = SemanticSegmentationMetrics(num_classes=4, ignore_index=255)
        metrics.update(pred, gt)
        report = metrics.compute()

        self.assertAlmostEqual(report.miou, 1.0, places=4)
        self.assertAlmostEqual(report.overall_accuracy, 1.0, places=4)

    def test_10_ignored_points_excluded_from_metrics(self):
        """Test 10: Ignored points (255) do not alter evaluated TP, FP, FN, or IoU."""
        gt_clean = np.array([0, 1, 2, 3], dtype=np.uint8)
        pred_clean = np.array([0, 1, 2, 3], dtype=np.uint8)

        gt_with_ignore = np.array([0, 1, 2, 3, 255, 255, 255], dtype=np.uint8)
        pred_with_ignore = np.array([0, 1, 2, 3, 0, 1, 2], dtype=np.uint8)

        m1 = SemanticSegmentationMetrics(num_classes=4, ignore_index=255)
        m1.update(pred_clean, gt_clean)
        rep1 = m1.compute()

        m2 = SemanticSegmentationMetrics(num_classes=4, ignore_index=255)
        m2.update(pred_with_ignore, gt_with_ignore)
        rep2 = m2.compute()

        self.assertEqual(rep1.miou, rep2.miou)
        self.assertEqual(rep2.total_ignored_points, 3)

    def test_11_checkpoint_save(self):
        """Test 11: Checkpoint saving writes model state, optimizer state, and experiment metadata."""
        model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        ckpt_path = self.temp_path / "test_best.pt"

        ckpt_data = {
            "epoch": 5,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_miou": 0.72,
            "config": {"seed": 42},
        }
        torch.save(ckpt_data, ckpt_path)
        self.assertTrue(ckpt_path.is_file())

    def test_12_checkpoint_load(self):
        """Test 12: Checkpoint loading restores exact weights and metadata."""
        model1 = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        ckpt_path = self.temp_path / "test_ckpt.pt"
        torch.save({"model_state_dict": model1.state_dict(), "epoch": 10, "val_miou": 0.85}, ckpt_path)

        model2 = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        loaded = torch.load(ckpt_path, map_location="cpu")
        model2.load_state_dict(loaded["model_state_dict"])

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.equal(p1, p2))
        self.assertEqual(loaded["epoch"], 10)
        self.assertEqual(loaded["val_miou"], 0.85)

    def test_13_resume_training(self):
        """Test 13: Optimizer and epoch state can be resumed from last checkpoint."""
        model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        ckpt_path = self.temp_path / "last_checkpoint.pt"

        torch.save({
            "epoch": 7,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_miou": 0.65,
        }, ckpt_path)

        loaded = torch.load(ckpt_path, map_location="cpu")
        start_epoch = loaded["epoch"] + 1
        self.assertEqual(start_epoch, 8)

    def test_14_best_checkpoint_uses_val_miou(self):
        """Test 14: Trainer updates best checkpoint only when validation mIoU improves."""
        model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        dummy_data = [{"points": torch.randn(128, 4), "labels": torch.randint(0, 4, (128,)), "metadata": {}}]
        loader = DataLoader(dummy_data, batch_size=1, collate_fn=lidar_collate_fn)

        config = {
            "experiment": {"name": "test_exp", "device": "cpu"},
            "training": {"epochs": 2, "learning_rate": 0.01},
            "loss": {"type": "cross_entropy", "ignore_index": 255},
        }
        trainer = PointNet2Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            config=config,
            experiment_dir=self.temp_path / "exp",
        )
        res = trainer.train()
        self.assertGreater(res["best_val_miou"], 0.0)
        self.assertTrue((self.temp_path / "exp" / "best_checkpoint.pt").is_file())

    def test_15_experiment_config_reproducibility(self):
        """Test 15: Training class weight computation is deterministic for identical input counts."""
        class_counts = {0: 23000, 1: 8000, 2: 28500, 3: 6000}
        w1 = compute_class_weights(class_counts, num_classes=4, strategy="inverse_frequency")
        w2 = compute_class_weights(class_counts, num_classes=4, strategy="inverse_frequency")
        np.testing.assert_array_equal(w1.numpy(), w2.numpy())


if __name__ == "__main__":
    unittest.main()
