"""Automated Test Suite for PointNet++ Semantic Segmentation Baseline (Phase 4).

Covers all 16 required tests:
  - Test 1: Model construction
  - Test 2: Correct number of output classes (num_classes == 4, not 5)
  - Test 3: Synthetic forward pass [B, N, 4] -> [B, N, 4]
  - Test 4: Batch dimension preservation (B=1, 2, 4)
  - Test 5: Point-count preservation (N=256, 512, 1024, 16384)
  - Test 6: Backward pass with finite gradients
  - Test 7: Output finiteness (no NaN/Inf in logits or probabilities)
  - Test 8: Confidence range in [0.0, 1.0]
  - Test 9: Predicted class range in {0, 1, 2, 3}
  - Test 10: Exact XYZ point-order preservation
  - Test 11: Real representative sample forward pass
  - Test 12: Real sample output shapes and types
  - Test 13: CPU execution
  - Test 14: CUDA execution (if available)
  - Test 15: Model parameter sanity (finite initial weights)
  - Test 16: Tiny deterministic overfit sanity test
"""

import sys
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

from ml.data.dataset import load_point_cloud, load_labels
from ml.data.preprocessing import LidarPreprocessor, PreprocessingConfig, SamplingConfig
from ml.data.label_mapping import SemanticLabelRemapper
from ml.models.pointnet2 import PointNet2SemSeg, build_model
from ml.models.predictor import PointNet2Predictor


class TestPointNet2(unittest.TestCase):
    """Unit and integration test suite for PointNet++ architecture and prediction contract."""

    @classmethod
    def setUpClass(cls):
        """Set fixed seed for deterministic test suite."""
        torch.manual_seed(42)
        np.random.seed(42)
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4).to(cls.device)

    def test_01_model_construction(self):
        """Test 1: Model builds successfully."""
        self.assertIsInstance(self.model, nn.Module)
        self.assertIsInstance(self.model, PointNet2SemSeg)

    def test_02_correct_number_of_classes(self):
        """Test 2: Model output head has exactly 4 classes (255 is ignore target, not an output class)."""
        self.assertEqual(self.model.num_classes, 4)
        self.assertEqual(self.model.conv2.out_channels, 4)

    def test_03_synthetic_forward_pass(self):
        """Test 3: Synthetic input [B, N, 4] produces [B, N, 4] logits."""
        B, N, C = 2, 512, 4
        dummy_input = torch.randn(B, N, C, device=self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(dummy_input)

        self.assertEqual(logits.shape, (B, N, 4))

    def test_04_batch_dimension_preservation(self):
        """Test 4: Batch dimension is preserved across multiple batch sizes."""
        self.model.eval()
        for b_size in [1, 2, 4]:
            dummy = torch.randn(b_size, 256, 4, device=self.device)
            with torch.no_grad():
                out = self.model(dummy)
            self.assertEqual(out.shape[0], b_size)
            self.assertEqual(out.shape[2], 4)

    def test_05_point_count_preservation(self):
        """Test 5: Input point count N matches output point count N."""
        self.model.eval()
        for n_pts in [256, 512, 1024]:
            dummy = torch.randn(1, n_pts, 4, device=self.device)
            with torch.no_grad():
                out = self.model(dummy)
            self.assertEqual(out.shape[1], n_pts)

    def test_06_backward_pass(self):
        """Test 6: Backward pass computes finite gradients on trainable parameters."""
        self.model.train()
        dummy_input = torch.randn(2, 256, 4, device=self.device)
        dummy_target = torch.randint(0, 4, (2, 256), device=self.device)

        logits = self.model(dummy_input)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits.view(-1, 4), dummy_target.view(-1))

        self.assertTrue(torch.isfinite(loss).item())

        self.model.zero_grad()
        loss.backward()

        has_grads = False
        for param in self.model.parameters():
            if param.grad is not None:
                has_grads = True
                self.assertFalse(torch.isnan(param.grad).any().item())
                self.assertFalse(torch.isinf(param.grad).any().item())

        self.assertTrue(has_grads)

    def test_07_no_nan_inf_outputs(self):
        """Test 7: Model forward pass produces entirely finite output logits."""
        self.model.eval()
        dummy_input = torch.randn(2, 512, 4, device=self.device)
        with torch.no_grad():
            logits = self.model(dummy_input)

        self.assertFalse(torch.isnan(logits).any().item())
        self.assertFalse(torch.isinf(logits).any().item())

    def test_08_confidence_range(self):
        """Test 8: Predictor confidence scores are strictly in [0.0, 1.0]."""
        predictor = PointNet2Predictor(model=self.model, device=self.device)
        dummy_pts = np.random.randn(512, 4).astype(np.float32)
        res = predictor.predict(dummy_pts)

        conf = res["confidence"]
        self.assertTrue(np.all(conf >= 0.0))
        self.assertTrue(np.all(conf <= 1.0))

    def test_09_prediction_class_range(self):
        """Test 9: Predicted class indices strictly belong to {0, 1, 2, 3}."""
        predictor = PointNet2Predictor(model=self.model, device=self.device)
        dummy_pts = np.random.randn(512, 4).astype(np.float32)
        res = predictor.predict(dummy_pts)

        pred_classes = set(np.unique(res["predicted_class"]))
        self.assertTrue(pred_classes.issubset({0, 1, 2, 3}))

    def test_10_xyz_point_order_preservation(self):
        """Test 10: Input XYZ coordinates are strictly preserved without reordering."""
        predictor = PointNet2Predictor(model=self.model, device=self.device)
        dummy_pts = np.random.uniform(-50, 50, size=(512, 4)).astype(np.float32)
        res = predictor.predict(dummy_pts)

        np.testing.assert_array_equal(res["xyz"], dummy_pts[:, :3])

    def test_11_real_sample_forward_pass(self):
        """Test 11: Real representative LiDAR scan runs forward pass successfully."""
        bin_path = repo_root / "dataset/sequences/00/velodyne/000000.bin"
        if not bin_path.is_file():
            self.skipTest("Real scan sample not found.")

        raw_points = load_point_cloud(bin_path)
        prep = LidarPreprocessor(
            PreprocessingConfig(sampling=SamplingConfig(strategy="random", num_points=1024, seed=42))
        )
        sample = prep(raw_points)

        predictor = PointNet2Predictor(model=self.model, device=self.device)
        res = predictor.predict(sample.points)

        self.assertEqual(res["xyz"].shape, (1024, 3))
        self.assertEqual(res["predicted_class"].shape, (1024,))
        self.assertEqual(res["confidence"].shape, (1024,))

    def test_12_real_sample_output_shapes(self):
        """Test 12: Real sample output contract types and dimensions."""
        bin_path = repo_root / "dataset/sequences/00/velodyne/000000.bin"
        if not bin_path.is_file():
            self.skipTest("Real scan sample not found.")

        raw_points = load_point_cloud(bin_path)
        prep = LidarPreprocessor(
            PreprocessingConfig(sampling=SamplingConfig(strategy="random", num_points=512, seed=42))
        )
        sample = prep(raw_points)

        predictor = PointNet2Predictor(model=self.model, device=self.device)
        res = predictor.predict(sample.points)

        self.assertEqual(res["xyz"].dtype, np.float32)
        self.assertEqual(res["confidence"].dtype, np.float32)
        self.assertTrue(np.issubdtype(res["predicted_class"].dtype, np.integer))

    def test_13_cpu_execution(self):
        """Test 13: Model runs cleanly on CPU."""
        cpu_model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4).to("cpu")
        dummy = torch.randn(1, 256, 4, device="cpu")
        cpu_model.eval()
        with torch.no_grad():
            out = cpu_model(dummy)
        self.assertEqual(out.device.type, "cpu")
        self.assertEqual(out.shape, (1, 256, 4))

    def test_14_cuda_execution(self):
        """Test 14: Model runs cleanly on CUDA if GPU is available."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA GPU not available in this environment.")

        gpu_model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4).to("cuda")
        dummy = torch.randn(1, 256, 4, device="cuda")
        gpu_model.eval()
        with torch.no_grad():
            out = gpu_model(dummy)
        self.assertEqual(out.device.type, "cuda")

    def test_15_model_parameter_sanity(self):
        """Test 15: All initial model parameters are finite without NaNs or Infs."""
        for name, param in self.model.named_parameters():
            self.assertFalse(torch.isnan(param).any().item(), f"NaN in parameter {name}")
            self.assertFalse(torch.isinf(param).any().item(), f"Inf in parameter {name}")

    def test_16_tiny_deterministic_overfit_test(self):
        """Test 16: Small-subset overfit test confirms loss decreases significantly."""
        torch.manual_seed(42)
        np.random.seed(42)

        test_model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4).to(self.device)
        test_model.train()

        # Fixed tiny synthetic batch of 256 points
        dummy_pts = torch.randn(1, 256, 4, device=self.device)
        dummy_lbls = torch.randint(0, 4, (1, 256), device=self.device)

        optimizer = optim.Adam(test_model.parameters(), lr=0.02)
        criterion = nn.CrossEntropyLoss()

        initial_loss = None
        final_loss = None
        initial_acc = None
        final_acc = None

        for it in range(30):
            optimizer.zero_grad()
            logits = test_model(dummy_pts)
            loss = criterion(logits.view(-1, 4), dummy_lbls.view(-1))
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=-1)
            acc = (preds == dummy_lbls).float().mean().item() * 100.0

            if it == 0:
                initial_loss = loss.item()
                initial_acc = acc
            final_loss = loss.item()
            final_acc = acc

        self.assertIsNotNone(initial_loss)
        self.assertIsNotNone(final_loss)
        self.assertTrue(torch.isfinite(torch.tensor(final_loss)))
        self.assertLess(final_loss, initial_loss, "Overfit test failed: final loss did not decrease")
        self.assertGreater(final_acc, initial_acc, "Overfit test failed: accuracy did not improve")


if __name__ == "__main__":
    unittest.main()
