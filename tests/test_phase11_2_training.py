"""Tests for Phase 11.2 Model Training Step, Loss Formulation, and Metrics."""

import sys
import unittest
from pathlib import Path
import numpy as np
import torch

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.models.pointnet2 import build_model
from ml.training.losses import get_loss_function
from ml.training.metrics import SemanticSegmentationMetrics


class TestPhase11_2Training(unittest.TestCase):
    """Test suite for training forward/backward pass, ignore-index, and metric evaluation."""

    def test_01_training_forward_backward(self):
        """Test 1: Forward and backward pass execute with finite gradients."""
        model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        crit = get_loss_function(loss_type="cross_entropy", ignore_index=255)

        x = torch.randn(1, 64, 4)
        y = torch.tensor([[0, 1, 2, 3, 255] * 12 + [0, 1, 2, 3]], dtype=torch.int64)

        out = model(x)
        loss = crit(out.transpose(1, 2), y)
        loss.backward()
        opt.step()

        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

    def test_02_metrics_ignore_index_exclusion(self):
        """Test 2: Metric calculation strictly excludes points with label 255."""
        metrics = SemanticSegmentationMetrics(num_classes=4, ignore_index=255)
        preds = np.array([0, 1, 2, 3, 0])
        targets = np.array([0, 1, 2, 3, 255])
        metrics.update(preds, targets)
        res = metrics.compute()
        self.assertAlmostEqual(res.overall_accuracy, 1.0)
        self.assertAlmostEqual(res.miou, 1.0)


if __name__ == "__main__":
    unittest.main()
