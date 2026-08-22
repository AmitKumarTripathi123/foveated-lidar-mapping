"""Unit tests for SPVCNN architecture and checkpoint handling."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
import torch

from phase2.models.spvcnn import SPVCNN, build_spvcnn, load_spvcnn_checkpoint


class TestSPVCNN(unittest.TestCase):
    def test_01_spvcnn_construction(self):
        """Test SPVCNN constructs successfully with expected parameters and layers."""
        model = SPVCNN(num_classes=19, in_channels=4, base_channels=32)
        total_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(total_params, 100000)
        self.assertEqual(model.num_classes, 19)
        self.assertEqual(model.in_channels, 4)

    def test_02_spvcnn_forward_pass(self):
        """Test SPVCNN forward pass executes cleanly on CPU."""
        model = SPVCNN(num_classes=19, in_channels=4, base_channels=32)
        model.eval()

        n_pts = 100
        n_voxels = 40
        features = torch.randn(n_pts, 4)
        pt_to_voxel = torch.randint(0, n_voxels, (n_pts,))

        with torch.no_grad():
            logits = model(features, pt_to_voxel, n_voxels)

        self.assertEqual(logits.shape, (n_pts, 19))
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())

    def test_03_checkpoint_load_save(self):
        """Test SPVCNN loads and saves checkpoints with exact weights."""
        model = SPVCNN(num_classes=19, in_channels=4, base_channels=32)
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = os.path.join(tmp_dir, "test_spvcnn.pt")
            torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

            new_model = SPVCNN(num_classes=19, in_channels=4, base_channels=32)
            report = load_spvcnn_checkpoint(new_model, ckpt_path, strict=True)
            self.assertEqual(len(report["missing_keys"]), 0)
            self.assertEqual(len(report["unexpected_keys"]), 0)

            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                self.assertTrue(torch.allclose(p1, p2))

    def test_04_build_spvcnn_factory(self):
        """Test build_spvcnn helper constructs and loads model."""
        model = build_spvcnn(num_classes=4, in_channels=4, device="cpu")
        self.assertEqual(model.num_classes, 4)


if __name__ == "__main__":
    unittest.main()
