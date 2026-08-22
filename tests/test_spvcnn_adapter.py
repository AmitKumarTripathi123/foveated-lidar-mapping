"""Unit tests for SPVCNN Input and Label Adapters."""

import unittest
import numpy as np
import torch

from src.types import SuperClass
from phase2.models.spvcnn_adapter import SPVCNNInputAdapter, SPVCNNLabelAdapter


class TestSPVCNNAdapter(unittest.TestCase):
    def test_01_input_adapter_preparation(self):
        """Tests input coordinate quantization, unique voxel hashing, and tensor shapes."""
        adapter = SPVCNNInputAdapter(voxel_size=0.05)

        pts = np.array([
            [1.01, 2.01, -1.5, 0.5],
            [1.02, 2.02, -1.5, 0.6],  # Same voxel as pt 0
            [5.00, 5.00, 0.0, 0.8],   # Different voxel
        ], dtype=np.float32)

        bundle = adapter.prepare_input(pts)
        self.assertEqual(bundle["num_points"], 3)
        self.assertEqual(bundle["num_voxels"], 2)
        self.assertEqual(bundle["features"].shape, (3, 4))
        self.assertEqual(bundle["point_to_voxel_idx"].shape, (3,))
        self.assertEqual(bundle["point_to_voxel_idx"][0].item(), bundle["point_to_voxel_idx"][1].item())

    def test_02_label_adapter_remapping(self):
        """Tests 19-class SemanticKITTI to 4-class SIH super-class conversion."""
        adapter = SPVCNNLabelAdapter(native_source="semantickitti")

        # Native classes: 8 (road), 10 (sidewalk), 12 (building), 0 (car)
        native = np.array([8, 10, 12, 0], dtype=np.int64)
        sih_classes = adapter.remap_predictions(native)

        self.assertEqual(sih_classes[0], SuperClass.DRIVABLE_TERRAIN)
        self.assertEqual(sih_classes[1], SuperClass.NON_DRIVABLE_TERRAIN)
        self.assertEqual(sih_classes[2], SuperClass.STATIC_OBSTACLE)
        self.assertEqual(sih_classes[3], SuperClass.DYNAMIC_OBJECT)

    def test_03_label_adapter_process_logits(self):
        """Tests converting raw logits tensor to super-classes, probability distributions, and confidence."""
        adapter = SPVCNNLabelAdapter(native_source="semantickitti")

        # 2 points, 19 classes
        logits = torch.zeros((2, 19), dtype=torch.float32)
        logits[0, 8] = 10.0  # Point 0 is strongly road (class 8 -> superclass 0)
        logits[1, 12] = 10.0 # Point 1 is strongly building (class 12 -> superclass 2)

        sih_preds, super_probs, confs = adapter.process_logits(logits)

        self.assertEqual(sih_preds[0], 0)
        self.assertEqual(sih_preds[1], 2)
        self.assertEqual(super_probs.shape, (2, 4))
        self.assertAlmostEqual(super_probs[0, 0], 1.0, places=2)
        self.assertAlmostEqual(super_probs[1, 2], 1.0, places=2)
        self.assertAlmostEqual(confs[0], 1.0, places=2)
        self.assertAlmostEqual(confs[1], 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
