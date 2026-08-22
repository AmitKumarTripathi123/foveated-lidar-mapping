"""Unit tests for SPVCNN Input and Label Adapters."""

import sys
import unittest
from pathlib import Path
import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.spvcnn_adapter import SPVCNNInputAdapter
from ml.models.spvcnn_label_adapter import SPVCNNLabelAdapter, SEMANTICKITTI_TO_SIH


class TestSPVCNNAdapter(unittest.TestCase):
    """Test suite for SPVCNN input adapter and label remapping."""

    def setUp(self):
        self.input_adapter = SPVCNNInputAdapter(voxel_size=0.05)
        self.label_adapter = SPVCNNLabelAdapter(native_source="semantickitti")

    def test_01_input_adapter_quantization(self):
        """Test 1: Input adapter quantizes coordinates and extracts voxels."""
        pts = np.array([
            [1.00, 1.00, 0.50, 0.1],
            [1.01, 1.01, 0.51, 0.2],  # Same 5cm voxel as point 0
            [5.00, 5.00, 0.50, 0.3],  # Different voxel
        ], dtype=np.float32)

        bundle = self.input_adapter.prepare_input(pts)
        self.assertEqual(bundle["num_points"], 3)
        self.assertEqual(bundle["num_voxels"], 2)
        self.assertEqual(bundle["point_to_voxel_idx"].shape[0], 3)
        self.assertEqual(bundle["voxel_to_point_idx"].shape[0], 2)

    def test_02_point_order_preservation(self):
        """Test 2: Original XYZ coordinates and point order are 100% preserved."""
        pts = np.random.uniform(-20, 20, (500, 4)).astype(np.float32)
        bundle = self.input_adapter.prepare_input(pts)
        np.testing.assert_array_equal(bundle["raw_xyz"], pts[:, :3])
        np.testing.assert_array_equal(bundle["xyz"].numpy(), pts[:, :3])

    def test_03_voxel_projection(self):
        """Test 3: Voxel-to-point prediction projection works accurately."""
        pts = np.array([
            [1.00, 1.00, 0.50, 0.1],
            [1.01, 1.01, 0.51, 0.2],
            [5.00, 5.00, 0.50, 0.3],
        ], dtype=np.float32)
        bundle = self.input_adapter.prepare_input(pts)

        # 2 voxels with class 8 (road) and class 12 (building)
        voxel_preds = np.array([8, 12])
        point_preds = self.input_adapter.project_voxel_predictions_to_points(
            voxel_preds, bundle["point_to_voxel_idx"].numpy()
        )
        self.assertEqual(len(point_preds), 3)
        self.assertEqual(point_preds[0], 8)
        self.assertEqual(point_preds[1], 8)
        self.assertEqual(point_preds[2], 12)

    def test_04_semantickitti_to_sih_remapping(self):
        """Test 4: SemanticKITTI 19 classes map strictly into {0, 1, 2, 3}."""
        native_classes = np.array([0, 8, 10, 12, 14, 255])
        sih_classes = self.label_adapter.remap_predictions(native_classes)

        # 0(car)->3, 8(road)->0, 10(sidewalk)->1, 12(building)->2, 14(vegetation)->2, 255->255
        expected = np.array([3, 0, 1, 2, 2, 255])
        np.testing.assert_array_equal(sih_classes, expected)

    def test_05_unknown_class_handling(self):
        """Test 5: Unmapped class IDs gracefully fallback to 255 (ignore)."""
        unmapped = np.array([999, -5, 100])
        sih_classes = self.label_adapter.remap_predictions(unmapped)
        self.assertTrue(np.all(sih_classes == 255))


if __name__ == "__main__":
    unittest.main()

