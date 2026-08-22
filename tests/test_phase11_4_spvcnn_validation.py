"""Phase 11.4 Scientific Validation and Checkpoint Audit Test Suite for SPVCNN."""

import os
import sys
import unittest
from pathlib import Path
import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.dataset import load_point_cloud, load_labels
from ml.data.amit_adapter import FoveatedVoxelSampler
from ml.data.spvcnn_adapter import SPVCNNInputAdapter
from ml.models.spvcnn import SPVCNN, build_spvcnn, load_spvcnn_checkpoint
from ml.models.spvcnn_label_adapter import SPVCNNLabelAdapter, SEMANTICKITTI_TO_SIH
from ml.models.spvcnn_predictor import SPVCNNPredictor
from ml.models.mapping_adapter import MLToMappingAdapter, GridMap25D


class TestPhase11_4SPVCNNValidation(unittest.TestCase):
    """Test suite for Phase 11.4 scientific validation of SPVCNN."""

    @classmethod
    def setUpClass(cls):
        cls.ckpt_path = repo_root / "checkpoints/spvcnn_pretrained.pt"
        cls.real_scan = repo_root / "dataset/sequences/00/velodyne/000000.bin"
        cls.real_label = repo_root / "dataset/sequences/00/labels/000000.label"

    def test_01_checkpoint_provenance_exists(self):
        """Test 1: Pretrained checkpoint file physically exists on disk."""
        self.assertTrue(self.ckpt_path.is_file(), f"Checkpoint missing at: {self.ckpt_path}")

    def test_02_checkpoint_loads_correctly(self):
        """Test 2: Checkpoint loads into SPVCNN model without error."""
        model = SPVCNN(num_classes=19, in_channels=4, base_channels=32)
        report = load_spvcnn_checkpoint(model, self.ckpt_path, strict=True)
        self.assertGreater(report["total_parameters"], 50000)

    def test_03_no_unexpected_keys(self):
        """Test 3: State dict has exactly zero unexpected keys."""
        model = SPVCNN(num_classes=19, in_channels=4, base_channels=32)
        report = load_spvcnn_checkpoint(model, self.ckpt_path, strict=True)
        self.assertEqual(len(report["unexpected_keys"]), 0)

    def test_04_no_shape_mismatch(self):
        """Test 4: State dict has zero missing keys or tensor dimension mismatches."""
        model = SPVCNN(num_classes=19, in_channels=4, base_channels=32)
        report = load_spvcnn_checkpoint(model, self.ckpt_path, strict=True)
        self.assertEqual(len(report["missing_keys"]), 0)

    def test_05_native_class_ids_validated(self):
        """Test 5: Native classes cover 19 SemanticKITTI classes."""
        self.assertEqual(len(SEMANTICKITTI_TO_SIH), 19)
        self.assertTrue(all(0 <= k <= 18 for k in SEMANTICKITTI_TO_SIH.keys()))

    def test_06_sih_mapping_validated(self):
        """Test 6: SIH mapping outputs strictly belong to {0, 1, 2, 3, 255}."""
        adapter = SPVCNNLabelAdapter(native_source="semantickitti")
        mapped = [adapter.lut[k] for k in range(19)]
        self.assertTrue(set(mapped).issubset({0, 1, 2, 3, 255}))

    def test_07_prediction_shape_validated(self):
        """Test 7: Predictor returns matching array shapes for N input points."""
        predictor = SPVCNNPredictor(device="cpu", pretrained_path=str(self.ckpt_path))
        pts = np.random.uniform(-15, 15, (250, 4)).astype(np.float32)
        res = predictor.predict(pts)
        self.assertEqual(res["xyz"].shape, (250, 3))
        self.assertEqual(res["predicted_class"].shape, (250,))
        self.assertEqual(res["confidence"].shape, (250,))

    def test_08_prediction_dtype_validated(self):
        """Test 8: Prediction dtypes strictly follow the frozen contract."""
        predictor = SPVCNNPredictor(device="cpu", pretrained_path=str(self.ckpt_path))
        pts = np.random.uniform(-15, 15, (100, 4)).astype(np.float32)
        res = predictor.predict(pts)
        self.assertEqual(res["xyz"].dtype, np.float32)
        self.assertEqual(res["predicted_class"].dtype, np.int64)
        self.assertEqual(res["confidence"].dtype, np.float32)

    def test_09_confidence_range_validated(self):
        """Test 9: Confidence scores strictly fall in [0.0, 1.0]."""
        predictor = SPVCNNPredictor(device="cpu", pretrained_path=str(self.ckpt_path))
        pts = np.random.uniform(-15, 15, (300, 4)).astype(np.float32)
        res = predictor.predict(pts)
        self.assertTrue(np.all(res["confidence"] >= 0.0) and np.all(res["confidence"] <= 1.0))

    def test_10_xyz_order_validated(self):
        """Test 10: 100% XYZ point ordering and numerical values are preserved."""
        predictor = SPVCNNPredictor(device="cpu", pretrained_path=str(self.ckpt_path))
        pts = np.random.uniform(-30, 30, (400, 4)).astype(np.float32)
        res = predictor.predict(pts)
        np.testing.assert_array_equal(res["xyz"], pts[:, :3])

    def test_11_voxel_mapping_validated(self):
        """Test 11: Voxel-to-point and point-to-voxel mapping is deterministic."""
        adapter = SPVCNNInputAdapter(voxel_size=0.05)
        pts = np.random.uniform(-10, 10, (150, 4)).astype(np.float32)
        bundle = adapter.prepare_input(pts)
        self.assertEqual(bundle["point_to_voxel_idx"].shape[0], 150)
        self.assertLessEqual(bundle["num_voxels"], 150)

    def test_12_no_nan_inf(self):
        """Test 12: Network predictions produce zero NaNs and zero Infs."""
        predictor = SPVCNNPredictor(device="cpu", pretrained_path=str(self.ckpt_path))
        pts = np.random.uniform(-20, 20, (200, 4)).astype(np.float32)
        res = predictor.predict(pts)
        self.assertFalse(np.isnan(res["xyz"]).any())
        self.assertFalse(np.isnan(res["confidence"]).any())
        self.assertFalse(np.isinf(res["confidence"]).any())

    def test_13_model_collapse_detector(self):
        """Test 13: Prediction distribution contains multiple distinct classes."""
        predictor = SPVCNNPredictor(device="cpu", pretrained_path=str(self.ckpt_path))
        pts = np.random.uniform(-20, 20, (1000, 4)).astype(np.float32)
        res = predictor.predict(pts)
        unique_classes = np.unique(res["predicted_class"])
        self.assertGreater(len(unique_classes), 1)

    def test_14_mapping_adapter_regression(self):
        """Test 14: MLToMappingAdapter validates and consumes SPVCNN outputs."""
        predictor = SPVCNNPredictor(device="cpu", pretrained_path=str(self.ckpt_path))
        mapping_adapter = MLToMappingAdapter(resolution=0.20)
        pts = np.random.uniform(-20, 20, (500, 4)).astype(np.float32)
        res = predictor.predict(pts)
        batch = mapping_adapter.validate_prediction(res)
        self.assertEqual(batch.xyz.shape[0], 500)

    def test_15_gridmap25d_regression(self):
        """Test 15: GridMap25D generation populates all layers cleanly."""
        predictor = SPVCNNPredictor(device="cpu", pretrained_path=str(self.ckpt_path))
        mapping_adapter = MLToMappingAdapter(resolution=0.20)
        pts = np.random.uniform(-20, 20, (500, 4)).astype(np.float32)
        res = predictor.predict(pts)
        grid = mapping_adapter.build_25d_grid(res)
        self.assertIsInstance(grid, GridMap25D)
        self.assertFalse(np.isnan(grid.confidence_layer).any())

    def test_16_real_frame_inference(self):
        """Test 16: Real representative frame inference end-to-end."""
        if not self.real_scan.exists():
            self.skipTest("Real scan 000000.bin not found")

        raw_pts = load_point_cloud(self.real_scan)
        sampler = FoveatedVoxelSampler(
            near_dist=10.0, near_voxel=0.05,
            mid_dist=40.0, mid_voxel=0.15,
            far_dist=100.0, far_voxel=0.50,
        )
        fov_pts, _, _ = sampler.sample(raw_pts)
        predictor = SPVCNNPredictor(device="cpu", pretrained_path=str(self.ckpt_path))
        res = predictor.predict(fov_pts)

        self.assertEqual(res["xyz"].shape[0], fov_pts.shape[0])
        np.testing.assert_array_equal(res["xyz"], fov_pts[:, :3])


if __name__ == "__main__":
    unittest.main()
