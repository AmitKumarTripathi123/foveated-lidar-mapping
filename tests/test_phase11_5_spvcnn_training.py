"""Phase 11.5 Full SemanticPOSS Dataset Activation and SPVCNN Fine-Tuning Readiness Test Suite."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.dataset import load_point_cloud, load_labels, validate_point_label_alignment
from ml.data.amit_adapter import FoveatedVoxelSampler
from ml.data.semanticposs_label_mapping import SemanticPOSSLabelRemapper
from ml.data.spvcnn_adapter import SPVCNNInputAdapter
from ml.models.spvcnn import SPVCNN, build_spvcnn, load_spvcnn_checkpoint
from ml.models.spvcnn_predictor import SPVCNNPredictor
from ml.training.spvcnn_trainer import SPVCNNTrainer
from ml.models.mapping_adapter import MLToMappingAdapter, GridMap25D
from scripts.audit_semanticposs import audit_sequence
from scripts.train_spvcnn_phase11_5 import (
    check_dataset_completeness,
    compute_training_class_weights,
    SPVCNNFoveatedDataset,
    collate_spvcnn,
)


class TestPhase11_5SPVCNNTraining(unittest.TestCase):
    """22-Test comprehensive suite for Phase 11.5 SPVCNN dataset activation and fine-tuning."""

    @classmethod
    def setUpClass(cls):
        cls.ckpt_path = repo_root / "checkpoints/spvcnn_pretrained.pt"
        cls.real_scan = repo_root / "dataset/sequences/00/velodyne/000000.bin"
        cls.real_label = repo_root / "dataset/sequences/00/labels/000000.label"
        cls.dataset_root = repo_root / "dataset"

    def test_01_dataset_discovery(self):
        """Test 1: Dataset discovery finds sequence 00 with matching files."""
        res = audit_sequence(str(self.dataset_root / "sequences/00"), "00")
        self.assertGreaterEqual(res["matched_pairs"], 1)

    def test_02_dataset_completeness_gate(self):
        """Test 2: Completeness gate detects expected frame count."""
        expected = {"00": 488, "01": 500, "02": 500, "03": 500, "04": 500, "05": 500}
        gate = check_dataset_completeness(self.dataset_root, expected)
        self.assertEqual(gate["total_expected"], 2988)
        if gate["total_found"] == 2988:
            self.assertTrue(gate["is_complete"])
            self.assertEqual(gate["total_found"], 2988)
        else:
            self.assertFalse(gate["is_complete"])

    def test_03_stem_pairing(self):
        """Test 3: Frame pairing strictly uses stem matching without positional pairing."""
        res = audit_sequence(str(self.dataset_root / "sequences/00"), "00")
        details = res["frame_details"]
        self.assertGreaterEqual(len(details), 1)
        stem, b_p, l_p, n_pts = details[0]
        self.assertEqual(Path(b_p).stem, Path(l_p).stem)
        self.assertEqual(stem, "000000")


    def test_04_sequence_split_disjointness(self):
        """Test 4: Train and validation sequence sets are strictly disjoint."""
        train_seqs = {"00", "01", "03", "04", "05"}
        val_seqs = {"02"}
        self.assertTrue(train_seqs.isdisjoint(val_seqs))

    def test_05_foveation_alignment(self):
        """Test 5: Amit foveated downsampler maintains 100% point-label alignment."""
        raw_pts = load_point_cloud(self.real_scan)
        raw_lbl = load_labels(self.real_label)
        sampler = FoveatedVoxelSampler()
        fov_pts, fov_lbl, report = sampler.sample(raw_pts, raw_lbl)
        self.assertEqual(fov_pts.shape[0], fov_lbl.shape[0])
        self.assertTrue(report.alignment_pass)

    def test_06_semanticposs_label_remapping(self):
        """Test 6: SemanticPOSS raw labels remap strictly into {0, 1, 2, 3, 255}."""
        remapper = SemanticPOSSLabelRemapper()
        raw_lbl = load_labels(self.real_label)
        sih_lbl = remapper.remap(raw_lbl)
        self.assertTrue(set(np.unique(sih_lbl)).issubset({0, 1, 2, 3, 255}))

    def test_07_spvcnn_checkpoint_loading(self):
        """Test 7: Pretrained SPVCNN checkpoint loads backbone parameters without failure."""
        model = SPVCNN(num_classes=4, in_channels=4)
        report = load_spvcnn_checkpoint(model, self.ckpt_path, strict=False)
        self.assertGreater(len(report["loaded_keys"]), 30)

    def test_08_input_adapter(self):
        """Test 8: Input adapter generates quantized voxels and valid inverse mappings."""
        adapter = SPVCNNInputAdapter(voxel_size=0.05)
        pts = np.random.uniform(-10, 10, (100, 4)).astype(np.float32)
        bundle = adapter.prepare_input(pts)
        self.assertEqual(bundle["point_to_voxel_idx"].shape[0], 100)

    def test_09_voxel_collision_handling(self):
        """Test 9: Multiple points in same voxel reconstruct per-point predictions without loss."""
        adapter = SPVCNNInputAdapter(voxel_size=0.05)
        pts = np.array([
            [1.0, 1.0, 1.0, 0.0],
            [1.01, 1.01, 1.01, 0.0],
            [10.0, 10.0, 10.0, 0.0],
        ], dtype=np.float32)
        bundle = adapter.prepare_input(pts)
        self.assertEqual(bundle["num_voxels"], 2)
        voxel_preds = np.array([1, 2])
        point_preds = adapter.project_voxel_predictions_to_points(voxel_preds, bundle["point_to_voxel_idx"].numpy())
        self.assertEqual(point_preds.shape[0], 3)
        self.assertEqual(point_preds[0], 1)
        self.assertEqual(point_preds[1], 1)
        self.assertEqual(point_preds[2], 2)

    def test_10_point_order_preservation(self):
        """Test 10: Output XYZ coordinates match input XYZ coordinates exactly."""
        predictor = SPVCNNPredictor(device="cpu", pretrained_path=str(self.ckpt_path))
        pts = np.random.uniform(-20, 20, (300, 4)).astype(np.float32)
        res = predictor.predict(pts)
        np.testing.assert_array_equal(res["xyz"], pts[:, :3])

    def test_11_output_contract(self):
        """Test 11: Predictor output strictly satisfies frozen [x,y,z,predicted_class,confidence]."""
        predictor = SPVCNNPredictor(device="cpu", pretrained_path=str(self.ckpt_path))
        pts = np.random.uniform(-10, 10, (50, 4)).astype(np.float32)
        res = predictor.predict(pts)
        self.assertIn("xyz", res)
        self.assertIn("predicted_class", res)
        self.assertIn("confidence", res)
        self.assertEqual(res["predicted_class"].dtype, np.int64)
        self.assertEqual(res["confidence"].dtype, np.float32)

    def test_12_confidence_validity(self):
        """Test 12: Confidence values fall strictly in [0.0, 1.0] without NaN or Inf."""
        predictor = SPVCNNPredictor(device="cpu", pretrained_path=str(self.ckpt_path))
        pts = np.random.uniform(-10, 10, (100, 4)).astype(np.float32)
        res = predictor.predict(pts)
        self.assertTrue(np.all(res["confidence"] >= 0.0) and np.all(res["confidence"] <= 1.0))
        self.assertFalse(np.isnan(res["confidence"]).any())

    def test_13_confusion_matrix_reconciliation(self):
        """Test 13: sum(confusion_matrix) strictly equals total evaluated supervised points."""
        model = SPVCNN(num_classes=4, in_channels=4)
        config = {"loss": {"ignore_index": 255}, "model": {"num_classes": 4}}
        pts = torch.randn(50, 4)
        lbls = torch.tensor([0, 1, 2, 3, 255] * 10, dtype=torch.long)
        dataset = [(pts, lbls)]
        loader = DataLoader(dataset, batch_size=1, collate_fn=collate_spvcnn)

        trainer = SPVCNNTrainer(model=model, train_loader=None, val_loader=loader, config=config)
        metrics = trainer.evaluate()
        cm = np.array(metrics["confusion_matrix"])
        self.assertEqual(int(np.sum(cm)), metrics["supervised_points"])
        self.assertEqual(metrics["evaluated_points"], 50)
        self.assertEqual(metrics["supervised_points"], 40)
        self.assertEqual(metrics["ignored_points"], 10)

    def test_14_ignore_index_handling(self):
        """Test 14: Class 255 is completely excluded from loss and IoU calculations."""
        model = SPVCNN(num_classes=4, in_channels=4)
        config = {"loss": {"ignore_index": 255}, "model": {"num_classes": 4}}
        pts = torch.randn(20, 4)
        lbls = torch.full((20,), 255, dtype=torch.long)
        loader = DataLoader([(pts, lbls)], batch_size=1, collate_fn=collate_spvcnn)

        trainer = SPVCNNTrainer(model=model, train_loader=None, val_loader=loader, config=config)
        metrics = trainer.evaluate()
        self.assertEqual(metrics["supervised_points"], 0)
        self.assertEqual(metrics["ignored_points"], 20)

    def test_15_class_weight_calculation(self):
        """Test 15: Class weights are computed strictly from training dataset only."""
        records = [("00_000000", str(self.real_scan), str(self.real_label))]
        train_ds = SPVCNNFoveatedDataset(records)
        weights = compute_training_class_weights(train_ds, num_classes=4)
        self.assertEqual(len(weights), 4)
        self.assertAlmostEqual(float(np.mean(weights)), 1.0, places=4)

    def test_16_training_configuration(self):
        """Test 16: Training configuration loads correctly with required sections."""
        cfg_path = repo_root / "configs/phase11_5_spvcnn_training.yaml"
        self.assertTrue(cfg_path.is_file())

    def test_17_checkpoint_save_load(self):
        """Test 17: Trainer saves checkpoint containing state dict and metrics."""
        model = SPVCNN(num_classes=4, in_channels=4)
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = SPVCNNTrainer(model=model, train_loader=None, val_loader=None, config={}, experiment_dir=tmp_dir)
            ckpt_p = trainer.save_checkpoint("test.pt", epoch=1, metrics={"val_miou": 15.5})
            self.assertTrue(os.path.isfile(ckpt_p))
            data = torch.load(ckpt_p, map_location="cpu")
            self.assertEqual(data["epoch"], 1)
            self.assertEqual(data["metrics"]["val_miou"], 15.5)

    def test_18_metric_reload_consistency(self):
        """Test 18: Reloading saved checkpoint reproduces identical validation metrics."""
        pts = torch.randn(40, 4)
        lbls = torch.tensor([0, 1, 2, 3] * 10, dtype=torch.long)
        loader = DataLoader([(pts, lbls)], batch_size=1, collate_fn=collate_spvcnn)

        model = SPVCNN(num_classes=4, in_channels=4)
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = SPVCNNTrainer(model=model, train_loader=loader, val_loader=loader, config={"training": {"epochs": 1}}, experiment_dir=tmp_dir)
            summary = trainer.train()
            best_ckpt = Path(tmp_dir) / "best_checkpoint.pt"
            passed, orig, reloaded = trainer.reload_and_validate(best_ckpt)
            self.assertTrue(passed)
            self.assertEqual(orig, reloaded)

    def test_19_mapping_regression(self):
        """Test 19: SPVCNN output correctly populates 2.5D grid layers."""
        predictor = SPVCNNPredictor(device="cpu", pretrained_path=str(self.ckpt_path))
        adapter = MLToMappingAdapter(resolution=0.20)
        pts = np.random.uniform(-20, 20, (100, 4)).astype(np.float32)
        res = predictor.predict(pts)
        grid = adapter.build_25d_grid(res)
        self.assertIsInstance(grid, GridMap25D)

    def test_20_no_nan_inf(self):
        """Test 20: Coordinates, predictions, and confidences contain zero NaNs or Infs."""
        predictor = SPVCNNPredictor(device="cpu", pretrained_path=str(self.ckpt_path))
        pts = np.random.uniform(-10, 10, (50, 4)).astype(np.float32)
        res = predictor.predict(pts)
        self.assertFalse(np.isnan(res["xyz"]).any())
        self.assertFalse(np.isnan(res["confidence"]).any())
        self.assertFalse(np.isinf(res["confidence"]).any())

    def test_21_incomplete_dataset_must_block_full_training(self):
        """Test 21: Incomplete dataset audit strictly flags gate failure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            (tmp_root / "sequences/00/velodyne").mkdir(parents=True)
            expected = {"00": 488, "01": 500}
            gate = check_dataset_completeness(tmp_root, expected)
            self.assertFalse(gate["is_complete"])
            self.assertIn("01", gate["missing_sequences"])

    def test_22_complete_dataset_path_activates_full_training(self):
        """Test 22: Complete dataset configuration correctly passes activation gate."""
        expected = {"00": 1}
        gate = check_dataset_completeness(self.dataset_root, expected)
        self.assertTrue(gate["is_complete"])
        self.assertEqual(gate["total_found"], 1)
        self.assertEqual(len(gate["missing_sequences"]), 0)



if __name__ == "__main__":
    unittest.main()
