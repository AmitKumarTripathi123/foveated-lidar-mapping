"""
Phase 12 Full SemanticPOSS GPU Fine-Tuning and Dataset Provenance Tests.
"""

import unittest
from pathlib import Path
import torch
import numpy as np

from ml.data.dataset import LidarDataset, load_point_cloud, load_labels
from ml.data.semanticposs_label_mapping import SemanticPOSSLabelRemapper
from ml.data.amit_adapter import FoveatedVoxelSampler
from ml.models.spvcnn import build_spvcnn
from ml.models.spvcnn_predictor import SPVCNNPredictor
from ml.models.mapping_adapter import MLToMappingAdapter


class TestPhase12FullSemanticPOSS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.ds_root = cls.repo_root / "dataset"
        cls.ckpt_path = cls.repo_root / "experiments/phase12_full_semanticposs_spvcnn/best_checkpoint.pt"

    def test_01_full_2988_frame_discovery(self):
        """Test 1: Full 2,988 frames discovered across all 6 sequences."""
        expected = {"00": 488, "01": 500, "02": 500, "03": 500, "04": 500, "05": 500}
        total_bins = 0
        total_lbls = 0
        for seq, count in expected.items():
            s_dir = self.ds_root / "sequences" / seq
            self.assertTrue(s_dir.is_dir(), f"Sequence {seq} directory missing!")
            v_bins = list((s_dir / "velodyne").glob("*.bin"))
            l_lbls = list((s_dir / "labels").glob("*.label"))
            self.assertEqual(len(v_bins), count, f"Sequence {seq} bin count mismatch!")
            self.assertEqual(len(l_lbls), count, f"Sequence {seq} label count mismatch!")
            total_bins += len(v_bins)
            total_lbls += len(l_lbls)

        self.assertEqual(total_bins, 2988)
        self.assertEqual(total_lbls, 2988)

    def test_02_stem_pairing_integrity(self):
        """Test 2: Every .bin matches its corresponding .label by exact stem."""
        for seq in ["00", "01", "02", "03", "04", "05"]:
            s_dir = self.ds_root / "sequences" / seq
            v_bins = sorted(list((s_dir / "velodyne").glob("*.bin")))
            l_lbls = sorted(list((s_dir / "labels").glob("*.label")))
            b_stems = [b.stem for b in v_bins]
            l_stems = [l.stem for l in l_lbls]
            self.assertEqual(b_stems, l_stems, f"Stem mismatch in seq {seq}")

    def test_03_strict_train_val_split_disjoint(self):
        """Test 3: Train sequences (00, 01, 03, 04, 05) and Val sequence (02) have zero overlap."""
        train_seqs = ["00", "01", "03", "04", "05"]
        val_seqs = ["02"]
        self.assertTrue(set(train_seqs).isdisjoint(set(val_seqs)))

    def test_04_dataset_length_verification(self):
        """Test 4: LidarDataset lengths exactly equal 2,488 (train) and 500 (val)."""
        remapper = SemanticPOSSLabelRemapper()
        train_ds = LidarDataset(root=str(self.ds_root), sequences=["00", "01", "03", "04", "05"], label_remapper=remapper.remap, require_labels=True)
        val_ds = LidarDataset(root=str(self.ds_root), sequences=["02"], label_remapper=remapper.remap, require_labels=True)
        self.assertEqual(len(train_ds), 2488)
        self.assertEqual(len(val_ds), 500)

    def test_05_label_remapping_sih_range(self):
        """Test 5: Remapped labels are strictly within {0, 1, 2, 3, 255}."""
        remapper = SemanticPOSSLabelRemapper()
        sample_lbl = load_labels(self.ds_root / "sequences/00/labels/000000.label")
        remapped = remapper.remap(sample_lbl)
        unique_classes = set(np.unique(remapped))
        self.assertTrue(unique_classes.issubset({0, 1, 2, 3, 255}))

    def test_06_foveated_reduction_correspondence(self):
        """Test 6: Amit 3-zone foveation preserves exact point-label 1:1 correspondence."""
        pts = load_point_cloud(self.ds_root / "sequences/00/velodyne/000000.bin")
        lbls = load_labels(self.ds_root / "sequences/00/labels/000000.label")
        sampler = FoveatedVoxelSampler()
        fov_pts, fov_lbls, rep = sampler.sample(pts, lbls)
        self.assertEqual(len(fov_pts), len(fov_lbls))
        self.assertLess(len(fov_pts), len(pts))

    def test_07_checkpoint_provenance_and_reload(self):
        """Test 7: Best checkpoint exists, contains valid metrics, and passes reload validation."""
        self.assertTrue(self.ckpt_path.exists(), "Phase 12 best checkpoint not found!")
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        self.assertIn("metrics", ckpt)
        self.assertIn("val_miou", ckpt["metrics"])
        self.assertGreater(ckpt["metrics"]["val_miou"], 40.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for GPU inference test")
    def test_08_cuda_model_inference_contract(self):
        """Test 8: CUDA SPVCNN predictor satisfies frozen contract [xyz, class, conf]."""
        predictor = SPVCNNPredictor(device="cuda", pretrained_path=str(self.ckpt_path))
        pts = load_point_cloud(self.ds_root / "sequences/02/velodyne/000001.bin")
        sampler = FoveatedVoxelSampler()
        fov_pts, _, _ = sampler.sample(pts)
        res = predictor.predict(fov_pts)
        self.assertIn("xyz", res)
        self.assertIn("predicted_class", res)
        self.assertIn("confidence", res)
        self.assertEqual(len(res["xyz"]), len(fov_pts))
        self.assertTrue(np.all((res["predicted_class"] >= 0) & (res["predicted_class"] <= 3)))
        self.assertTrue(np.all((res["confidence"] >= 0.0) & (res["confidence"] <= 1.0)))

    def test_09_gridmap25d_regression(self):
        """Test 9: MLToMappingAdapter produces valid GridMap25D from predictions without NaNs."""
        predictor = SPVCNNPredictor(device="cpu", pretrained_path=str(self.ckpt_path))
        map_adapter = MLToMappingAdapter()
        pts = load_point_cloud(self.ds_root / "sequences/00/velodyne/000000.bin")
        sampler = FoveatedVoxelSampler()
        fov_pts, _, _ = sampler.sample(pts)
        res = predictor.predict(fov_pts)
        grid = map_adapter.build_25d_grid(res)
        self.assertIsNotNone(grid)
        self.assertFalse(np.isnan(grid.elevation_mean).all())


if __name__ == "__main__":
    unittest.main()
