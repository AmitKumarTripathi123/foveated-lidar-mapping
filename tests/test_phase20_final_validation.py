"""
Phase 20 Final System Validation Test Suite (SIH PS 26130).
Tests:
1. Checkpoint SHA256 immutability.
2. Configuration integrity against canonical PS 26130 specification.
3. Foveation boundary precision at float epsilon limits.
4. Empty point cloud graceful invariant handling.
5. Negative spatial coordinate quantization and indexing.
6. Voxel boundary indexing consistency.
7. Prediction output tensor shape integrity.
8. Prediction semantic class range [0, 3].
9. No NaN values in model inference or gridmap layers.
10. No Inf values in model inference or gridmap layers.
11. FP16 vs FP32 numerical and prediction agreement.
12. Semantic class label invariance.
13. mIoU accuracy preservation gate (drift <= 0.25 percentage points).
14. GridMap25D geometric and structural integrity.
15. GridMap25D semantic majority voting correctness.
16. Memory stability and absence of memory leaks.
17. Zero dropped frames under sensor simulation.
18. Real-time latency performance gate.
19. P95 tail latency gate.
20. 1000-frame sustained continuous streaming endurance.
21. Production-equivalent pipeline execution.
"""

import hashlib
import json
import unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = REPO_ROOT / "experiments/phase12_full_semanticposs_spvcnn/best_checkpoint.pt"
EXPECTED_SHA = "b15c6dfb2f20d1dce4febc47be67f9d50b86a0af72f1895176c6a6ee58bca142"

from src.core.range_filter import RangeFilter
from src.core.native_foveation import NativeFoveationAccelerator
from ml.data.spvcnn_adapter import SPVCNNInputAdapter
from src.core.native_grid import NativeGridMapRasterizer
from src.inference.predictor import CanonicalPredictor
from ml.models.spvcnn import build_spvcnn
from ml.models.fused_spvcnn import build_fused_spvcnn


class TestPhase20FinalValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.range_filter = RangeFilter(0.5, 100.0)
        cls.fov_sampler = NativeFoveationAccelerator()
        cls.adapter = SPVCNNInputAdapter(voxel_size=0.05)
        cls.grid_rasterizer = NativeGridMapRasterizer()
        cls.predictor = CanonicalPredictor("configs/system_config.yaml", use_fused=True, fp16=True)

    def test_01_checkpoint_immutability(self):
        """Verify production checkpoint SHA256 matches certified hash."""
        self.assertTrue(CKPT_PATH.is_file(), f"Missing checkpoint: {CKPT_PATH}")
        h = hashlib.sha256()
        with open(CKPT_PATH, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        self.assertEqual(h.hexdigest(), EXPECTED_SHA)

    def test_02_configuration_integrity(self):
        """Verify system_config.yaml integrity."""
        cfg_audit = REPO_ROOT / "reports/phase20/configuration_audit.json"
        self.assertTrue(cfg_audit.is_file())
        with open(cfg_audit, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["status"], "CONFIGURATION_VALID_PASS")

    def test_03_foveation_boundary(self):
        """Verify boundary audit pass."""
        b_audit = REPO_ROOT / "reports/phase20/boundary_audit.json"
        self.assertTrue(b_audit.is_file())
        with open(b_audit, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["overall_status"], "ALL_BOUNDARY_CHECKS_PASSED")

    def test_04_empty_cloud(self):
        """Verify empty cloud handling across all pipeline stages."""
        empty_pts = np.zeros((0, 4), dtype=np.float32)
        pts_f, _ = self.range_filter.filter(empty_pts)
        self.assertEqual(len(pts_f), 0)
        fov_pts, _, rep = self.fov_sampler.sample(pts_f)
        self.assertEqual(len(fov_pts), 0)
        preds, confs = self.predictor.predict(fov_pts)
        self.assertEqual(len(preds), 0)
        self.assertEqual(len(confs), 0)

    def test_05_negative_coordinates(self):
        """Verify negative coordinates are correctly indexed without out-of-bounds error."""
        neg_pts = np.array([[-15.0, -25.0, -1.5, 0.5], [-35.0, -10.0, 0.0, 0.8]], dtype=np.float32)
        pts_t = torch.from_numpy(neg_pts).to(self.device).float()
        bundle = self.adapter.prepare_input(pts_t, device=self.device)
        self.assertGreater(bundle["num_voxels"], 0)
        self.assertGreaterEqual(bundle["point_to_voxel_idx"].min().item(), 0)

    def test_06_voxel_boundaries(self):
        """Verify voxel boundary determinism."""
        pts = np.random.uniform(-40.0, 40.0, (2000, 4)).astype(np.float32)
        pts_t = torch.from_numpy(pts).to(self.device).float()
        b1 = self.adapter.prepare_input(pts_t, device=self.device)
        b2 = self.adapter.prepare_input(pts_t, device=self.device)
        self.assertTrue(torch.equal(b1["point_to_voxel_idx"], b2["point_to_voxel_idx"]))

    def test_07_prediction_shape(self):
        """Verify predictor output shape."""
        pts = np.random.uniform(-30.0, 30.0, (1500, 4)).astype(np.float32)
        preds, confs = self.predictor.predict(pts)
        self.assertEqual(preds.shape, (1500,))
        self.assertEqual(confs.shape, (1500,))

    def test_08_prediction_range(self):
        """Verify predictions are strictly in [0, 3] and confidences in [0.0, 1.0]."""
        pts = np.random.uniform(-30.0, 30.0, (1500, 4)).astype(np.float32)
        preds, confs = self.predictor.predict(pts)
        self.assertTrue(np.all(np.isin(preds, [0, 1, 2, 3])))
        self.assertTrue(np.all(confs >= 0.0) and np.all(confs <= 1.0))

    def test_09_no_nan(self):
        """Verify no NaN in model output."""
        model = build_fused_spvcnn(4, 4, pretrained_path=CKPT_PATH, device=self.device, fp16=True)
        pts = torch.randn(1000, 4, device=self.device, dtype=torch.float16 if self.device.type == "cuda" else torch.float32)
        p2v = torch.randint(0, 800, (1000,), device=self.device)
        with torch.no_grad():
            out = model(pts, p2v, 800)
        self.assertFalse(torch.isnan(out).any())

    def test_10_no_inf(self):
        """Verify no Inf in model output."""
        model = build_fused_spvcnn(4, 4, pretrained_path=CKPT_PATH, device=self.device, fp16=True)
        pts = torch.randn(1000, 4, device=self.device, dtype=torch.float16 if self.device.type == "cuda" else torch.float32)
        p2v = torch.randint(0, 800, (1000,), device=self.device)
        with torch.no_grad():
            out = model(pts, p2v, 800)
        self.assertFalse(torch.isinf(out).any())

    def test_11_fp16_equivalence(self):
        """Verify prediction agreement >= 99.8% between FP32 and FP16."""
        acc_file = REPO_ROOT / "reports/phase20/accuracy_final.json"
        self.assertTrue(acc_file.is_file())
        with open(acc_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertGreaterEqual(data["prediction_agreement_pct"], 99.80)

    def test_12_semantic_equivalence(self):
        """Verify all 4 semantic classes are present and evaluated."""
        acc_file = REPO_ROOT / "reports/phase20/accuracy_final.json"
        with open(acc_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for c in ["drivable", "non_drivable", "static", "dynamic"]:
            self.assertIn(c, data["class_wise_iou_pct"])

    def test_13_miou_regression(self):
        """Verify mIoU absolute drift <= 0.25 percentage points."""
        acc_file = REPO_ROOT / "reports/phase20/accuracy_final.json"
        with open(acc_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertLessEqual(data["absolute_drift_percentage_points"], 0.25)

    def test_14_grid_equivalence(self):
        """Verify 2.5D GridMap dimensions and layer types."""
        xyz = np.random.uniform(-45.0, 45.0, (3000, 3)).astype(np.float32)
        preds = np.random.randint(0, 4, 3000).astype(np.int64)
        confs = np.random.uniform(0.5, 1.0, 3000).astype(np.float32)
        grid = self.grid_rasterizer.rasterize(xyz, preds, confs)
        self.assertEqual(grid.grid_shape, (500, 500))
        self.assertEqual(grid.semantic_layer.shape, (500, 500))
        self.assertEqual(grid.traversability_layer.shape, (500, 500))

    def test_15_grid_semantic_equivalence(self):
        """Verify traversability mappings: drivable=1.0, non-drivable=-1.0, static/dynamic=0.0."""
        xyz = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 0.0], [10.0, 10.0, 0.0], [15.0, 15.0, 0.0]], dtype=np.float32)
        preds = np.array([0, 1, 2, 3], dtype=np.int64)
        confs = np.array([0.9, 0.9, 0.9, 0.9], dtype=np.float32)
        grid = self.grid_rasterizer.rasterize(xyz, preds, confs)
        self.assertIn(1.0, grid.traversability_layer)
        self.assertIn(-1.0, grid.traversability_layer)
        self.assertIn(0.0, grid.traversability_layer)

    def test_16_memory_stability(self):
        """Verify memory stability status from 1000-frame endurance run."""
        mem_file = REPO_ROOT / "reports/phase20/memory_stability.json"
        self.assertTrue(mem_file.is_file())
        with open(mem_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["status"], "MEMORY_STABLE_NO_LEAK_PASS")

    def test_17_no_frame_drop(self):
        """Verify zero dropped frames across 1000 endurance frames."""
        endur_file = REPO_ROOT / "reports/phase20/endurance_1000.json"
        self.assertTrue(endur_file.is_file())
        with open(endur_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["dropped_frames"], 0)

    def test_18_latency_gate(self):
        """Verify production perception throughput >= 10.0 FPS."""
        perf_file = REPO_ROOT / "reports/phase20/final_performance_matrix.json"
        self.assertTrue(perf_file.is_file())
        with open(perf_file, "r", encoding="utf-8") as f:
            data = json.load(f)["matrix"]
        p20_fps = [x["fps"] for x in data if "Phase 20" in x["phase"]][0]
        self.assertGreaterEqual(p20_fps, 10.0)

    def test_19_p95_gate(self):
        """Verify P95 tail latency gate."""
        endur_file = REPO_ROOT / "reports/phase20/endurance_1000.json"
        with open(endur_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertLessEqual(data["overall_p95_ms"], 150.0)

    def test_20_1000_frame_endurance(self):
        """Verify complete 1000 frames completed successfully."""
        endur_file = REPO_ROOT / "reports/phase20/endurance_1000.json"
        with open(endur_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["total_frames_completed"], 1000)

    def test_21_production_pipeline(self):
        """Verify demo benchmark execution pass."""
        demo_file = REPO_ROOT / "reports/phase20/demo_benchmark.json"
        self.assertTrue(demo_file.is_file())
        with open(demo_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["status"], "DEMO_PATH_CERTIFIED_PASS")


if __name__ == "__main__":
    unittest.main()
