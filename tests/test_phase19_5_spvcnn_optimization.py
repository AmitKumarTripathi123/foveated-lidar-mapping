"""
Phase 19.5 Test Suite: SPVCNN Inference Acceleration & Accuracy Optimization (SIH PS 26130).
Tests:
1. Checkpoint SHA256 immutability.
2. FP32 reference model baseline.
3. FP16 output shape and dtype integrity.
4. AMP output shape and dtype integrity.
5. Prediction validity and confidence bounds [0.0, 1.0].
6. Prediction class range [0, 3] within semantic ontology.
7. Semantic accuracy regression gate (mIoU drift <= 0.25 percentage points).
8. Class-wise IoU stability across drivable, non-drivable, static, dynamic.
9. Sparse coordinate quantization and indexer equivalence.
10. Active voxel count consistency.
11. No NaN values in model activations or logits.
12. No Inf values in model activations or logits.
13. GPU VRAM stability across repeated iterations.
14. Latency gate: Fused FP16 faster than FP32 Base.
15. P95 tail latency gate.
16. Pipeline regression recovery gate.
17. Zero dropped frames under sensor simulation.
"""

import hashlib
import json
import unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from ml.models.spvcnn import SPVCNN, build_spvcnn
from ml.models.fused_spvcnn import FusedSPVCNN, build_fused_spvcnn
from src.inference.predictor import CanonicalPredictor
from ml.data.spvcnn_adapter import SPVCNNInputAdapter


class TestPhase19_5SPVCNNOptimization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.ckpt_path = cls.repo_root / "experiments/phase12_full_semanticposs_spvcnn/best_checkpoint.pt"
        cls.expected_sha = "b15c6dfb2f20d1dce4febc47be67f9d50b86a0af72f1895176c6a6ee58bca142"
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.adapter = SPVCNNInputAdapter(voxel_size=0.05)

    def test_01_checkpoint_immutability(self):
        """Test 1: Verify production checkpoint exists and matches frozen SHA256 hash."""
        self.assertTrue(self.ckpt_path.is_file(), f"Checkpoint missing: {self.ckpt_path}")
        h = hashlib.sha256()
        with open(self.ckpt_path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        self.assertEqual(h.hexdigest(), self.expected_sha, f"Checkpoint SHA256 mismatch! Got: {h.hexdigest()}")

    def test_02_fp32_reference(self):
        """Test 2: Verify FP32 reference model initializes and evaluates correctly."""
        model = build_spvcnn(4, 4, pretrained_path=self.ckpt_path, device=self.device).eval()
        pts = torch.randn(1000, 4, device=self.device, dtype=torch.float32)
        p2v = torch.randint(0, 800, (1000,), device=self.device)
        with torch.no_grad():
            out = model(pts, p2v, 800)
        self.assertEqual(out.shape, (1000, 4))
        self.assertEqual(out.dtype, torch.float32)

    def test_03_fp16_output_shape(self):
        """Test 3: Verify Fused FP16 model produces valid shape and Half dtype."""
        model = build_fused_spvcnn(4, 4, pretrained_path=self.ckpt_path, device=self.device, fp16=True).eval()
        pts = torch.randn(1000, 4, device=self.device, dtype=torch.float16 if self.device.type == "cuda" else torch.float32)
        p2v = torch.randint(0, 800, (1000,), device=self.device)
        with torch.no_grad():
            out = model(pts, p2v, 800)
        self.assertEqual(out.shape, (1000, 4))
        if self.device.type == "cuda":
            self.assertEqual(out.dtype, torch.float16)

    def test_04_amp_output_shape(self):
        """Test 4: Verify AMP autocast produces valid shape."""
        if self.device.type != "cuda":
            self.skipTest("CUDA required for AMP test")
        model = build_fused_spvcnn(4, 4, pretrained_path=self.ckpt_path, device=self.device, fp16=False).eval()
        pts = torch.randn(1000, 4, device=self.device, dtype=torch.float32)
        p2v = torch.randint(0, 800, (1000,), device=self.device)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(pts, p2v, 800)
        self.assertEqual(out.shape, (1000, 4))

    def test_05_prediction_validity(self):
        """Test 5: Verify CanonicalPredictor predictions and confidence scores."""
        predictor = CanonicalPredictor("configs/system_config.yaml", use_fused=True, fp16=True)
        raw_pts = np.random.uniform(-30.0, 30.0, (5000, 4)).astype(np.float32)
        raw_pts[:, 2] = np.random.uniform(-2.0, 2.0, 5000)
        raw_pts[:, 3] = np.random.uniform(0.0, 1.0, 5000)

        preds, confs = predictor.predict(raw_pts)
        self.assertEqual(len(preds), 5000)
        self.assertEqual(len(confs), 5000)
        self.assertTrue(np.all(confs >= 0.0) and np.all(confs <= 1.0))

    def test_06_prediction_class_range(self):
        """Test 6: Verify all predicted classes fall within [0, 3] ontology."""
        predictor = CanonicalPredictor("configs/system_config.yaml", use_fused=True, fp16=True)
        raw_pts = np.random.uniform(-20.0, 20.0, (2000, 4)).astype(np.float32)
        preds, _ = predictor.predict(raw_pts)
        self.assertTrue(np.all(np.isin(preds, [0, 1, 2, 3])))

    def test_07_accuracy_regression(self):
        """Test 7: Verify that optimized model satisfies the mIoU accuracy gate (drift <= 0.25%)."""
        rep_file = self.repo_root / "reports/phase19_5/accuracy_comparison.json"
        self.assertTrue(rep_file.is_file(), "Accuracy comparison report missing")
        with open(rep_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        drift = data["absolute_drift_percentage_points"]
        self.assertLessEqual(drift, 0.25, f"mIoU drift {drift}% exceeds threshold 0.25%!")

    def test_08_class_iou_regression(self):
        """Test 8: Verify individual class IoU regressions remain stable."""
        rep_file = self.repo_root / "reports/phase19_5/accuracy_comparison.json"
        self.assertTrue(rep_file.is_file(), "Accuracy comparison report missing")
        with open(rep_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for cls_name, cls_info in data["class_comparison"].items():
            drift = abs(cls_info["drift_pct"])
            self.assertLessEqual(drift, 1.0, f"Class {cls_name} drift {drift}% exceeds tolerance 1.0%!")

    def test_09_sparse_coordinate_equivalence(self):
        """Test 9: Verify SPVCNNInputAdapter coordinate hashing is deterministic."""
        pts = torch.randn(3000, 4, device=self.device, dtype=torch.float32)
        b1 = self.adapter.prepare_input(pts, device=self.device)
        b2 = self.adapter.prepare_input(pts, device=self.device)
        self.assertEqual(b1["num_voxels"], b2["num_voxels"])
        self.assertTrue(torch.equal(b1["point_to_voxel_idx"], b2["point_to_voxel_idx"]))

    def test_10_active_voxel_count(self):
        """Test 10: Verify active voxel count is valid and bounded."""
        pts = torch.randn(5000, 4, device=self.device, dtype=torch.float32)
        bundle = self.adapter.prepare_input(pts, device=self.device)
        self.assertTrue(0 < bundle["num_voxels"] <= 5000)

    def test_11_no_nan(self):
        """Test 11: Verify no NaN values occur in forward pass."""
        model = build_fused_spvcnn(4, 4, pretrained_path=self.ckpt_path, device=self.device, fp16=True).eval()
        pts = torch.randn(2000, 4, device=self.device, dtype=torch.float16 if self.device.type == "cuda" else torch.float32)
        p2v = torch.randint(0, 1500, (2000,), device=self.device)
        with torch.no_grad():
            out = model(pts, p2v, 1500)
        self.assertFalse(torch.isnan(out).any(), "NaN detected in Fused SPVCNN output!")

    def test_12_no_inf(self):
        """Test 12: Verify no Inf values occur in forward pass."""
        model = build_fused_spvcnn(4, 4, pretrained_path=self.ckpt_path, device=self.device, fp16=True).eval()
        pts = torch.randn(2000, 4, device=self.device, dtype=torch.float16 if self.device.type == "cuda" else torch.float32)
        p2v = torch.randint(0, 1500, (2000,), device=self.device)
        with torch.no_grad():
            out = model(pts, p2v, 1500)
        self.assertFalse(torch.isinf(out).any(), "Inf detected in Fused SPVCNN output!")

    def test_13_gpu_memory_stability(self):
        """Test 13: Verify VRAM remains stable over multiple inference passes."""
        if self.device.type != "cuda":
            self.skipTest("CUDA required for VRAM test")
        torch.cuda.empty_cache()
        model = build_fused_spvcnn(4, 4, pretrained_path=self.ckpt_path, device=self.device, fp16=True).eval()
        pts = torch.randn(40000, 4, device=self.device, dtype=torch.float16)
        p2v = torch.randint(0, 30000, (40000,), device=self.device)

        # Initial pass
        with torch.no_grad():
            _ = model(pts, p2v, 30000)
        torch.cuda.synchronize()
        mem_init = torch.cuda.memory_allocated()

        # Repeated passes
        for _ in range(20):
            with torch.no_grad():
                _ = model(pts, p2v, 30000)
        torch.cuda.synchronize()
        mem_final = torch.cuda.memory_allocated()

        self.assertLessEqual(mem_final, mem_init * 1.05, f"VRAM leakage: init={mem_init}, final={mem_final}")

    def test_14_latency_gate(self):
        """Test 14: Verify FP16 Fused model achieves latency reduction over FP32 Base."""
        rep_file = self.repo_root / "reports/phase19_5/precision_benchmark.json"
        self.assertTrue(rep_file.is_file(), "Precision benchmark not found")
        with open(rep_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertGreaterEqual(data["speedup_fp16_vs_base_fp32"], 1.20, "Speedup is below 1.20x threshold!")

    def test_15_p95_gate(self):
        """Test 15: Verify P95 tail latency remains controlled."""
        rep_file = self.repo_root / "reports/phase19_5/precision_benchmark.json"
        self.assertTrue(rep_file.is_file(), "Precision benchmark not found")
        with open(rep_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        p95 = data["fused_fp16_native"]["p95_ms"]
        self.assertLessEqual(p95, 50.0, f"P95 latency {p95} ms exceeds gate of 50.0 ms!")

    def test_16_pipeline_regression(self):
        """Test 16: Verify end-to-end perception pipeline achieves valid real-time throughput."""
        rep_file = self.repo_root / "reports/phase19_5/pipeline_benchmark.json"
        self.assertTrue(rep_file.is_file(), "Pipeline benchmark not found")
        with open(rep_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertLessEqual(data["phase19_5_optimized"]["mean_ms"], 60.0)

    def test_17_zero_dropped_frames(self):
        """Test 17: Verify zero dropped frames during sensor simulation evaluation."""
        rep_file = self.repo_root / "reports/phase19_5/phase19_5_summary.json"
        self.assertTrue(rep_file.is_file(), "Summary report not found")
        with open(rep_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertTrue(data["performance_gates"]["zero_dropped_frames"])


if __name__ == "__main__":
    unittest.main()
