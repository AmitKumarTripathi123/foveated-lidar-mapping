"""
Phase 18 Canonical Foveated Architecture Freeze Unit & System Tests:
- Asserts system_config.yaml single source of truth integrity
- Asserts checkpoint SHA256 cryptographic immutability
- Tests canonical 3-zone distance geometry (5cm / 15cm / 50cm)
- Tests hierarchical CellKey spatial indexing and parent-child relationships
- Tests canonical FoveatedPipeline end-to-end execution
- Tests standardized benchmark JSON schema
- Tests canonical dashboard visualization artifacts
"""

import unittest
from pathlib import Path
import json
import hashlib
import numpy as np
import yaml

from src.core.hierarchy import FoveatedHierarchyEngine
from src.core.types import CellKey, SuperClass
from src.core.lidar_loader import load_lidar_points
from src.inference.pipeline import FoveatedPipeline, PipelineResult
from ml.pipeline.production_pipeline import verify_file_sha256


class TestPhase18CanonicalFreeze(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.config_path = cls.repo_root / "configs/system_config.yaml"
        cls.ckpt_path = cls.repo_root / "experiments/phase12_full_semanticposs_spvcnn/best_checkpoint.pt"
        cls.benchmark_json = cls.repo_root / "reports/phase18/canonical_benchmark.json"
        cls.dash_png = cls.repo_root / "reports/phase18/figures/canonical_dashboard.png"
        cls.dash_html = cls.repo_root / "reports/phase18/canonical_dashboard.html"

    def test_01_single_source_config_integrity(self):
        """Test 1: system_config.yaml exists and defines all required project sections."""
        self.assertTrue(self.config_path.is_file(), "configs/system_config.yaml missing!")
        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        required_sections = ["project", "lidar", "foveation", "semantic_classes", "model", "grid", "benchmark"]
        for sec in required_sections:
            self.assertIn(sec, cfg, f"Missing section '{sec}' in system_config.yaml!")

        # Verify canonical foveation values
        fov = cfg["foveation"]
        self.assertEqual(fov["near"]["radius"], 10.0)
        self.assertEqual(fov["near"]["resolution"], 0.05)
        self.assertEqual(fov["mid"]["radius"], 40.0)
        self.assertEqual(fov["mid"]["resolution"], 0.15)
        self.assertEqual(fov["far"]["radius"], 100.0)
        self.assertEqual(fov["far"]["resolution"], 0.50)

    def test_02_checkpoint_immutability(self):
        """Test 2: Certified checkpoint SHA256 strictly matches production baseline."""
        expected_sha = "b15c6dfb2f20d1dce4febc47be67f9d50b86a0af72f1895176c6a6ee58bca142"
        self.assertTrue(self.ckpt_path.is_file())
        self.assertTrue(verify_file_sha256(self.ckpt_path, expected_sha))

    def test_03_canonical_foveation_geometry_invariants(self):
        """Test 3: Hierarchy engine maps distance intervals strictly to 5cm, 15cm, and 50cm."""
        engine = FoveatedHierarchyEngine(self.config_path)

        # Near zone (<10m) -> 0.05m (level 0)
        z_near = engine.resolve_zone(5.0)
        self.assertIsNotNone(z_near)
        self.assertEqual(z_near.level, 0)
        self.assertEqual(z_near.resolution, 0.05)

        # Mid zone (10-40m) -> 0.15m (level 1)
        z_mid = engine.resolve_zone(25.0)
        self.assertIsNotNone(z_mid)
        self.assertEqual(z_mid.level, 1)
        self.assertEqual(z_mid.resolution, 0.15)

        # Far zone (40-100m) -> 0.50m (level 2)
        z_far = engine.resolve_zone(70.0)
        self.assertIsNotNone(z_far)
        self.assertEqual(z_far.level, 2)
        self.assertEqual(z_far.resolution, 0.50)

        # Outer range (>100m) -> Filtered
        self.assertIsNone(engine.resolve_zone(105.0))

    def test_04_hierarchical_cell_indexing_and_parent_relations(self):
        """Test 4: CellKey generation and parent key navigation are mathematically consistent."""
        engine = FoveatedHierarchyEngine(self.config_path)

        # Point in near zone (x=2.0, y=2.0)
        res = engine.point_to_cell_key(2.0, 2.0)
        self.assertIsNotNone(res)
        key, zone = res
        self.assertEqual(key.level, 0)
        self.assertEqual(key.ix, int(2.0 / 0.05))
        self.assertEqual(key.iy, int(2.0 / 0.05))

        # Parent key at level 1
        parent = engine.get_parent_key(key)
        self.assertIsNotNone(parent)
        self.assertEqual(parent.level, 1)

    def test_05_canonical_pipeline_end_to_end(self):
        """Test 5: Canonical FoveatedPipeline processes raw LiDAR frame and produces GridMap25D."""
        pipeline = FoveatedPipeline(self.config_path)
        sample_bin = self.repo_root / "dataset/sequences/02/velodyne/000001.bin"
        self.assertTrue(sample_bin.is_file())

        pts = load_lidar_points(sample_bin)
        res = pipeline.run(pts)

        self.assertIsInstance(res, PipelineResult)
        self.assertEqual(res.grid_map.grid_shape, (500, 500))
        self.assertGreater(res.foveated_points_count, 0)
        self.assertLessEqual(res.foveated_points_count, res.raw_points_count)
        self.assertGreater(res.total_latency_ms, 0.0)

    def test_06_benchmark_json_standard_schema(self):
        """Test 6: canonical_benchmark.json matches required standard schema and metrics."""
        self.assertTrue(self.benchmark_json.is_file(), "canonical_benchmark.json missing!")
        with open(self.benchmark_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        required_keys = ["config", "config_hash", "checkpoint_sha256", "frames", "fps", "latency_mean_ms", "latency_p95_ms", "memory_mb", "cell_count"]
        for k in required_keys:
            self.assertIn(k, data, f"Missing key '{k}' in canonical benchmark JSON!")

        self.assertTrue(data["checkpoint_verified"])
        self.assertGreater(data["fps"], 0.0)
        self.assertEqual(data["memory_mb"], 4.77)

    def test_07_canonical_visualization_artifacts(self):
        """Test 7: Canonical dashboard figure and HTML HUD exist and are non-empty."""
        self.assertTrue(self.dash_png.is_file(), "canonical_dashboard.png missing!")
        self.assertTrue(self.dash_html.is_file(), "canonical_dashboard.html missing!")
        self.assertGreater(self.dash_png.stat().st_size, 1000)
        self.assertGreater(self.dash_html.stat().st_size, 500)


if __name__ == "__main__":
    unittest.main()
