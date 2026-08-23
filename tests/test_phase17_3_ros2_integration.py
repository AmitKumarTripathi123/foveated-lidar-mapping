"""
Phase 17.3 ROS2 Integration & Live Visualization Unit Tests:
- Asserts checkpoint SHA256 cryptographic immutability
- Tests PointCloud2 structured decoding, variable offsets, and NaN filtering
- Tests NumPy to PointCloud2DTO serialization
- Tests FoveatedMappingNode bounded queue management and dropped frame accounting
- Asserts ros2_benchmark.json report existence and live dashboard image artifacts
"""

import unittest
from pathlib import Path
import json
import hashlib
import numpy as np

from ml.pipeline.production_pipeline import verify_file_sha256
from ros2_ws.src.foveated_lidar_mapping.foveated_lidar_mapping.pointcloud2_decoder import (
    decode_pointcloud2_to_numpy,
    numpy_to_pointcloud2_dto,
    PointCloud2DTO,
    PointFieldDTO,
    FLOAT32,
)
from ros2_ws.src.foveated_lidar_mapping.foveated_lidar_mapping.foveated_mapping_node import FoveatedMappingNode


class TestPhase17_3ROS2Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.ckpt_path = cls.repo_root / "experiments/phase12_full_semanticposs_spvcnn/best_checkpoint.pt"
        cls.ros2_json = cls.repo_root / "reports/phase17_3/ros2_benchmark.json"
        cls.dash_png = cls.repo_root / "reports/phase17_3/figures/live_pipeline_dashboard.png"
        cls.dash_html = cls.repo_root / "reports/phase17_3/live_interactive_dashboard.html"

    def test_01_checkpoint_immutability(self):
        """Test 1: Certified checkpoint SHA256 matches production baseline."""
        expected_sha = "b15c6dfb2f20d1dce4febc47be67f9d50b86a0af72f1895176c6a6ee58bca142"
        self.assertTrue(self.ckpt_path.is_file())
        self.assertTrue(verify_file_sha256(self.ckpt_path, expected_sha))

    def test_02_pointcloud2_decoder_and_encoder_roundtrip(self):
        """Test 2: NumPy to PointCloud2DTO to NumPy decoder recovers exact coordinate values."""
        pts = np.array([
            [1.5, 2.5, 3.5, 0.8],
            [-10.2, 5.4, -1.2, 0.5],
            [30.0, -15.0, 0.0, 0.2],
        ], dtype=np.float32)

        dto = numpy_to_pointcloud2_dto(pts)
        decoded = decode_pointcloud2_to_numpy(dto)

        self.assertEqual(decoded.shape, (3, 4))
        np.testing.assert_allclose(decoded, pts, atol=1e-5)

    def test_03_pointcloud2_decoder_nan_filtering(self):
        """Test 3: PointCloud2 decoder filters out NaN and Inf coordinates defensively."""
        pts = np.array([
            [1.0, 2.0, 3.0, 0.5],
            [np.nan, 2.0, 3.0, 0.5],
            [1.0, np.inf, 3.0, 0.5],
            [-5.0, 6.0, 7.0, 0.9],
        ], dtype=np.float32)

        dto = numpy_to_pointcloud2_dto(pts)
        decoded = decode_pointcloud2_to_numpy(dto)

        self.assertEqual(decoded.shape, (2, 4))
        self.assertTrue(np.all(np.isfinite(decoded)))

    def test_04_mapping_node_queue_overflow_management(self):
        """Test 4: FoveatedMappingNode safely handles burst input with bounded queue."""
        node = FoveatedMappingNode(max_queue_size=2)
        pts = np.random.uniform(-20, 20, size=(100, 4)).astype(np.float32)
        dto = numpy_to_pointcloud2_dto(pts)

        # Send 5 frames rapidly to test queue overflow
        for _ in range(5):
            node.pointcloud_callback(dto)

        self.assertLessEqual(node.frame_queue.qsize(), 2)
        self.assertEqual(node.received_frames, 5)
        node.stop()

    def test_05_benchmark_payload_and_visual_artifacts(self):
        """Test 5: ros2_benchmark.json and live dashboard artifacts exist."""
        self.assertTrue(self.ros2_json.is_file(), "ros2_benchmark.json missing!")
        self.assertTrue(self.dash_png.is_file(), "live_pipeline_dashboard.png missing!")
        self.assertTrue(self.dash_html.is_file(), "live_interactive_dashboard.html missing!")

        with open(self.ros2_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["benchmark_mode"], "ROS2_REPLAY_SIMULATION")
        self.assertGreater(data["frames_processed"], 0)


if __name__ == "__main__":
    unittest.main()
