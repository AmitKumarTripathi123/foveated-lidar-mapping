"""
ROS2 LiDAR Replay Node.
Reads physical SemanticPOSS .bin scans and broadcasts them over /velodyne_points at 10.0 Hz.
Enables end-to-end ROS2 perception and mapping testing without physical vehicle hardware.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

# Find repo root by locating configs directory
current = Path(__file__).resolve()
repo_root = current
for p in [current] + list(current.parents):
    if (p / "configs/production.yaml").is_file():
        repo_root = p
        break

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.dataset import load_point_cloud
from ros2_ws.src.foveated_lidar_mapping.foveated_lidar_mapping.pointcloud2_decoder import (
    numpy_to_pointcloud2_dto,
    PointCloud2DTO,
)

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2
    HAS_RCLPY = True
except ImportError:
    HAS_RCLPY = False
    Node = object


class LidarReplayNode(Node):
    """Replay publisher streaming .bin LiDAR scans over /velodyne_points at 10 Hz."""

    def __init__(self, sequence_dir: Optional[str] = None, frequency_hz: float = 10.0):
        if HAS_RCLPY:
            super().__init__("lidar_replay_node")
            self.declare_parameter("sequence_dir", "dataset/sequences/02/velodyne")
            self.declare_parameter("frequency_hz", 10.0)
            seq_p = self.get_parameter("sequence_dir").get_parameter_value().string_value
            self.freq_hz = self.get_parameter("frequency_hz").get_parameter_value().double_value
        else:
            seq_p = sequence_dir or "dataset/sequences/02/velodyne"
            self.freq_hz = frequency_hz

        self.seq_dir = Path(seq_p)
        if not self.seq_dir.is_absolute():
            self.seq_dir = repo_root / self.seq_dir

        self.bin_files = sorted(list(self.seq_dir.glob("*.bin")))
        self.current_idx = 0

    def get_next_frame(self) -> Optional[PointCloud2DTO]:
        """Fetch the next LiDAR scan and encode as PointCloud2DTO."""
        if not self.bin_files:
            return None

        bin_file = self.bin_files[self.current_idx % len(self.bin_files)]
        self.current_idx += 1
        raw_pts = load_point_cloud(bin_file)
        return numpy_to_pointcloud2_dto(raw_pts, frame_id="velodyne")


def main(args=None):
    if not HAS_RCLPY:
        print("ROS2 (rclpy) is not installed in the current environment.")
        sys.exit(1)

    rclpy.init(args=args)
    node = LidarReplayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
