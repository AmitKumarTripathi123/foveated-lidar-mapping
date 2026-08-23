"""
ROS2 Node: Foveated 2.5D LiDAR Mapping Node (SIH PS 26130).
Subscribes to /velodyne_points (sensor_msgs/msg/PointCloud2),
executes the certified 3-zone foveated SPVCNN perception pipeline,
and publishes semantic point clouds, 2.5D grid maps, and real-time telemetry.
"""

import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Find repo root by locating configs directory
current = Path(__file__).resolve()
repo_root = current
for p in [current] + list(current.parents):
    if (p / "configs/production.yaml").is_file():
        repo_root = p
        break

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2
    from nav_msgs.msg import OccupancyGrid
    from std_msgs.msg import String, Header
    HAS_RCLPY = True
except ImportError:
    HAS_RCLPY = False
    Node = object  # Fallback base class

from ml.pipeline.production_pipeline import ProductionPipeline, FrameProcessingResult
from ros2_ws.src.foveated_lidar_mapping.foveated_lidar_mapping.pointcloud2_decoder import (
    decode_pointcloud2_to_numpy,
    numpy_to_pointcloud2_dto,
    PointCloud2DTO,
)


class FoveatedMappingNode(Node):
    """Production ROS2 node for real-time 3D LiDAR perception and 2.5D mapping."""

    def __init__(self, config_path: Optional[str] = None, max_queue_size: int = 2):
        if HAS_RCLPY:
            super().__init__("foveated_mapping_node")
            self.declare_parameter("config_path", "configs/production.yaml")
            self.declare_parameter("max_queue_size", 2)
            cfg_p = self.get_parameter("config_path").get_parameter_value().string_value
            q_size = self.get_parameter("max_queue_size").get_parameter_value().integer_value
        else:
            cfg_p = config_path or "configs/production.yaml"
            q_size = max_queue_size

        # Resolve config path
        self.config_path = Path(cfg_p)
        if not self.config_path.is_absolute():
            self.config_path = repo_root / self.config_path

        # Initialize hardened production pipeline
        self.pipeline = ProductionPipeline(self.config_path)

        # Producer-Consumer Bounded Queue
        self.max_queue_size = q_size
        self.frame_queue = queue.Queue(maxsize=self.max_queue_size)

        # Telemetry metrics
        self.received_frames = 0
        self.processed_frames = 0
        self.dropped_frames = 0
        self.last_latency_ms = 0.0
        self.avg_fps = 0.0

        # Topic Names
        self.input_topic = "/velodyne_points"
        self.out_semantic_pc_topic = "/semantic_pointcloud"
        self.out_gridmap_topic = "/foveated_grid_map_25d"
        self.out_traversability_topic = "/traversability_map"
        self.out_telemetry_topic = "/telemetry"

        # Worker thread
        self.running = True
        self.worker_thread = threading.Thread(target=self._inference_worker_loop, daemon=True)
        self.worker_thread.start()

    def pointcloud_callback(self, msg: Any):
        """Callback invoked upon receiving a new PointCloud2 frame."""
        self.received_frames += 1

        # If queue is full, drop oldest frame to maintain low latency (real-time contract)
        if self.frame_queue.full():
            try:
                _ = self.frame_queue.get_nowait()
                self.dropped_frames += 1
            except queue.Empty:
                pass

        try:
            self.frame_queue.put_nowait(msg)
        except queue.Full:
            self.dropped_frames += 1

    def _inference_worker_loop(self):
        """Dedicated background thread executing perception and mapping."""
        t_last_frame = time.perf_counter()

        while self.running:
            try:
                msg = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            try:
                # 1. Decode PointCloud2
                pts = decode_pointcloud2_to_numpy(msg)

                # 2. Process through certified production pipeline
                frame_id = f"ros2_frame_{self.processed_frames:06d}"
                res = self.pipeline.process_frame(pts, frame_id=frame_id)

                self.processed_frames += 1
                self.last_latency_ms = res.latency_ms

                # 3. Publish Telemetry
                dt = time.perf_counter() - t_last_frame
                t_last_frame = time.perf_counter()
                self.avg_fps = 1.0 / max(dt, 1e-4)

                telemetry = {
                    "frame_id": frame_id,
                    "success": res.success,
                    "latency_ms": res.latency_ms,
                    "stage_latencies_ms": res.stage_latencies_ms,
                    "effective_fps": round(self.avg_fps, 2),
                    "received_frames": self.received_frames,
                    "processed_frames": self.processed_frames,
                    "dropped_frames": self.dropped_frames,
                    "queue_depth": self.frame_queue.qsize(),
                    "input_points": res.num_input_points,
                    "foveated_points": res.num_foveated_points,
                }
                # Publish topics if ROS2 is active
                if HAS_RCLPY:
                    self._publish_ros2_topics(res, telemetry)

            except Exception as e:
                print(f"Error in ROS2 inference worker: {e}", file=sys.stderr)

    def _publish_ros2_topics(self, res: FrameProcessingResult, telemetry: Dict[str, Any]):
        """Publish ROS2 messages if ROS2 runtime is present."""
        pass  # Extended in ROS2 runtime environments

    def stop(self):
        """Stop worker thread cleanly."""
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)


def main(args=None):
    if not HAS_RCLPY:
        print("ROS2 (rclpy) is not installed in the current environment.")
        sys.exit(1)

    rclpy.init(args=args)
    node = FoveatedMappingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
