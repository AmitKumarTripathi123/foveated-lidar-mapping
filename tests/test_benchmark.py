"""Unit tests for PerformanceBenchmark."""
import unittest
import numpy as np

from src.types import PointCloudFrame
from src.benchmark import PerformanceBenchmark


class TestPerformanceBenchmark(unittest.TestCase):
    def test_benchmark_run(self):
        bench = PerformanceBenchmark(max_range=100.0)
        pts = np.random.uniform(0, 0.2, size=(500, 4)).astype(np.float32)
        lbls = np.zeros(500, dtype=np.uint32)
        frame = PointCloudFrame(points=pts, labels=lbls)

        cfg_dict = {
            "type": "foveated",
            "description": "Test Foveation",
            "bands": [
                {"name": "near", "min_range": 0.0, "max_range": 10.0, "voxel_size": 0.05},
                {"name": "mid", "min_range": 10.0, "max_range": 40.0, "voxel_size": 0.15},
                {"name": "far", "min_range": 40.0, "max_range": 100.0, "voxel_size": 0.50}
            ]
        }

        res = bench.benchmark_pipeline_on_frames([frame], config_name="test_cfg", config_dict=cfg_dict, repeats_per_frame=2)
        self.assertGreater(res.fps_mean, 0.0)
        self.assertGreater(res.point_reduction_percent, 0.0)
        self.assertIn("total_pipeline", res.stage_timings)


if __name__ == "__main__":
    unittest.main()
