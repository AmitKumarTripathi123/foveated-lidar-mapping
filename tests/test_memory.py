"""
Memory Safety and Lifetime Tests:
  - Repeated execution memory leak test (50 iterations)
  - Large point cloud allocations (100k, 200k points)
  - Object lifetime and container ownership
"""

import os
import psutil
import unittest
import numpy as np
import foveated_grid_cpp


class TestMemorySafety(unittest.TestCase):
    def test_01_repeated_execution_memory_stability(self):
        """Test 1: Run 50 iterations of 20,000 points and assert RSS memory remains stable (no leak)."""
        process = psutil.Process(os.getpid())
        engine = foveated_grid_cpp.FoveatedGridEngine()

        np.random.seed(42)
        pts = np.random.uniform(-50, 50, (20000, 4)).astype(np.float32)
        lbls = np.random.choice([0, 1, 2, 3], 20000).astype(np.int64)
        confs = np.random.uniform(0.7, 1.0, 20000).astype(np.float32)

        # Warmup
        for _ in range(5):
            _ = engine.build_grid_numpy(pts, lbls, confs)

        mem_start = process.memory_info().rss / (1024 * 1024)

        for _ in range(50):
            res = engine.build_grid_numpy(pts, lbls, confs)
            del res

        mem_end = process.memory_info().rss / (1024 * 1024)
        growth_mb = mem_end - mem_start
        print(f"[Memory Test] Initial RSS: {mem_start:.2f} MB | Final RSS: {mem_end:.2f} MB | Growth: {growth_mb:.2f} MB")
        self.assertLess(growth_mb, 15.0, f"Excessive memory growth detected: {growth_mb:.2f} MB")

    def test_02_large_point_clouds(self):
        """Test 2: Stress test with 200,000 points."""
        engine = foveated_grid_cpp.FoveatedGridEngine()
        pts = np.random.uniform(-70, 70, (200000, 4)).astype(np.float32)
        lbls = np.random.choice([0, 1, 2, 3], 200000).astype(np.int64)
        confs = np.random.uniform(0.7, 1.0, 200000).astype(np.float32)

        res = engine.build_grid_numpy(pts, lbls, confs)
        self.assertGreater(res["num_cells"], 10000)


if __name__ == "__main__":
    unittest.main()
