"""
Phase 6 Performance and Copy-Overhead Benchmark:
Measures detailed latency breakdown:
  A. Python Preprocessing (Range filtering / Foveation)
  B. pybind11 Input Conversion / Buffer View
  C. Pure C++ Grid Execution
  D. Output NumPy Conversion
  E. Total Python -> C++ -> Python Latency
  Binding Overhead = Total Python-C++-Python - Pure C++ Execution Time
"""

import time
import unittest
import numpy as np
import foveated_grid_cpp
from src.foveated_grid import FoveatedGrid25D


class TestPhase6Performance(unittest.TestCase):
    def test_01_latency_breakdown_and_overhead(self):
        """Measure latency breakdown and binding overhead across 66,402 points (SemanticPOSS scale)."""
        engine = foveated_grid_cpp.FoveatedGridEngine()
        N = 66402
        np.random.seed(42)
        raw_pts = np.random.uniform(-70, 70, (N, 4)).astype(np.float32)
        lbls = np.random.choice([0, 1, 2, 3], N).astype(np.int64)
        confs = np.random.uniform(0.7, 1.0, N).astype(np.float32)

        # Stage A: Preprocessing (range filter)
        t0 = time.perf_counter()
        r = np.sqrt(raw_pts[:, 0]**2 + raw_pts[:, 1]**2)
        mask = (r >= 0.5) & (r < 100.0)
        filtered_pts = raw_pts[mask]
        filtered_lbls = lbls[mask]
        filtered_confs = confs[mask]
        stage_a_ms = (time.perf_counter() - t0) * 1000.0

        # Stage B: pybind11 Input conversion into C++ ClassifiedPoint vector
        pt_objs = [
            foveated_grid_cpp.ClassifiedPoint(
                float(filtered_pts[i, 0]), float(filtered_pts[i, 1]), float(filtered_pts[i, 2]),
                float(filtered_pts[i, 3]), int(filtered_lbls[i]), float(filtered_confs[i])
            )
            for i in range(len(filtered_pts))
        ]

        # Stage C: Pure C++ Grid Execution (calling engine.build_grid with pre-built C++ vector)
        iters = 10
        t0 = time.perf_counter()
        for _ in range(iters):
            cells = engine.build_grid(pt_objs)
        pure_cpp_ms = (time.perf_counter() - t0) / iters * 1000.0

        # Stage D + Total: build_grid_numpy (NumPy direct passing + pure C++ + NumPy output conversion)
        t0 = time.perf_counter()
        for _ in range(iters):
            res_dict = engine.build_grid_numpy(filtered_pts, filtered_lbls, filtered_confs)
        total_py_cpp_py_ms = (time.perf_counter() - t0) / iters * 1000.0

        binding_overhead_ms = max(0.0, total_py_cpp_py_ms - pure_cpp_ms)
        throughput_fps = 1000.0 / total_py_cpp_py_ms

        print("\n" + "=" * 80)
        print("  PHASE 6 PERFORMANCE & BINDING OVERHEAD AUDIT")
        print("=" * 80)
        print(f"Points Processed:               {len(filtered_pts):,d}")
        print(f"A. Python Preprocessing:         {stage_a_ms:.3f} ms")
        print(f"B+D. pybind11 Binding Overhead: {binding_overhead_ms:.3f} ms")
        print(f"C. Pure C++ Grid Execution:      {pure_cpp_ms:.3f} ms")
        print(f"E. Total Python->C++->Python:    {total_py_cpp_py_ms:.3f} ms")
        print(f"Throughput:                     {throughput_fps:.2f} FPS")
        print("=" * 80)

        self.assertLess(total_py_cpp_py_ms, 50.0, f"Grid generation latency too high: {total_py_cpp_py_ms:.2f} ms")


if __name__ == "__main__":
    unittest.main()
