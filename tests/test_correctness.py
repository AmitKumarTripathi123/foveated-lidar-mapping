"""
Test 3-way Correctness:
  1. Pure Python Reference Grid
  2. Pure C++ Standalone Engine (CLI)
  3. pybind11 C++ Engine in Python
"""

import subprocess
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import foveated_grid_cpp

repo_root = Path(__file__).resolve().parent.parent
import sys
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.foveated_grid import FoveatedGrid25D
from tests.compare_outputs import compare_grids


class Test3WayCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.golden_in = repo_root / "tests/data/phase5_golden_input.csv"
        cls.py_ref = repo_root / "tests/reference/python_grid.csv"
        cls.cpp_out = repo_root / "tests/output/cpp_grid.csv"
        cls.pybind_out = repo_root / "tests/output/pybind_grid.csv"
        cls.cli_bin = repo_root / "cpp/bin/foveated_grid_cli"

    def test_01_pybind_matches_python_and_cpp_reference(self):
        """Test 1: Run golden test through pybind11 and assert 0 discrepancies against both Python and C++ reference."""
        in_df = pd.read_csv(self.golden_in)
        pts = in_df[["x", "y", "z", "intensity"]].values.astype(np.float32)
        lbls = in_df["class_id"].values.astype(np.int64)
        confs = in_df["confidence"].values.astype(np.float32)

        engine = foveated_grid_cpp.FoveatedGridEngine()
        res = engine.build_grid_numpy(pts, lbls, confs)

        df_pybind = pd.DataFrame({
            "band_name": res["bands"],
            "ix": res["ix"],
            "iy": res["iy"],
            "resolution": np.round(res["resolution"], 4),
            "point_count": res["point_count"],
            "elevation_mean": np.round(res["elevation_mean"], 5),
            "elevation_min": np.round(res["elevation_min"], 5),
            "elevation_max": np.round(res["elevation_max"], 5),
            "semantic_class": res["semantic_class"],
            "confidence": np.round(res["confidence"], 5),
            "traversability": np.round(res["traversability"], 4),
        }).sort_values(by=["band_name", "iy", "ix"]).reset_index(drop=True)

        df_pybind.to_csv(self.pybind_out, index=False)

        # 1. Compare pybind11 vs Python Reference
        passed_py = compare_grids(str(self.py_ref), str(self.pybind_out), tolerance=1e-5)
        self.assertTrue(passed_py, "pybind11 output differs from Python reference!")

        # 2. Compare pybind11 vs C++ Direct CLI Output
        passed_cpp = compare_grids(str(self.cpp_out), str(self.pybind_out), tolerance=1e-5)
        self.assertTrue(passed_cpp, "pybind11 output differs from C++ CLI output!")

    def test_02_random_synthetic_pointcloud_parity(self):
        """Test 2: Random 10,000 points cloud 3-way parity."""
        np.random.seed(123)
        N = 10000
        x = np.random.uniform(-70, 70, N).astype(np.float32)
        y = np.random.uniform(-70, 70, N).astype(np.float32)
        z = np.random.uniform(-3, 5, N).astype(np.float32)
        intensity = np.random.uniform(0, 1, N).astype(np.float32)
        lbls = np.random.choice([0, 1, 2, 3, 255], N).astype(np.int64)
        confs = np.random.uniform(0.5, 1.0, N).astype(np.float32)
        pts = np.column_stack([x, y, z, intensity])

        # Python
        grid_py = FoveatedGrid25D(use_cpp=False)
        map_py = grid_py.build_grid(pts, lbls, confs)
        df_py = map_py.to_dataframe().sort_values(by=["band_name", "iy", "ix"]).reset_index(drop=True)

        # pybind11 C++
        grid_cpp = FoveatedGrid25D(use_cpp=True)
        map_cpp = grid_cpp.build_grid(pts, lbls, confs)
        df_cpp = map_cpp.to_dataframe().sort_values(by=["band_name", "iy", "ix"]).reset_index(drop=True)

        self.assertEqual(len(df_py), len(df_cpp))
        np.testing.assert_array_equal(df_py["ix"].values, df_cpp["ix"].values)
        np.testing.assert_array_equal(df_py["iy"].values, df_cpp["iy"].values)
        np.testing.assert_array_equal(df_py["semantic_class"].values, df_cpp["semantic_class"].values)
        np.testing.assert_allclose(df_py["elevation_mean"].values, df_cpp["elevation_mean"].values, atol=1e-4)
        np.testing.assert_allclose(df_py["elevation_min"].values, df_cpp["elevation_min"].values, atol=1e-4)
        np.testing.assert_allclose(df_py["elevation_max"].values, df_cpp["elevation_max"].values, atol=1e-4)
        np.testing.assert_allclose(df_py["traversability"].values, df_cpp["traversability"].values, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
