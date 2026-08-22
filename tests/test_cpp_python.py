"""
Test pybind11 module bindings and basic Python <-> C++ interoperability.
"""

import unittest
import numpy as np
import foveated_grid_cpp
from src.foveated_grid import FoveatedGrid25D, DEFAULT_FROZEN_BANDS


class TestCPPPythonBindings(unittest.TestCase):
    def test_01_module_attributes(self):
        """Test 1: Check module symbols and classes."""
        self.assertTrue(hasattr(foveated_grid_cpp, "FoveatedGridEngine"))
        self.assertTrue(hasattr(foveated_grid_cpp, "ClassifiedPoint"))
        self.assertTrue(hasattr(foveated_grid_cpp, "GridCell"))
        self.assertTrue(hasattr(foveated_grid_cpp, "FoveationBand"))
        self.assertTrue(hasattr(foveated_grid_cpp, "SuperClass"))

    def test_02_point_and_cell_constructors(self):
        """Test 2: Struct construction and field access."""
        pt = foveated_grid_cpp.ClassifiedPoint(1.0, 2.0, 3.0, 0.5, 2, 0.95)
        self.assertAlmostEqual(pt.x, 1.0)
        self.assertAlmostEqual(pt.y, 2.0)
        self.assertAlmostEqual(pt.z, 3.0)
        self.assertEqual(pt.class_id, 2)
        self.assertAlmostEqual(pt.confidence, 0.95)

        band = foveated_grid_cpp.FoveationBand("near", 0.0, 10.0, 0.05)
        self.assertTrue(band.contains(5.0))
        self.assertFalse(band.contains(10.0))

    def test_03_engine_object_interface(self):
        """Test 3: Engine build_grid with ClassifiedPoint list."""
        engine = foveated_grid_cpp.FoveatedGridEngine()
        pts = [
            foveated_grid_cpp.ClassifiedPoint(2.01, 3.02, 1.0, 0.5, 0, 0.9),
            foveated_grid_cpp.ClassifiedPoint(2.02, 3.03, 3.0, 0.5, 2, 0.95),
        ]
        cells = engine.build_grid(pts)
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0].ix, 40)
        self.assertEqual(cells[0].iy, 60)
        self.assertEqual(cells[0].point_count, 2)
        self.assertAlmostEqual(cells[0].elevation_mean, 2.0)
        self.assertEqual(cells[0].semantic_class, 2)

    def test_04_engine_numpy_interface(self):
        """Test 4: Engine build_grid_numpy with raw NumPy arrays."""
        engine = foveated_grid_cpp.FoveatedGridEngine()
        pts = np.array([[2.01, 3.02, 1.0, 0.5], [2.02, 3.03, 3.0, 0.5]], dtype=np.float32)
        lbls = np.array([0, 2], dtype=np.int64)
        confs = np.array([0.9, 0.95], dtype=np.float32)

        res = engine.build_grid_numpy(pts, lbls, confs)
        self.assertEqual(res["num_cells"], 1)
        self.assertEqual(res["ix"][0], 40)
        self.assertEqual(res["iy"][0], 60)
        self.assertEqual(res["semantic_class"][0], 2)
        self.assertEqual(res["point_count"][0], 2)

    def test_05_foveated_grid_transparent_dispatch(self):
        """Test 5: FoveatedGrid25D uses C++ engine and returns identical GridMap25D."""
        grid_cpp = FoveatedGrid25D(use_cpp=True)
        grid_py = FoveatedGrid25D(use_cpp=False)

        pts = np.random.uniform(-20, 20, (1000, 4)).astype(np.float32)
        lbls = np.random.choice([0, 1, 2, 3], 1000).astype(np.int64)
        confs = np.random.uniform(0.7, 1.0, 1000).astype(np.float32)

        map_cpp = grid_cpp.build_grid(pts, lbls, confs)
        map_py = grid_py.build_grid(pts, lbls, confs)

        df_cpp = map_cpp.to_dataframe().sort_values(by=["band_name", "iy", "ix"]).reset_index(drop=True)
        df_py = map_py.to_dataframe().sort_values(by=["band_name", "iy", "ix"]).reset_index(drop=True)

        self.assertEqual(len(df_cpp), len(df_py))
        np.testing.assert_array_equal(df_cpp["ix"].values, df_py["ix"].values)
        np.testing.assert_array_equal(df_cpp["iy"].values, df_py["iy"].values)
        np.testing.assert_array_equal(df_cpp["semantic_class"].values, df_py["semantic_class"].values)
        np.testing.assert_allclose(df_cpp["elevation_mean"].values, df_py["elevation_mean"].values, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
