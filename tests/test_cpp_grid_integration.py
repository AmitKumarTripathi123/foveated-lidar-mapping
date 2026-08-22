"""
Integration Unit Tests for C++ Foveated Grid Engine.
Verifies compilation, execution, golden comparison, and edge-case behavior.
"""

import subprocess
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
from tests.compare_outputs import compare_grids


class TestCPPGridIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli_bin = repo_root / "cpp/bin/foveated_grid_cli"
        cls.test_bin = repo_root / "cpp/bin/foveated_grid_tests"
        cls.golden_in = repo_root / "tests/data/phase5_golden_input.csv"
        cls.py_ref = repo_root / "tests/reference/python_grid.csv"
        cls.cpp_out = repo_root / "tests/output/cpp_grid.csv"

        # Build C++ if binaries do not exist
        if not cls.cli_bin.exists() or not cls.test_bin.exists():
            res = subprocess.run(["./cpp/build.sh"], cwd=str(repo_root), capture_output=True, text=True)
            assert res.returncode == 0, f"C++ build failed: {res.stderr}"

    def test_01_cpp_unit_tests_pass(self):
        """Test 1: Standalone C++ unit test binary executes and passes 100%."""
        res = subprocess.run([str(self.test_bin)], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"C++ test runner failed: {res.stderr}")
        self.assertIn("ALL C++ UNIT TESTS PASSED", res.stdout)

    def test_02_golden_output_matches_python_reference(self):
        """Test 2: Golden test input produces bitwise/numerical match with Python reference."""
        res = subprocess.run([
            str(self.cli_bin),
            "--input", str(self.golden_in),
            "--output", str(self.cpp_out)
        ], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"C++ CLI execution failed: {res.stderr}")

        passed = compare_grids(str(self.py_ref), str(self.cpp_out), tolerance=1e-5)
        self.assertTrue(passed, "C++ output differed from Python reference!")

    def test_03_empty_input(self):
        """Test 3: Empty input produces empty grid CSV without crashing."""
        empty_csv = repo_root / "tests/data/empty_in.csv"
        empty_out = repo_root / "tests/output/empty_out.csv"
        empty_csv.write_text("x,y,z,intensity,class_id,confidence\n", encoding="utf-8")

        res = subprocess.run([
            str(self.cli_bin),
            "--input", str(empty_csv),
            "--output", str(empty_out)
        ], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertTrue(empty_out.exists())

        out_df = pd.read_csv(empty_out)
        self.assertEqual(len(out_df), 0)


if __name__ == "__main__":
    unittest.main()
