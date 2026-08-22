"""
Independent boundary, edge case, and sanitizer verification script.
"""

import sys
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.foveated_grid import FoveatedGrid25D, distance_to_band, xy_to_cell, DEFAULT_FROZEN_BANDS
from tests.compare_outputs import compare_grids

def test_boundaries():
    print("Testing distance band resolution boundaries:")
    test_radii = [
        (0.0, "near_field", 0.05),
        (5.0, "near_field", 0.05),
        (9.9999, "near_field", 0.05),
        (10.0, "mid_near_field", 0.10),
        (10.0001, "mid_near_field", 0.10),
        (29.9999, "mid_near_field", 0.10),
        (30.0, "mid_far_field", 0.25),
        (30.0001, "mid_far_field", 0.25),
        (59.9999, "mid_far_field", 0.25),
        (60.0, "far_field", 0.50),
        (60.0001, "far_field", 0.50),
        (99.9999, "far_field", 0.50),
        (100.0, None, None),
        (100.001, None, None),
        (-1.0, None, None),
        (float("nan"), None, None),
        (float("inf"), None, None),
        (float("-inf"), None, None),
    ]

    for r, exp_band, exp_res in test_radii:
        b = distance_to_band(r)
        if exp_band is None:
            assert b is None, f"Expected None for r={r}, got {b}"
        else:
            assert b is not None and b.name == exp_band and abs(b.voxel_size - exp_res) < 1e-6, f"Failed for r={r}: {b}"

    print("[+] All Python distance band boundary checks passed!")

def test_floor_indexing():
    print("Testing mathematical floor coordinate indexing:")
    test_coords = [
        (0.0, 0.0, 0.05, 0, 0),
        (0.0499, 0.0499, 0.05, 0, 0),
        (0.05, 0.05, 0.05, 1, 1),
        (-0.0001, -0.0001, 0.05, -1, -1),
        (-0.05, -0.05, 0.05, -1, -1),
        (-0.0501, -0.0501, 0.05, -2, -2),
        (2.01, 3.02, 0.05, 40, 60),
        (-4.25, -3.15, 0.05, -85, -63),
    ]

    for x, y, res, exp_ix, exp_iy in test_coords:
        ix, iy = xy_to_cell(x, y, res)
        assert ix == exp_ix and iy == exp_iy, f"Failed xy_to_cell({x}, {y}, {res}): got ({ix}, {iy}), expected ({exp_ix}, {exp_iy})"

    print("[+] All mathematical floor indexing tests passed!")

if __name__ == "__main__":
    test_boundaries()
    test_floor_indexing()
