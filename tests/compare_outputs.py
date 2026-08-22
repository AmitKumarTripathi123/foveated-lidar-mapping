#!/usr/bin/env python3
"""
Phase 5 Primary Gate: Deterministic Python Reference vs C++ Engine Output Comparison.
Validates:
  - Exact cell count match
  - Exact coordinate (band_name, ix, iy) match
  - Exact resolution match
  - Exact point count match
  - Elevation mean/min/max within strict floating-point tolerance (|diff| <= 1e-5)
  - Exact semantic class priority match
  - Confidence within tolerance (|diff| <= 1e-5)
  - Traversability within tolerance (|diff| <= 1e-5)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np


def compare_grids(python_csv: str, cpp_csv: str, tolerance: float = 1e-5) -> bool:
    py_path = Path(python_csv)
    cpp_path = Path(cpp_csv)

    if not py_path.exists():
        print(f"[-] ERROR: Python reference CSV not found: {py_path}", file=sys.stderr)
        return False
    if not cpp_path.exists():
        print(f"[-] ERROR: C++ output CSV not found: {cpp_path}", file=sys.stderr)
        return False

    py_df = pd.read_csv(py_path).sort_values(by=["band_name", "iy", "ix"]).reset_index(drop=True)
    cpp_df = pd.read_csv(cpp_path).sort_values(by=["band_name", "iy", "ix"]).reset_index(drop=True)

    print("=" * 80)
    print("  PHASE 5 CORRECTNESS GATE: PYTHON VS C++ GRID OUTPUT COMPARISON")
    print("=" * 80)
    print(f"Python Reference: {py_path} ({len(py_df)} cells)")
    print(f"C++ Engine:       {cpp_path} ({len(cpp_df)} cells)")
    print(f"Tolerance:        {tolerance}")
    print("-" * 80)

    if len(py_df) != len(cpp_df):
        print(f"[-] FAIL: Cell count mismatch! Python={len(py_df)}, C++={len(cpp_df)}")
        return False

    all_passed = True
    mismatches = []

    for i in range(len(py_df)):
        py_row = py_df.iloc[i]
        cpp_row = cpp_df.iloc[i]

        cell_id = f"[{py_row['band_name']}] ({py_row['ix']}, {py_row['iy']})"

        # Check discrete fields (exact match)
        for field in ["band_name", "ix", "iy", "point_count", "semantic_class"]:
            if py_row[field] != cpp_row[field]:
                mismatches.append(f"FAIL: cell={cell_id} field={field} python={py_row[field]} cpp={cpp_row[field]}")
                all_passed = False

        # Check continuous fields (with strict tolerance)
        for field in ["resolution", "elevation_mean", "elevation_min", "elevation_max", "confidence", "traversability"]:
            val_py = float(py_row[field])
            val_cpp = float(cpp_row[field])
            diff = abs(val_py - val_cpp)
            if diff > tolerance:
                mismatches.append(f"FAIL: cell={cell_id} field={field} python={val_py:.6f} cpp={val_cpp:.6f} diff={diff:.8f}")
                all_passed = False

    if not all_passed:
        print("[-] MISMATCHES DETECTED:")
        for m in mismatches[:20]:
            print(f"    {m}")
        print("-" * 80)
        print("RESULT: FAIL")
        return False

    print("[+] All discrete fields matched exactly.")
    print(f"[+] All floating-point fields matched within tolerance (max diff <= {tolerance}).")
    print("-" * 80)
    print("RESULT: PASS")
    print("=" * 80)
    return True


if __name__ == "__main__":
    py_f = sys.argv[1] if len(sys.argv) > 1 else "tests/reference/python_grid.csv"
    cpp_f = sys.argv[2] if len(sys.argv) > 2 else "tests/output/cpp_grid.csv"
    success = compare_grids(py_f, cpp_f)
    sys.exit(0 if success else 1)
