"""
Phase 5 Performance Benchmark: Pure C++ Reference Grid Engine vs Python Reference.
Measures latency, throughput, point scales (1K, 10K, 100K, 500K), and verifies output equivalence.
"""

import time
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
import sys
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.foveated_grid import FoveatedGrid25D
from tests.compare_outputs import compare_grids


def run_scaling_benchmark():
    cli_bin = repo_root / "cpp/bin/foveated_grid_cli"
    scales = [1000, 10000, 100000, 500000]
    results = []

    print("=" * 85)
    print("  PHASE 5 PERFORMANCE BENCHMARK: C++ ENGINE VS PYTHON GRID")
    print("=" * 85)
    print(f"{'Point Count':12s} | {'Python Time (ms)':18s} | {'C++ Time (ms)':15s} | {'Speedup':10s} | {'Correctness'}")
    print("-" * 85)

    tmp_dir = repo_root / "benchmarks/tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for N in scales:
        np.random.seed(42)
        x = np.random.uniform(-70, 70, N).astype(np.float32)
        y = np.random.uniform(-70, 70, N).astype(np.float32)
        z = np.random.uniform(-2, 4, N).astype(np.float32)
        intensity = np.random.uniform(0, 1, N).astype(np.float32)
        class_id = np.random.choice([0, 1, 2, 3], size=N).astype(np.int64)
        conf = np.random.uniform(0.7, 1.0, size=N).astype(np.float32)

        pts = np.column_stack([x, y, z, intensity])

        # Write points to CSV for C++
        in_csv = tmp_dir / f"in_{N}.csv"
        df_in = pd.DataFrame({
            "x": x, "y": y, "z": z, "intensity": intensity,
            "class_id": class_id, "confidence": conf
        })
        df_in.to_csv(in_csv, index=False)

        # 1. Benchmark Python Grid
        py_builder = FoveatedGrid25D()
        # Warmup
        for _ in range(2):
            _ = py_builder.build_grid(pts, class_id, conf)

        t0 = time.perf_counter()
        iters = 5 if N <= 100000 else 2
        for _ in range(iters):
            py_grid = py_builder.build_grid(pts, class_id, conf)
        py_ms = (time.perf_counter() - t0) / iters * 1000.0

        # Save Python reference
        py_csv = tmp_dir / f"py_out_{N}.csv"
        df_py = py_grid.to_dataframe()
        out_py = pd.DataFrame({
            "band_name": df_py["band_name"],
            "ix": df_py["ix"].astype(int),
            "iy": df_py["iy"].astype(int),
            "resolution": df_py["resolution"].round(4),
            "point_count": df_py["point_count"].astype(int),
            "elevation_mean": df_py["elevation_mean"].round(5),
            "elevation_min": df_py["elevation_min"].round(5),
            "elevation_max": df_py["elevation_max"].round(5),
            "semantic_class": df_py["semantic_class"].astype(int),
            "confidence": df_py["confidence"].round(5),
            "traversability": df_py["traversability"].round(4),
        }).sort_values(by=["band_name", "iy", "ix"]).reset_index(drop=True)
        out_py.to_csv(py_csv, index=False)

        # 2. Benchmark C++ Grid
        cpp_csv = tmp_dir / f"cpp_out_{N}.csv"
        # Run CLI with benchmark flag
        proc = subprocess.run([
            str(cli_bin),
            "--input", str(in_csv),
            "--output", str(cpp_csv),
            "--benchmark", str(iters)
        ], capture_output=True, text=True)

        cpp_ms = 0.0
        for line in proc.stdout.split("\n"):
            if "Mean Processing Latency:" in line:
                cpp_ms = float(line.split(":")[1].replace("ms", "").strip())
                break

        speedup = py_ms / max(cpp_ms, 1e-4)

        # Correctness check
        passed = compare_grids(str(py_csv), str(cpp_csv), tolerance=1e-4)
        status = "PASS" if passed else "FAIL"

        print(f"{N:10,d} pts | {py_ms:14.2f} ms | {cpp_ms:12.2f} ms | {speedup:8.2f}x | {status}")

        results.append({
            "points": N,
            "python_time_ms": round(py_ms, 3),
            "cpp_time_ms": round(cpp_ms, 3),
            "speedup": round(speedup, 2),
            "python_fps": round(1000.0 / py_ms, 2),
            "cpp_fps": round(1000.0 / max(cpp_ms, 1e-4), 2),
            "correctness": status
        })

    # Save results to benchmarks/phase5_results.csv
    res_df = pd.DataFrame(results)
    out_res = repo_root / "benchmarks/phase5_results.csv"
    res_df.to_csv(out_res, index=False)
    print("-" * 85)
    print(f"Benchmark results successfully written to {out_res}\n")
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    run_scaling_benchmark()
