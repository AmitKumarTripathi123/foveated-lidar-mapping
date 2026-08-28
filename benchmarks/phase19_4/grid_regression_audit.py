"""
Phase 19.4 Grid Regression Audit (SIH PS 26130).
Validates that Grid latency recovers to <= 7.76 ms (Phase 19.2 baseline)
while maintaining 100% cell set and semantic equivalence.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.core.foveated_grid import HierarchicalFoveatedGridEngine
from src.core.native_grid import NativeGridMapRasterizer
from benchmarks.phase19_2.correctness_audit import compare_grid_maps


def run_grid_regression_audit(
    num_frames: int = 100,
    warmup_frames: int = 10,
    out_json: Path = Path("reports/phase19_4/grid_regression_audit.json"),
) -> Dict[str, Any]:
    """Execute grid regression audit across 100 evaluation frames."""
    rasterizer = NativeGridMapRasterizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(42)
    test_clouds = []
    for _ in range(num_frames + warmup_frames):
        xyz = np.random.uniform(-45.0, 45.0, (48000, 3)).astype(np.float32)
        c = np.random.randint(0, 4, 48000).astype(np.int64)
        conf = np.random.uniform(0.5, 1.0, 48000).astype(np.float32)
        test_clouds.append((xyz, c, conf))

    # 1. Warmup
    print("Warming up grid rasterizer...")
    for i in range(warmup_frames):
        xyz, c, conf = test_clouds[i]
        _ = rasterizer.rasterize(xyz, c, conf)
        if device.type == "cuda":
            t_x = torch.from_numpy(xyz).to(device)
            t_c = torch.from_numpy(c).to(device)
            t_conf = torch.from_numpy(conf).to(device)
            _ = rasterizer.rasterize(t_x, t_c, t_conf, mode="cuda")

    # 2. Benchmark GPU Grid Latency
    grid_latencies = []
    if device.type == "cuda":
        cuda_clouds = [(torch.from_numpy(x).to(device), torch.from_numpy(c).to(device), torch.from_numpy(cf).to(device)) for x, c, cf in test_clouds]
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)

        for i in range(warmup_frames, len(cuda_clouds)):
            t_xyz, t_c, t_conf = cuda_clouds[i]

            start_ev.record()
            _ = rasterizer.rasterize(t_xyz, t_c, t_conf, mode="cuda")
            end_ev.record()
            torch.cuda.synchronize()
            grid_latencies.append(float(start_ev.elapsed_time(end_ev)))

    mean_grid_ms = round(float(np.mean(grid_latencies)), 2)
    p95_grid_ms = round(float(np.percentile(grid_latencies, 95)), 2)

    # 3. Equivalence check against reference
    xyz_eval, c_eval, conf_eval = test_clouds[0]
    engine = HierarchicalFoveatedGridEngine()
    grid_ref = engine.build_25d_grid_reference_python(xyz_eval, c_eval, conf_eval)
    grid_nat = rasterizer.rasterize(xyz_eval, c_eval, conf_eval)
    cmp_res = compare_grid_maps(grid_ref, grid_nat)

    payload = {
        "status": "GRID_REGRESSION_RECOVERED" if mean_grid_ms <= 7.76 else "GRID_REGRESSION_ACTIVE",
        "phase19_2_baseline_ms": 7.76,
        "phase19_3_regressed_ms": 12.14,
        "phase19_4_measured_ms": mean_grid_ms,
        "phase19_4_p95_ms": p95_grid_ms,
        "target_met": mean_grid_ms <= 7.76,
        "equivalence": cmp_res,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload


if __name__ == "__main__":
    res = run_grid_regression_audit()
    print(f"Grid Regression Audit Summary: Measured {res['phase19_4_measured_ms']} ms (Target <= 7.76 ms: {res['target_met']})")
