"""
Full End-to-End Pipeline Verification for Phase 6:
  LiDAR File (.bin)
       ↓
  1. LiDAR Loading (LiDARDataLoader)
       ↓
  2. Range Preprocessing & Filtering (RangeFilter)
       ↓
  3. SPVCNN Neural Inference (Phase2Predictor.predict_frame)
       ↓
  4. ClassifiedPoint[] / Predictions
       ↓
  5. pybind11 Zero-Copy Buffer Interface
       ↓
  6. Pure C++ Foveated Grid Engine
       ↓
  7. 2.5D GridMap25D Object & Downstream Navigation Querying
"""

import sys
import time
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.data_loader import LiDARDataLoader
from src.range_filter import RangeFilter
from phase2.inference.predictor import Phase2Predictor
from src.foveated_grid import FoveatedGrid25D, GridMap25D


def run_full_pipeline_test():
    print("=" * 80)
    print("  PHASE 6 FULL PIPELINE DEMONSTRATION & VERIFICATION")
    print("=" * 80)

    # 1. Load Real LiDAR scan
    loader = LiDARDataLoader(dataset_path=str(repo_root / "dataset"), sequence_id="00")
    frames = loader.discover_frames()
    if len(frames) > 0:
        scan_p, _ = frames[0]
        frame = loader.load_frame(scan_p)
        print(f"1. Loaded Real LiDAR Scan: {len(frame.points):,d} points from {scan_p.name}")
    else:
        print("Warning: No frames discovered, generating representative LiDAR frame")
        raw_pts = np.random.uniform(-50, 50, (66402, 4)).astype(np.float32)
        from src.types import PointCloudFrame
        frame = PointCloudFrame(frame_id="000000", sequence_id="00", timestamp=0.0, points=raw_pts)

    # 2. Preprocessing
    t0 = time.perf_counter()
    rf = RangeFilter(min_range=0.5, max_range=100.0)
    filtered_frame, report = rf.filter_frame(frame)
    t_prep = (time.perf_counter() - t0) * 1000.0
    filtered_pts = filtered_frame.points
    print(f"2. Preprocessed & Range-Filtered: {len(filtered_pts):,d} points ({t_prep:.2f} ms)")

    # 3. SPVCNN Inference
    t0 = time.perf_counter()
    predictor = Phase2Predictor(device="cpu")
    pred = predictor.predict_frame(filtered_frame)
    t_ml = (time.perf_counter() - t0) * 1000.0
    print(f"3. SPVCNN Inference: {len(pred.predicted_class):,d} point predictions ({t_ml:.2f} ms)")

    # 4 + 5 + 6. pybind11 + C++ Foveated Grid Engine
    t0 = time.perf_counter()
    grid_builder = FoveatedGrid25D(use_cpp=True)
    grid_map = grid_builder.build_grid(filtered_pts, pred.predicted_class, pred.confidence)
    t_grid = (time.perf_counter() - t0) * 1000.0
    print(f"4. C++ Foveated Grid Engine (via pybind11): {len(grid_map)} 2.5D cells ({t_grid:.2f} ms)")

    # 7. Downstream Python Query & Export Validation
    df = grid_map.to_dataframe()
    assert len(df) > 0, "DataFrame export is empty!"
    assert "elevation_mean" in df.columns
    assert "traversability" in df.columns
    assert "semantic_class" in df.columns

    # Test spatial query
    sample_cell = grid_map.get_cell_at_xy(2.0, 3.0)
    print(f"5. Downstream Cell Query at (2.0, 3.0): {sample_cell}")

    total_latency = t_prep + t_ml + t_grid
    fps = 1000.0 / total_latency
    print("-" * 80)
    print(f"TOTAL END-TO-END PIPELINE LATENCY: {total_latency:.2f} ms ({fps:.2f} FPS)")
    print(f"  - Preprocessing: {t_prep:.2f} ms ({(t_prep/total_latency)*100:.1f}%)")
    print(f"  - SPVCNN ML:     {t_ml:.2f} ms ({(t_ml/total_latency)*100:.1f}%)")
    print(f"  - C++ Grid Map:  {t_grid:.2f} ms ({(t_grid/total_latency)*100:.1f}%)")
    print("=" * 80)
    print("RESULT: FULL PIPELINE INTEGRATION SUCCESSFUL (PASS)")


if __name__ == "__main__":
    run_full_pipeline_test()
