"""
Generates deterministic Phase 5 golden test dataset and executes the Python reference engine.
Produces:
  - tests/data/phase5_golden_input.csv
  - tests/reference/python_grid.csv
"""

import sys
import math
from pathlib import Path
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.types import SuperClass
from src.foveated_grid import FoveatedGrid25D, DEFAULT_FROZEN_BANDS


def build_golden_input() -> pd.DataFrame:
    rows = []

    # 1. Origin point
    rows.append({"x": 0.0, "y": 0.0, "z": 0.0, "intensity": 0.5, "class_id": 0, "confidence": 0.95})

    # 2. Near field normal points [0.0, 10.0) m, res=0.05m
    rows.append({"x": 2.01, "y": 3.02, "z": 1.10, "intensity": 0.8, "class_id": 0, "confidence": 0.90})
    rows.append({"x": 2.03, "y": 3.04, "z": 1.50, "intensity": 0.7, "class_id": 1, "confidence": 0.85})  # Same cell (40, 60)
    rows.append({"x": 2.04, "y": 3.01, "z": 0.90, "intensity": 0.9, "class_id": 2, "confidence": 0.92})  # Same cell -> Static Obstacle overrides!

    # 3. Negative coordinates in near field
    rows.append({"x": -0.02, "y": -0.03, "z": -0.50, "intensity": 0.6, "class_id": 0, "confidence": 0.88})  # Cell (-1, -1)
    rows.append({"x": -0.04, "y": -0.01, "z": -0.20, "intensity": 0.6, "class_id": 3, "confidence": 0.99})  # Cell (-1, -1) -> Dynamic Object overrides!
    rows.append({"x": -4.25, "y": -3.15, "z": 0.35, "intensity": 0.4, "class_id": 1, "confidence": 0.80})   # Cell (-85, -63)

    # 4. Exact Band Boundaries
    # Near/Mid-Near boundary: r = 10.0m
    rows.append({"x": 9.99, "y": 0.0, "z": 0.10, "intensity": 0.5, "class_id": 0, "confidence": 0.90})     # r = 9.99m -> near_field (res=0.05)
    rows.append({"x": 10.00, "y": 0.0, "z": 0.20, "intensity": 0.5, "class_id": 0, "confidence": 0.90})    # r = 10.00m -> mid_near_field (res=0.10)

    # Mid-Near/Mid-Far boundary: r = 30.0m
    rows.append({"x": 0.0, "y": 29.99, "z": 0.30, "intensity": 0.5, "class_id": 2, "confidence": 0.85})    # r = 29.99m -> mid_near_field (res=0.10)
    rows.append({"x": 0.0, "y": 30.00, "z": 0.40, "intensity": 0.5, "class_id": 2, "confidence": 0.85})    # r = 30.00m -> mid_far_field (res=0.25)

    # Mid-Far/Far boundary: r = 60.0m
    rows.append({"x": 59.99, "y": 0.0, "z": 0.50, "intensity": 0.5, "class_id": 1, "confidence": 0.75})    # r = 59.99m -> mid_far_field (res=0.25)
    rows.append({"x": 60.00, "y": 0.0, "z": 0.60, "intensity": 0.5, "class_id": 1, "confidence": 0.75})    # r = 60.00m -> far_field (res=0.50)

    # Far field upper boundary: r = 99.99m
    rows.append({"x": -70.70, "y": 70.70, "z": -1.20, "intensity": 0.3, "class_id": 2, "confidence": 0.70}) # r = 99.985m -> far_field (res=0.50)

    # 5. Out of Range & Invalid Points (Should be dropped by reference engine)
    rows.append({"x": 100.00, "y": 0.0, "z": 0.0, "intensity": 0.1, "class_id": 0, "confidence": 0.50})    # r = 100.0m (>= 100)
    rows.append({"x": -80.00, "y": -80.00, "z": 0.0, "intensity": 0.1, "class_id": 0, "confidence": 0.50}) # r = 113.1m (>= 100)
    rows.append({"x": np.nan, "y": 5.0, "z": 1.0, "intensity": 0.0, "class_id": 0, "confidence": 0.0})     # NaN coordinate

    # 6. Multiple points in Far Field with mixed semantic priorities
    rows.append({"x": -45.10, "y": 45.10, "z": 2.00, "intensity": 0.5, "class_id": 0, "confidence": 0.80})  # Far cell (-91, 90)
    rows.append({"x": -45.20, "y": 45.30, "z": 3.50, "intensity": 0.6, "class_id": 2, "confidence": 0.90})  # Same cell -> Static Obstacle
    rows.append({"x": -45.40, "y": 45.40, "z": 1.00, "intensity": 0.4, "class_id": 3, "confidence": 0.95})  # Same cell -> Dynamic Object (Highest!)

    return pd.DataFrame(rows)


def run_python_reference(input_df: pd.DataFrame) -> pd.DataFrame:
    pts = input_df[["x", "y", "z", "intensity"]].values.astype(np.float32)
    lbls = input_df["class_id"].values.astype(np.int64)
    confs = input_df["confidence"].values.astype(np.float32)

    grid_builder = FoveatedGrid25D()
    grid_map = grid_builder.build_grid(pts, lbls, confs)
    df = grid_map.to_dataframe()

    # Select standard golden reference fields and sort deterministically
    out_df = pd.DataFrame({
        "band_name": df["band_name"],
        "ix": df["ix"].astype(int),
        "iy": df["iy"].astype(int),
        "resolution": df["resolution"].round(4),
        "point_count": df["point_count"].astype(int),
        "elevation_mean": df["elevation_mean"].round(5),
        "elevation_min": df["elevation_min"].round(5),
        "elevation_max": df["elevation_max"].round(5),
        "semantic_class": df["semantic_class"].astype(int),
        "confidence": df["confidence"].round(5),
        "traversability": df["traversability"].round(4),
    })

    out_df = out_df.sort_values(by=["band_name", "iy", "ix"]).reset_index(drop=True)
    return out_df


if __name__ == "__main__":
    in_df = build_golden_input()
    in_path = Path("tests/data/phase5_golden_input.csv")
    in_path.parent.mkdir(parents=True, exist_ok=True)
    in_df.to_csv(in_path, index=False)
    print(f"Generated Golden Input at {in_path} ({len(in_df)} points)")

    ref_df = run_python_reference(in_df)
    ref_path = Path("tests/reference/python_grid.csv")
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref_df.to_csv(ref_path, index=False)
    print(f"Generated Python Reference Grid at {ref_path} ({len(ref_df)} cells)")
    print(ref_df.to_string())
