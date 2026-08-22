#!/usr/bin/env python3
"""Offline 3-Zone Foveated Voxel Preprocessing and Cache CLI (Master Task).

Usage:
    python scripts/preprocess_foveated.py --manifest data_manifest.json --out-dir processed
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.dataset import load_point_cloud, load_labels
from ml.data.preprocessing import filter_invalid_points
from ml.data.amit_adapter import FoveatedVoxelSampler
from ml.data.label_mapping import SemanticLabelRemapper


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Preprocess and cache foveated LiDAR point clouds.")
    parser.add_argument("--manifest", type=str, default="data_manifest.json", help="Path to data_manifest.json")
    parser.add_argument("--out-dir", type=str, default="processed", help="Directory to save cached .npy files")

    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"Error: Manifest not found at {manifest_path.resolve()}. Run scripts/generate_manifest.py first.")
        return 1

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    out_base = Path(args.out_dir)
    sampler = FoveatedVoxelSampler()
    remapper = SemanticLabelRemapper()

    all_reports = []

    for split in ["train", "val", "test"]:
        records = manifest.get(split, [])
        if not records:
            continue

        split_out = out_base / split
        split_out.mkdir(parents=True, exist_ok=True)

        for rec in records:
            seq = rec["sequence"]
            frame = rec["frame"]
            raw_pts = load_point_cloud(rec["point_path"])
            raw_lbls = load_labels(rec["label_path"]) if rec.get("label_path") else None

            # 1. Invalid removal
            v_pts, v_lbls, _ = filter_invalid_points(raw_pts, raw_lbls)

            # 2. Amit's 3-Zone Foveated Voxel Sampling
            fov_pts, fov_lbls, report = sampler.sample(v_pts, v_lbls)
            all_reports.append((seq, frame, report))

            # 3. SIH Label Remapping
            if fov_lbls is not None:
                sih_lbls = remapper.remap(fov_lbls)
            else:
                sih_lbls = None

            # 4. Save to cache
            pts_out = split_out / f"{seq}_{frame}_pts.npy"
            lbl_out = split_out / f"{seq}_{frame}_lbl.npy"

            np.save(pts_out, fov_pts)
            if sih_lbls is not None:
                np.save(lbl_out, sih_lbls)

            print(
                f"[{split.upper()}] Processed {seq}/{frame}: "
                f"{report.original_count:,} pts -> {report.foveated_count:,} pts "
                f"({report.overall_reduction_pct:.2f}% reduction)"
            )

    print("\n" + "=" * 78)
    print("                 AMIT FOVEATED VOXEL SAMPLING PERFORMANCE REPORT")
    print("=" * 78)
    for seq, frame, rep in all_reports:
        print(f"\nScan: Sequence {seq} / Frame {frame}")
        print(f"  Original Points : {rep.original_count:,}")
        print(f"  Foveated Points : {rep.foveated_count:,}")
        print(f"  Total Reduction : {rep.overall_reduction_pct:.2f}%\n")
        print(f"  {'Zone Name':<24} | {'Voxel Size':<10} | {'Input Pts':>10} | {'Output Pts':>10} | {'Reduction':>10}")
        print("  " + "-" * 74)
        for zs in rep.zone_stats:
            print(
                f"  {zs.zone_name:<24} | {zs.voxel_size:8.2f}m | {zs.input_count:10,d} | "
                f"{zs.output_count:10,d} | {zs.reduction_pct:9.2f}%"
            )
        print("  " + "-" * 74)
    print("=" * 78 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
