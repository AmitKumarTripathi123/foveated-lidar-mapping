#!/usr/bin/env python3
"""LiDAR Dataset Inspection and Validation CLI Tool.

Usage:
    python scripts/inspect_lidar.py \
        --scan dataset/sequences/00/velodyne/000000.bin \
        --label dataset/sequences/00/labels/000000.label
"""

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.dataset import validate_dataset_pair


def format_status(passed: bool) -> str:
    """Format boolean check status with icon."""
    return "[PASS]" if passed else "[FAIL]"


def print_report(report) -> None:
    """Print a clean, structured inspection report to stdout."""
    print("=" * 60)
    print("           LiDAR DATASET INSPECTION REPORT")
    print("=" * 60)

    # Point cloud information
    print("\nPoint Cloud:")
    print(f"  Path      : {report.scan_path}")
    print(f"  Exists    : {'YES' if report.scan_exists else 'NO'}")
    print(f"  Readable  : {'YES' if report.scan_readable else 'NO'}")
    print(f"  Dtype     : {report.point_dtype}")
    print(f"  Shape     : {report.points_shape}")
    print(f"  Points    : {report.num_points:,}")

    # Coordinates statistics
    if report.stats is not None:
        s = report.stats
        print("\nCoordinates (meters):")
        print(f"  X : min={s.x.min:9.3f}, max={s.x.max:9.3f}, mean={s.x.mean:9.3f}, std={s.x.std:9.3f}")
        print(f"  Y : min={s.y.min:9.3f}, max={s.y.max:9.3f}, mean={s.y.mean:9.3f}, std={s.y.std:9.3f}")
        print(f"  Z : min={s.z.min:9.3f}, max={s.z.max:9.3f}, mean={s.z.mean:9.3f}, std={s.z.std:9.3f}")

        print("\nIntensity (reflectance):")
        print(f"  min  : {s.intensity.min:9.3f}")
        print(f"  max  : {s.intensity.max:9.3f}")
        print(f"  mean : {s.intensity.mean:9.3f}")
        print(f"  std  : {s.intensity.std:9.3f}")

    # Labels information
    if report.label_path is not None:
        print("\nLabels:")
        print(f"  Path            : {report.label_path}")
        print(f"  Exists          : {'YES' if report.label_exists else 'NO'}")
        print(f"  Readable        : {'YES' if report.label_readable else 'NO'}")
        print(f"  Extracted Dtype : {report.label_dtype}")
        print(f"  Shape           : {report.labels_shape}")
        print(f"  Count           : {report.num_labels:,}" if report.num_labels is not None else "  Count : N/A")
        print(f"  Unique classes  : {len(report.label_distribution)}")

        if report.label_distribution:
            print("\n  Semantic Label Distribution:")
            print("  " + "-" * 48)
            print(f"  {'Class ID':>8} | {'Point Count':>12} | {'Percentage':>10}")
            print("  " + "-" * 48)
            for entry in report.label_distribution:
                print(f"  {entry.label_id:8d} | {entry.count:12,d} | {entry.percentage:9.2f}%")
            print("  " + "-" * 48)

    # Validation checklist
    print("\nValidation Checklist:")
    checks = [
        ("Point cloud file exists", report.scan_exists),
        ("Point cloud readable", report.scan_readable),
        ("Point cloud dtype = float32", report.point_dtype == "float32"),
        ("Point cloud shape = (N, 4)", len(report.points_shape) == 2 and report.points_shape[1] == 4),
        ("No unexpected NaN values", report.nan_check_pass),
        ("No unexpected Inf values", report.inf_check_pass),
        ("XYZ statistics available", report.stats is not None),
        ("Intensity statistics available", report.stats is not None),
    ]

    if report.label_path is not None:
        checks.extend([
            ("Label file exists", report.label_exists),
            ("Label file readable", report.label_readable),
            ("Semantic labels extracted using & 0xFFFF", report.label_readable),
            ("Labels shape = (N,)", report.labels_shape is not None and len(report.labels_shape) == 1),
            ("Point count == label count", report.alignment_pass),
            ("Label distribution available", len(report.label_distribution) > 0),
        ])

    for desc, passed in checks:
        icon = "[PASS]" if passed else "[FAIL]"
        print(f"  {icon} {desc}")

    if report.errors:
        print("\nErrors Encountered:")
        for err in report.errors:
            print(f"  [!] {err}")

    print("\n" + "=" * 60)
    if report.passed:
        print("                 PHASE 1 VALIDATION: PASS")
    else:
        print("                 PHASE 1 VALIDATION: FAIL")
    print("=" * 60 + "\n")


def main() -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Inspect and validate a SemanticKITTI LiDAR scan and label file."
    )
    parser.add_argument(
        "--scan",
        type=str,
        default="dataset/sequences/00/velodyne/000000.bin",
        help="Path to LiDAR .bin file (default: dataset/sequences/00/velodyne/000000.bin)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="dataset/sequences/00/labels/000000.label",
        help="Path to semantic .label file (default: dataset/sequences/00/labels/000000.label)",
    )

    args = parser.parse_args()

    report = validate_dataset_pair(args.scan, args.label)
    print_report(report)

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
