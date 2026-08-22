#!/usr/bin/env python3
"""LiDAR Dataset Preprocessing CLI Tool (Phase 2).

Usage:
    # 1. Keep-all preprocessing
    python scripts/preprocess_lidar.py \
        --scan dataset/sequences/00/velodyne/000000.bin \
        --label dataset/sequences/00/labels/000000.label \
        --strategy keep_all

    # 2. Controlled sampling with PyTorch tensor conversion
    python scripts/preprocess_lidar.py \
        --scan dataset/sequences/00/velodyne/000000.bin \
        --label dataset/sequences/00/labels/000000.label \
        --num-points 16384 \
        --seed 42 \
        --to-tensor
"""

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.dataset import load_point_cloud, load_labels
from ml.data.preprocessing import (
    LidarPreprocessor,
    PreprocessingConfig,
)

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def print_processing_report(sample, to_tensor: bool = False) -> None:
    """Print clean terminal processing report."""
    rep = sample.report

    print("=" * 60)
    print("           LiDAR PREPROCESSING REPORT")
    print("=" * 60)

    print(f"\nSequence : {rep.sequence or 'N/A'}")
    print(f"Frame    : {rep.frame or 'N/A'}")

    print("\nPoint Counts:")
    print(f"  Original points   : {rep.original_point_count:,}")
    print(f"  Invalid removed   : {rep.invalid_points_removed:,}")
    print(f"  Range filtered    : {rep.range_filtered_points:,}")
    print(f"  Sampled points    : {rep.sampled_points_count:,}")
    print(f"  Final points      : {rep.final_point_count:,}")
    if rep.final_label_count is not None:
        print(f"  Final labels      : {rep.final_label_count:,}")

    print("\nData Types:")
    print(f"  Input dtype       : {rep.input_dtype}")
    print(f"  Output dtype      : {rep.output_dtype}")

    print("\nSampling Configuration:")
    print(f"  Strategy          : {rep.sampling_strategy}")
    print(f"  Random seed       : {rep.sampling_seed}")

    print("\nValidation Status:")
    align_status = "[PASS]" if rep.alignment_pass else "[FAIL]"
    print(f"  {align_status} Point-label alignment verified ({rep.final_point_count} == {rep.final_label_count})")

    if to_tensor:
        print("\nPyTorch Tensor Conversion:")
        if _HAS_TORCH:
            points_tensor = torch.from_numpy(sample.points).float()
            labels_tensor = (
                torch.from_numpy(sample.labels.astype(int)).long()
                if sample.labels is not None
                else None
            )
            print(f"  Points tensor shape : {tuple(points_tensor.shape)} (dtype={points_tensor.dtype})")
            if labels_tensor is not None:
                print(f"  Labels tensor shape : {tuple(labels_tensor.shape)} (dtype={labels_tensor.dtype})")
            print("  [PASS] PyTorch tensor conversion verified")
        else:
            print("  [!] PyTorch not installed in environment")

    print("\n" + "=" * 60)
    if rep.passed:
        print("                 PREPROCESSING: PASS")
    else:
        print("                 PREPROCESSING: FAIL")
    print("=" * 60 + "\n")


def main() -> int:
    """CLI entrypoint for preprocessing."""
    parser = argparse.ArgumentParser(
        description="Preprocess and sample a LiDAR scan and label file for model readiness."
    )
    parser.add_argument(
        "--scan",
        type=str,
        default="dataset/sequences/00/velodyne/000000.bin",
        help="Path to .bin file",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="dataset/sequences/00/labels/000000.label",
        help="Path to .label file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to preprocessing.yaml configuration",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Sampling strategy: keep_all, random, deterministic, random_with_replacement, pad",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=None,
        help="Target number of points to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling reproducibility",
    )
    parser.add_argument(
        "--to-tensor",
        action="store_true",
        help="Test PyTorch tensor conversion",
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = PreprocessingConfig.from_yaml(args.config)
    else:
        config = PreprocessingConfig()

    # Apply CLI overrides
    if args.strategy:
        config.sampling.strategy = args.strategy
    if args.num_points is not None:
        config.sampling.num_points = args.num_points
        if args.strategy is None:
            config.sampling.strategy = "random"
    if args.seed is not None:
        config.sampling.seed = args.seed

    # Load raw data via Phase 1 loader
    scan_path = Path(args.scan)
    label_path = Path(args.label) if args.label and Path(args.label).is_file() else None

    points = load_point_cloud(scan_path)
    labels = load_labels(label_path) if label_path else None

    metadata = {
        "sequence": scan_path.parent.parent.name if scan_path.parent.parent.name.isdigit() else "00",
        "frame": scan_path.stem,
        "scan_path": str(scan_path),
        "label_path": str(label_path) if label_path else None,
    }

    # Run preprocessor
    preprocessor = LidarPreprocessor(config)
    processed = preprocessor(points, labels, metadata=metadata)

    print_processing_report(processed, to_tensor=args.to_tensor)

    return 0 if processed.report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
