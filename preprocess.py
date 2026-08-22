"""
preprocess.py
=============
Run the pipeline once over every frame in the chosen train/val sequences,
cache the processed (range-filtered, downsampled, remapped) output as
.npy files, and print the per-class point distribution so you know up
front whether you need class-weighted loss.

Usage:
    python3 preprocess.py
    python3 preprocess.py --dataset-root /path/to/SemanticPOSS --max-frames 10
"""

import argparse
import os
import json
import numpy as np

from dataset import FoveatedLidarDataset, build_file_list
from class_map import PROJECT_CLASSES, compute_class_weights

DEFAULT_DATASET_ROOT = os.environ.get("DATASET_ROOT", "dataset")
DEFAULT_OUTPUT_DIR = "processed"
DEFAULT_TRAIN_SEQUENCES = ["00", "01", "03", "04", "05"]
DEFAULT_VAL_SEQUENCES = ["02"]


def process_split(split_name, sequences, dataset_root=DEFAULT_DATASET_ROOT, output_dir=DEFAULT_OUTPUT_DIR, max_frames=None):
    bin_paths, label_paths = build_file_list(dataset_root, sequences, max_frames=max_frames)
    print(f"[{split_name}] {len(bin_paths)} frames from sequences {sequences}")

    if len(bin_paths) == 0:
        print(f"  [{split_name}] WARNING: 0 frames found for sequences {sequences}")
        return {
            "num_frames": 0,
            "total_points": 0,
            "class_counts": [0] * (len(PROJECT_CLASSES) + 1),
            "class_weights": [1.0] * len(PROJECT_CLASSES),
            "distribution": {},
        }

    ds = FoveatedLidarDataset(bin_paths, label_paths, train=False)

    out_dir = os.path.join(output_dir, split_name)
    os.makedirs(out_dir, exist_ok=True)

    num_classes = len(PROJECT_CLASSES)
    class_counts = np.zeros(num_classes + 1, dtype=np.int64)  # + ignore bucket
    total_points = 0

    for i in range(len(ds)):
        pts, lbls = ds[i]
        pts_np = pts.numpy()
        lbls_np = lbls.numpy()

        np.save(os.path.join(out_dir, f"{i:06d}_pts.npy"), pts_np)
        np.save(os.path.join(out_dir, f"{i:06d}_lbl.npy"), lbls_np)

        total_points += pts_np.shape[0]

        for c in range(num_classes):
            class_counts[c] += (lbls_np == c).sum()
        class_counts[num_classes] += (lbls_np == 255).sum()

        if (i + 1) % 100 == 0 or i == len(ds) - 1:
            print(f"  [{split_name}] processed {i + 1}/{len(ds)} frames")

    print(f"\n[{split_name}] class point distribution:")
    names = list(PROJECT_CLASSES.values()) + ["ignored"]
    total = class_counts.sum()
    dist_dict = {}
    for name, count in zip(names, class_counts):
        pct = 100 * count / total if total else 0
        dist_dict[name] = {"count": int(count), "percentage": round(pct, 2)}
        print(f"  {name:22s}: {count:>10d}  ({pct:5.2f}%)")

    class_weights = compute_class_weights(class_counts)
    print(f"\n[{split_name}] calculated inverse-frequency class weights:")
    for c_id, c_name in PROJECT_CLASSES.items():
        print(f"  Class {c_id} ({c_name:20s}): {class_weights[c_id]:.4f}")

    return {
        "num_frames": len(ds),
        "total_points": int(total_points),
        "class_counts": class_counts.tolist(),
        "class_weights": class_weights.tolist(),
        "distribution": dist_dict,
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess LiDAR dataset.")
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-sequences", nargs="+", default=DEFAULT_TRAIN_SEQUENCES)
    parser.add_argument("--val-sequences", nargs="+", default=DEFAULT_VAL_SEQUENCES)
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames per split (for debugging).")

    args = parser.parse_args()

    print("=== Processing training split ===")
    train_info = process_split("train", args.train_sequences, dataset_root=args.dataset_root, output_dir=args.output_dir, max_frames=args.max_frames)

    print("\n=== Processing validation split ===")
    val_info = process_split("val", args.val_sequences, dataset_root=args.dataset_root, output_dir=args.output_dir, max_frames=args.max_frames)

    metadata = {
        "dataset_root": args.dataset_root,
        "train_sequences": args.train_sequences,
        "val_sequences": args.val_sequences,
        "project_classes": PROJECT_CLASSES,
        "train": train_info,
        "val": val_info,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    meta_path = os.path.join(args.output_dir, "dataset_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone. Preprocessed files & metadata saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
