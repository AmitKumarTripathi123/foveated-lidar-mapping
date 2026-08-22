"""
preprocess.py
=============
Run the pipeline once over every frame in the chosen train/val sequences,
cache the processed (range-filtered, downsampled, remapped) output as
.npy files, and print the per-class point distribution so you know up
front whether you need class-weighted loss.

Usage:
    python3 preprocess.py

Output:
    processed/train/000000_pts.npy, 000000_lbl.npy, ...
    processed/val/000000_pts.npy,   000000_lbl.npy, ...
"""

import os
import json
import numpy as np

from dataset import FoveatedLidarDataset, build_file_list
from class_map import PROJECT_CLASSES, compute_class_weights

DATASET_ROOT = "dataset"
OUTPUT_DIR = "processed"

TRAIN_SEQUENCES = ["00", "01", "03", "04", "05"]
VAL_SEQUENCES = ["02"]


def process_split(split_name, sequences):
    bin_paths, label_paths = build_file_list(DATASET_ROOT, sequences)
    print(f"[{split_name}] {len(bin_paths)} frames from sequences {sequences}")

    ds = FoveatedLidarDataset(bin_paths, label_paths, train=False)

    out_dir = os.path.join(OUTPUT_DIR, split_name)
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


if __name__ == "__main__":
    print("=== Processing training split ===")
    train_info = process_split("train", TRAIN_SEQUENCES)

    print("\n=== Processing validation split ===")
    val_info = process_split("val", VAL_SEQUENCES)

    metadata = {
        "dataset_root": DATASET_ROOT,
        "train_sequences": TRAIN_SEQUENCES,
        "val_sequences": VAL_SEQUENCES,
        "project_classes": PROJECT_CLASSES,
        "train": train_info,
        "val": val_info,
    }

    meta_path = os.path.join(OUTPUT_DIR, "dataset_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone. Preprocessed files & metadata saved to: {os.path.abspath(OUTPUT_DIR)}")

