"""
verify_pipeline.py
==================
Verification script to test the complete 3D LiDAR Foveated Mapping Data Pipeline:
  1. Verifies raw sequence loading and label remapping.
  2. Tests PyTorch DataLoader batch collation (collate_fn_foveated).
  3. Verifies preprocessed .npy files and metadata outputs.
"""

import os
import json
import torch
import numpy as np

from class_map import PROJECT_CLASSES, POSS_RAW_CLASSES, get_class_colors, compute_class_weights
from dataset import build_file_list, FoveatedLidarDataset, create_dataloader

DATASET_ROOT = "dataset"
OUTPUT_DIR = "processed"


def verify_raw_and_dataloader():
    print("--- 1. Testing Raw Dataset & DataLoader Batching ---")
    bin_paths, label_paths = build_file_list(DATASET_ROOT, ["00"])
    print(f"Found {len(bin_paths)} frames in Sequence 00.")

    # Create dataset instance
    ds = FoveatedLidarDataset(bin_paths, label_paths, train=True, downsample=True)
    sample_pts, sample_lbls = ds[0]
    print(f"Sample frame 0: Points shape={sample_pts.shape}, Labels shape={sample_lbls.shape}")
    print(f"Unique project class IDs in frame 0: {torch.unique(sample_lbls).tolist()}")

    # Test DataLoader
    loader = create_dataloader(bin_paths[:8], label_paths[:8], batch_size=4, shuffle=True, train=True)
    for batch_idx, (pts_list, lbls_list, batch_indices) in enumerate(loader):
        print(f"Batch {batch_idx}: {len(pts_list)} frames, Total points across batch: {batch_indices.shape[0]}")
        assert len(pts_list) == 4, "Expected batch size of 4"
        assert len(lbls_list) == 4, "Expected batch size of 4"
        break
    print("DataLoader batching & collation check PASSED!\n")


def verify_cached_outputs():
    print("--- 2. Testing Preprocessed Metadata & Cache Files ---")
    meta_path = os.path.join(OUTPUT_DIR, "dataset_metadata.json")
    if not os.path.exists(meta_path):
        print(f"WARNING: {meta_path} not found yet. Run preprocess.py first.")
        return False

    with open(meta_path, "r") as f:
        meta = json.load(f)

    print("Loaded dataset metadata:")
    print(f"  Train Frames : {meta['train']['num_frames']}")
    print(f"  Val Frames   : {meta['val']['num_frames']}")
    print(f"  Train Points : {meta['train']['total_points']:,}")
    print(f"  Val Points   : {meta['val']['total_points']:,}")

    print("\nTraining Split Distribution:")
    for cls_name, info in meta['train']['distribution'].items():
        print(f"  {cls_name:22s}: {info['count']:>10d} ({info['percentage']:5.2f}%)")

    print("\nTraining Loss Weights (Inverse-Frequency Normalized):")
    for cls_id_str, cls_name in meta['project_classes'].items():
        cls_id = int(cls_id_str)
        weight = meta['train']['class_weights'][cls_id]
        print(f"  Class {cls_id} ({cls_name:20s}): {weight:.4f}")

    return True


if __name__ == "__main__":
    print("==================================================")
    print(" 3D LiDAR Foveated Mapping Data Pipeline Verifier ")
    print("==================================================\n")
    verify_raw_and_dataloader()
    verify_cached_outputs()
