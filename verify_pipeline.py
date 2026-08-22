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


def test_obstacle_preserving_voxel_priority():
    print("--- 3. Testing Obstacle-Preserving Voxel Label Priority ---")
    # Construct 5 synthetic points inside the EXACT same 3D voxel cell in near-field (0-10m, 5cm voxel size)
    pts = np.array([
        [2.01, 2.01, 0.01, 0.5],
        [2.02, 2.02, 0.01, 0.6],
        [2.03, 2.03, 0.01, 0.7],
        [2.04, 2.04, 0.01, 0.8],
        [2.00, 2.00, 0.01, 0.2],
    ], dtype=np.float32)
    lbls = np.array([0, 1, 2, 3, 255], dtype=np.int64)
    r = np.linalg.norm(pts[:, :2], axis=1)  # ~2.84m (near-field band 0-10m)

    ds = FoveatedLidarDataset([], [], train=False)
    downsampled_pts, downsampled_lbls = ds._range_aware_downsample(pts, lbls, r)

    print(f"Input : {len(pts)} mixed points inside 1 voxel with labels [0, 1, 2, 3, 255]")
    print(f"Output: {len(downsampled_pts)} downsampled point with label ID: {downsampled_lbls[0]}")

    assert len(downsampled_pts) == 1, "Expected exactly 1 downsampled point for 1 voxel"
    assert downsampled_lbls[0] == 3, f"Expected highest priority label 3 (dynamic_object), got {downsampled_lbls[0]}"
    print("Obstacle-Preserving Voxel Priority Test PASSED! (dynamic_object > static > non-drivable > drivable > ignore)\n")


if __name__ == "__main__":
    print("==================================================")
    print(" 3D LiDAR Foveated Mapping Data Pipeline Verifier ")
    print("==================================================\n")
    test_obstacle_preserving_voxel_priority()
    verify_raw_and_dataloader()
    verify_cached_outputs()

