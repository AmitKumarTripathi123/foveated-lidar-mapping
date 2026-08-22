"""
dataset.py
==========
FoveatedLidarDataset: a PyTorch Dataset for SemanticPOSS that

  1. Loads raw point clouds (.bin) + labels (.label)
  2. Remaps labels to project super-classes (see class_map.py)
  3. Range-filters to the foveated grid's outer radius
  4. Applies distance-aware voxel downsampling -- fine resolution near
     the sensor, coarse resolution far away -- mirroring the eventual
     variable-resolution 2.5D grid
  5. Optionally augments (train split only)

Expects the folder layout you've already verified:
    dataset/
      sequences/
        00/velodyne/*.bin, 00/labels/*.label
        01/...
        ...
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import open3d as o3d
    _HAS_O3D = True
except ImportError:
    _HAS_O3D = False

from class_map import remap_labels

# Range bands (meters) -> voxel size (meters).
# Mirrors the foveated grid design: 5cm within 10m, 15cm out to 40m,
# 50cm out to 100m. Adjust to match your final grid engine's resolution.
RANGE_BANDS = [
    (0.0, 10.0, 0.05),
    (10.0, 40.0, 0.15),
    (40.0, 100.0, 0.50),
]


def build_file_list(root, sequences, max_frames=None):
    """Collect matched (bin, label) file paths for the given sequence IDs using stem matching."""
    from pathlib import Path
    bin_paths, label_paths = [], []
    for seq in sequences:
        seq_dir = os.path.join(root, "sequences", str(seq))
        if not os.path.exists(seq_dir):
            seq_dir = os.path.join(root, str(seq))
        if not os.path.exists(seq_dir):
            continue

        velo_dir = os.path.join(seq_dir, "velodyne")
        if not os.path.exists(velo_dir):
            velo_dir = os.path.join(seq_dir, "scans")
        if not os.path.exists(velo_dir):
            velo_dir = seq_dir

        lbl_dir = os.path.join(seq_dir, "labels")
        if not os.path.exists(lbl_dir):
            lbl_dir = seq_dir

        bin_map = {Path(p).stem: p for p in glob.glob(os.path.join(velo_dir, "*.bin"))}
        lbl_map = {Path(p).stem: p for p in glob.glob(os.path.join(lbl_dir, "*.label"))}
        common_stems = sorted(set(bin_map.keys()) & set(lbl_map.keys()))

        for stem in common_stems:
            bin_paths.append(bin_map[stem])
            label_paths.append(lbl_map[stem])

    if max_frames is not None and max_frames > 0:
        bin_paths = bin_paths[:max_frames]
        label_paths = label_paths[:max_frames]

    return bin_paths, label_paths


class FoveatedLidarDataset(Dataset):
    def __init__(self, bin_paths, label_paths, max_range=100.0,
                 train=True, downsample=True):
        assert len(bin_paths) == len(label_paths), \
            "bin_paths and label_paths must be the same length"
        self.bin_paths = bin_paths
        self.label_paths = label_paths
        self.max_range = max_range
        self.train = train
        self.downsample = downsample

    def __len__(self):
        return len(self.bin_paths)

    def __getitem__(self, idx):
        points = np.fromfile(self.bin_paths[idx], dtype=np.float32).reshape(-1, 4)
        labels_raw = np.fromfile(self.label_paths[idx], dtype=np.uint32) & 0xFFFF
        labels = remap_labels(labels_raw)

        # range filter -- drop anything beyond our grid's outer radius
        r = np.linalg.norm(points[:, :2], axis=1)
        mask = r < self.max_range
        points, labels, r = points[mask], labels[mask], r[mask]

        if self.downsample:
            points, labels = self._range_aware_downsample(points, labels, r)

        if self.train:
            points, labels = self._augment(points, labels)

        return (
            torch.from_numpy(points).float(),
            torch.from_numpy(labels).long(),
        )

    def _range_aware_downsample(self, points, labels, r):
        """
        Fast distance-aware multi-band voxel downsampling with
        Obstacle-Preserving Voxel Aggregation.
        
        Priority hierarchy within each voxel:
          1. dynamic_object (3) -> highest priority (pedestrians, cars, riders)
          2. static_obstacle (2) -> second priority (poles, fences, signs, curbs)
          3. non_drivable_terrain (1) -> third priority
          4. drivable_terrain (0) -> fourth priority
          5. IGNORE_LABEL (255) -> lowest priority
        """
        # Priority mapping table indexed by label ID (0..255)
        priority_map = np.zeros(256, dtype=np.int32)
        priority_map[3] = 4    # dynamic_object
        priority_map[2] = 3    # static_obstacle
        priority_map[1] = 2    # non_drivable_terrain
        priority_map[0] = 1    # drivable_terrain
        priority_map[255] = 0  # IGNORE_LABEL

        out_pts, out_lbl = [], []
        for lo, hi, voxel_size in RANGE_BANDS:
            band_mask = (r >= lo) & (r < hi)
            if not np.any(band_mask):
                continue
            band_points = points[band_mask]
            band_labels = labels[band_mask]

            # Priority-based sorting so obstacle points take precedence inside each voxel
            safe_labels = np.clip(band_labels, 0, 255)
            point_priorities = priority_map[safe_labels]
            sort_idx = np.argsort(-point_priorities)

            sorted_points = band_points[sort_idx]
            sorted_labels = band_labels[sort_idx]

            # Quantize 3D coordinates into integer voxel indices
            voxel_coords = np.floor(sorted_points[:, :3] / voxel_size).astype(np.int64)
            _, keep_idx = np.unique(voxel_coords, axis=0, return_index=True)

            out_pts.append(sorted_points[keep_idx])
            out_lbl.append(sorted_labels[keep_idx])

        if not out_pts:
            return points, labels
        return np.concatenate(out_pts, axis=0), np.concatenate(out_lbl, axis=0)


    def _augment(self, points, labels):
        # Trigonometric random 2D yaw rotation around Z-axis
        theta = np.random.uniform(0, 2 * np.pi)
        cos_t, sin_t = np.cos(theta, dtype=np.float32), np.sin(theta, dtype=np.float32)
        x_orig = points[:, 0].copy()
        y_orig = points[:, 1].copy()
        points[:, 0] = x_orig * cos_t - y_orig * sin_t
        points[:, 1] = x_orig * sin_t + y_orig * cos_t

        # Small Gaussian spatial jitter on xyz
        points[:, :3] += np.random.normal(0, 0.01, (points.shape[0], 3)).astype(np.float32)

        # Random point dropout (simulates occlusion) -- drop up to 10%
        if np.random.rand() < 0.5:
            drop_ratio = np.random.uniform(0.0, 0.1)
            n = points.shape[0]
            keep = np.random.rand(n) > drop_ratio
            points, labels = points[keep], labels[keep]

        return points, labels





def collate_fn_foveated(batch):
    """
    Custom PyTorch collate function for 3D point cloud frames.
    Returns:
        points_list: list of [N_i, 4] Tensors (x, y, z, intensity)
        labels_list: list of [N_i] Tensors (class IDs)
        batch_indices: [sum(N_i)] Tensor assigning each point to its batch index
    """
    points_list, labels_list = zip(*batch)
    batch_indices = []
    for batch_id, pts in enumerate(points_list):
        batch_indices.append(torch.full((pts.shape[0],), batch_id, dtype=torch.long))
    
    batch_indices_tensor = torch.cat(batch_indices, dim=0)
    return points_list, labels_list, batch_indices_tensor


def create_dataloader(bin_paths, label_paths, batch_size=4, shuffle=True,
                      num_workers=0, train=True, downsample=True):
    """Factory function to build a PyTorch DataLoader for FoveatedLidarDataset."""
    dataset = FoveatedLidarDataset(
        bin_paths=bin_paths,
        label_paths=label_paths,
        train=train,
        downsample=downsample
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_foveated
    )

