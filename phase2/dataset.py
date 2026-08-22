"""
Phase 2 Dataset Adapter for 40-beam SemanticPOSS LiDAR point clouds.
Enforces single authoritative label mapping into project super-classes:
0: drivable_terrain, 1: non_drivable_terrain, 2: static_obstacle, 3: dynamic_object, 255: IGNORE_LABEL.
"""

import os
import glob
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from src.types import SuperClass, AggregationPolicy, FoveationBand
from src.range_filter import RangeFilter
from src.foveation import FoveatedVoxelizer

# Single Authoritative SemanticPOSS Raw -> Project Super-Class Mapping
SEMANTICPOSS_TO_PROJECT: Dict[int, int] = {
    0: SuperClass.IGNORE_LABEL,          # unlabeled -> 255
    1: SuperClass.IGNORE_LABEL,          # outlier -> 255
    4: SuperClass.DYNAMIC_OBJECT,        # person -> 3
    5: SuperClass.DYNAMIC_OBJECT,        # two-wheelers -> 3
    6: SuperClass.DYNAMIC_OBJECT,        # rider -> 3
    7: SuperClass.DYNAMIC_OBJECT,        # car -> 3
    8: SuperClass.DYNAMIC_OBJECT,        # other-vehicle / truck -> 3
    9: SuperClass.STATIC_OBSTACLE,       # building -> 2
    10: SuperClass.STATIC_OBSTACLE,      # fence -> 2
    11: SuperClass.STATIC_OBSTACLE,      # other-structure -> 2
    13: SuperClass.STATIC_OBSTACLE,      # pole -> 2
    14: SuperClass.STATIC_OBSTACLE,      # traffic-sign -> 2
    15: SuperClass.STATIC_OBSTACLE,      # cone -> 2
    16: SuperClass.STATIC_OBSTACLE,      # trashcan -> 2
    17: SuperClass.STATIC_OBSTACLE,      # vegetation -> 2
    18: SuperClass.STATIC_OBSTACLE,      # trunk -> 2
    19: SuperClass.NON_DRIVABLE_TERRAIN, # terrain (grass/lawn/dirt) -> 1
    20: SuperClass.NON_DRIVABLE_TERRAIN, # other-ground (sidewalk/curb) -> 1
    21: SuperClass.DRIVABLE_TERRAIN,     # ground/road (paved road) -> 0
    22: SuperClass.IGNORE_LABEL          # outlier -> 255
}


def remap_poss_labels(raw_labels: np.ndarray) -> np.ndarray:
    """Vectorized remapping of raw 16-bit SemanticPOSS labels to 4 super-classes + ignore."""
    raw_16 = raw_labels.astype(np.uint32) & 0xFFFF
    remapped = np.full(raw_16.shape, SuperClass.IGNORE_LABEL, dtype=np.int64)
    for raw_id, super_cls in SEMANTICPOSS_TO_PROJECT.items():
        remapped[raw_16 == raw_id] = super_cls
    return remapped


class Phase2Dataset(Dataset):
    """
    PyTorch Dataset for Phase 2 Semantic Segmentation on 40-beam SemanticPOSS data.
    Supports sequence-based train/val/test splits, range filtering, distance-aware foveation, and augmentation.
    """
    def __init__(
        self,
        dataset_root: Union[str, Path] = "data/semanticposs_sequence",
        sequences: Optional[List[str]] = None,
        split: str = "train",
        max_range: float = 100.0,
        downsample: bool = True,
        aggregation_policy: AggregationPolicy = AggregationPolicy.OBSTACLE_PRESERVING,
        max_points_per_frame: Optional[int] = 40000,
        foveation_config_path: str = "configs/foveation_default.yaml"
    ):
        self.dataset_root = Path(dataset_root)
        self.split = split.lower()
        self.max_range = float(max_range)
        self.downsample = downsample
        self.aggregation_policy = aggregation_policy
        self.max_points_per_frame = max_points_per_frame

        # Sequence discovery based on split
        if sequences is None:
            if self.split == "train":
                self.sequences = ["00", "01", "03", "04", "05"]
            elif self.split == "val":
                self.sequences = ["02"]
            else:
                self.sequences = ["01"]
        else:
            self.sequences = [str(s).zfill(2) for s in sequences]

        self.bin_paths, self.label_paths = self._discover_files()

        self.range_filter = RangeFilter(min_range=0.0, max_range=self.max_range)
        self.voxelizer = FoveatedVoxelizer(config_path=foveation_config_path, max_range=self.max_range)

    def _discover_files(self) -> Tuple[List[Path], List[Path]]:
        bins, labels = [], []
        for seq in self.sequences:
            seq_dir = self.dataset_root / "sequences" / seq
            if not seq_dir.exists():
                seq_dir = self.dataset_root / seq
            if not seq_dir.exists():
                continue

            v_dir = seq_dir / "velodyne"
            l_dir = seq_dir / "labels"
            if not v_dir.exists():
                continue

            b_files = sorted(v_dir.glob("*.bin"))
            for b in b_files:
                stem = b.stem
                l_file = l_dir / f"{stem}.label"
                if l_file.exists():
                    bins.append(b)
                    labels.append(l_file)
                else:
                    l_cands = list(l_dir.glob(f"{stem}.*"))
                    if l_cands:
                        bins.append(b)
                        labels.append(l_cands[0])

        if len(bins) == 0:
            all_bins = sorted(self.dataset_root.glob("**/*.bin"))
            for b in all_bins:
                l_cand = b.parent.parent / "labels" / f"{b.stem}.label"
                if l_cand.exists():
                    bins.append(b)
                    labels.append(l_cand)

        return bins, labels

    def __len__(self) -> int:
        return len(self.bin_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        bin_path = self.bin_paths[idx]
        lbl_path = self.label_paths[idx]

        raw_pts = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)
        raw_lbls = np.fromfile(str(lbl_path), dtype=np.uint32)

        n_pts = min(len(raw_pts), len(raw_lbls))
        raw_pts = raw_pts[:n_pts]
        raw_lbls = raw_lbls[:n_pts]

        mapped_lbls = remap_poss_labels(raw_lbls)

        r = np.sqrt(raw_pts[:, 0]**2 + raw_pts[:, 1]**2)
        valid_mask = (r >= 0.0) & (r <= self.max_range) & np.isfinite(raw_pts[:, :3]).all(axis=1)
        pts_filtered = raw_pts[valid_mask]
        lbls_filtered = mapped_lbls[valid_mask]

        if self.downsample and len(pts_filtered) > 0:
            from src.types import PointCloudFrame
            frame = PointCloudFrame(points=pts_filtered, labels=lbls_filtered.astype(np.uint32))
            fov_res = self.voxelizer.voxelize(frame, policy=self.aggregation_policy)
            pts_out = fov_res.foveated_frame.points
            lbls_out = fov_res.foveated_frame.labels.astype(np.int64)
        else:
            pts_out = pts_filtered
            lbls_out = lbls_filtered

        if self.split == "train" and len(pts_out) > 0:
            pts_out = self._augment(pts_out)

        if self.max_points_per_frame is not None and len(pts_out) > self.max_points_per_frame:
            choice = np.random.choice(len(pts_out), self.max_points_per_frame, replace=False)
            pts_out = pts_out[choice]
            lbls_out = lbls_out[choice]

        return {
            "points": torch.from_numpy(pts_out.astype(np.float32)),
            "labels": torch.from_numpy(lbls_out.astype(np.int64)),
            "frame_id": bin_path.stem,
            "sequence_id": bin_path.parent.parent.name,
            "raw_point_count": len(raw_pts),
            "foveated_point_count": len(pts_out)
        }

    def _augment(self, points: np.ndarray) -> np.ndarray:
        pts_aug = points.copy()
        angle = float(np.random.uniform(-np.pi, np.pi))
        cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
        rot_matrix = np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        xyz = np.nan_to_num(pts_aug[:, :3], nan=0.0, posinf=50.0, neginf=-50.0).astype(np.float32)
        pts_aug[:, :3] = np.dot(xyz, rot_matrix).astype(np.float32)

        scale = float(np.random.uniform(0.95, 1.05))
        pts_aug[:, :3] *= scale

        jitter = np.random.normal(0, 0.005, size=pts_aug[:, :3].shape).astype(np.float32)
        pts_aug[:, :3] += jitter

        return pts_aug.astype(np.float32)
