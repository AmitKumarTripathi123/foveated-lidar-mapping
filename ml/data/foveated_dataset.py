"""Foveated Cached & Normalized PyTorch Dataset (Master Task).

Combines:
  1. Amit''s 3-zone Foveated Voxel Sampling
  2. Atul''s Phase 3 SIH 4-Class Label Remapping
  3. Configurable Point-Count Normalization (to fixed N=16,384 or N=1,024 for PointNet++)
  4. PyTorch Dataset & DataLoader integration
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from ml.data.dataset import load_point_cloud, load_labels, validate_point_label_alignment
from ml.data.preprocessing import filter_invalid_points, sample_points
from ml.data.amit_adapter import FoveatedVoxelSampler, FoveatedSamplingReport
from ml.data.label_mapping import SemanticLabelRemapper


def normalize_point_count(
    points: np.ndarray,
    labels: Optional[np.ndarray],
    target_num_points: int = 16384,
    strategy: str = "random",
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Normalize variable foveated point count to fixed target dimension.

    Args:
        points: Point cloud array of shape (N, 4).
        labels: Optional label array of shape (N,).
        target_num_points: Target number of points (default: 16384).
        strategy: Sampling strategy (''random'', ''deterministic'', ''pad'', ''repeat'').
        seed: Optional random seed.

    Returns:
        Tuple: (normalized_points [target_N, 4], normalized_labels [target_N])
    """
    n_pts = points.shape[0]
    if n_pts == target_num_points:
        return points, labels

    if n_pts > target_num_points:
        samp_pts, samp_lbls, _ = sample_points(
            points, labels, num_points=target_num_points, strategy=strategy, seed=seed
        )
        return samp_pts, samp_lbls

    # n_pts < target_num_points: Controlled repetition / padding
    rng = np.random.RandomState(seed)
    repeat_idx = rng.choice(n_pts, size=target_num_points, replace=True)
    norm_pts = points[repeat_idx]
    norm_lbls = labels[repeat_idx] if labels is not None else None

    if norm_lbls is not None:
        validate_point_label_alignment(norm_pts, norm_lbls)

    return norm_pts, norm_lbls


class FoveatedLidarDataset(TorchDataset):
    """PyTorch Dataset loading foveated and normalized LiDAR point clouds."""

    def __init__(
        self,
        cached_dir: Optional[Union[str, Path]] = None,
        raw_manifest: Optional[Union[List[Dict[str, Any]], List[Any]]] = None,
        target_num_points: int = 16384,
        foveated_sampler: Optional[FoveatedVoxelSampler] = None,
        label_remapper: Optional[SemanticLabelRemapper] = None,
        to_tensor: bool = True,
        seed: int = 42,
    ):
        """Initialize Foveated Dataset.

        Args:
            cached_dir: Optional directory containing cached .npy files.
            raw_manifest: Optional list of raw frame records or FrameRecord objects.
            target_num_points: Target normalized point count (default: 16384).
            foveated_sampler: Optional FoveatedVoxelSampler instance.
            label_remapper: Optional SemanticLabelRemapper instance.
            to_tensor: Whether to convert outputs to PyTorch tensors.
            seed: Random seed for deterministic point count normalization.
        """
        self.cached_dir = Path(cached_dir) if cached_dir else None
        self.raw_manifest = raw_manifest or []
        self.target_num_points = target_num_points
        self.foveated_sampler = foveated_sampler or FoveatedVoxelSampler()
        self.label_remapper = label_remapper or SemanticLabelRemapper()
        self.to_tensor = to_tensor
        self.seed = seed

        # Discover cached files if directory provided
        self.cached_samples: List[Tuple[Path, Path]] = []
        if self.cached_dir and self.cached_dir.is_dir():
            pts_files = sorted(self.cached_dir.glob("*_pts.npy"))
            for p_file in pts_files:
                l_file = self.cached_dir / p_file.name.replace("_pts.npy", "_lbl.npy")
                if l_file.is_file():
                    self.cached_samples.append((p_file, l_file))

    def __len__(self) -> int:
        if self.cached_samples:
            return len(self.cached_samples)
        return len(self.raw_manifest)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get preprocessed, foveated, and normalized point cloud sample."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        if self.cached_samples:
            pts_path, lbl_path = self.cached_samples[idx]
            points = np.load(pts_path)
            labels = np.load(lbl_path)
            metadata = {"source": "cache", "pts_path": str(pts_path), "lbl_path": str(lbl_path)}
        else:
            rec = self.raw_manifest[idx]
            if hasattr(rec, "point_cloud_path"):
                raw_pts = load_point_cloud(rec.point_cloud_path)
                raw_lbls = load_labels(rec.label_path) if (rec.label_path and rec.has_label) else None
                metadata = {"source": "raw", "sequence": rec.sequence_id, "frame": rec.frame_id}
            else:
                raw_pts = load_point_cloud(rec["point_path"])
                raw_lbls = load_labels(rec["label_path"]) if rec.get("label_path") else None
                metadata = {"source": "raw", "sequence": rec.get("sequence"), "frame": rec.get("frame")}

            # Stage 1: Invalid removal
            valid_pts, valid_lbls, _ = filter_invalid_points(raw_pts, raw_lbls)

            # Stage 2: Amit''s 3-Zone Foveated Voxel Sampling
            fov_pts, fov_lbls, _ = self.foveated_sampler.sample(valid_pts, valid_lbls)

            # Stage 3: Phase 3 SIH Label Remapping
            if fov_lbls is not None:
                labels = self.label_remapper.remap(fov_lbls)
            else:
                labels = None
            points = fov_pts

        # Stage 4: Point-Count Normalization to target_num_points
        norm_pts, norm_lbls = normalize_point_count(
            points, labels, target_num_points=self.target_num_points, seed=self.seed + idx
        )

        if self.to_tensor:
            pts_t = torch.from_numpy(norm_pts.copy()).float()
            lbl_t = torch.from_numpy(norm_lbls.astype(np.int64).copy()).long() if norm_lbls is not None else None
            return {"points": pts_t, "labels": lbl_t, "metadata": metadata}

        return {"points": norm_pts, "labels": norm_lbls, "metadata": metadata}
