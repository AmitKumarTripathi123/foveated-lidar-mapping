"""
Phase 13: 3D LiDAR Point Cloud Augmentation for Autonomous Navigation.
Applies geometrically physically valid spatial transformations strictly to training data.
Validation data is never augmented.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch


class LidarAugmentor:
    """Configurable 3D point cloud augmentor for autonomous driving LiDAR data."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        is_training: bool = True,
    ):
        self.is_training = is_training
        cfg = config or {}
        aug_cfg = cfg.get("augmentation", {})

        self.enabled = bool(aug_cfg.get("enabled", False)) and self.is_training
        self.rotation_deg = float(aug_cfg.get("rotation_deg", 10.0))
        self.scale_range = (
            float(aug_cfg.get("min_scale", 0.95)),
            float(aug_cfg.get("max_scale", 1.05)),
        )
        self.jitter_std = float(aug_cfg.get("jitter_std", 0.01))
        self.jitter_clip = float(aug_cfg.get("jitter_clip", 0.03))
        self.flip_x_prob = float(aug_cfg.get("flip_x_prob", 0.5))
        self.flip_y_prob = float(aug_cfg.get("flip_y_prob", 0.5))
        self.translation_max = float(aug_cfg.get("translation_max", 0.1))

    def augment(
        self,
        points: Union[np.ndarray, torch.Tensor],
        labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Optional[Union[np.ndarray, torch.Tensor]]]:
        """Apply spatial 3D augmentations to point coordinates while preserving labels.

        Args:
            points: (N, 3) or (N, 4) point cloud [x, y, z, (intensity)].
            labels: Optional (N,) semantic labels.

        Returns:
            Tuple of (augmented_points, labels).
        """
        if not self.enabled:
            return points, labels

        is_torch = isinstance(points, torch.Tensor)
        if is_torch:
            pts_np = points.detach().cpu().numpy().copy()
        else:
            pts_np = np.asarray(points, dtype=np.float32).copy()

        xyz = pts_np[:, :3]
        intensity = pts_np[:, 3:] if pts_np.shape[1] > 3 else None

        # 1. Random Rotation around Z axis (yaw)
        if self.rotation_deg > 0:
            angle_rad = np.random.uniform(-self.rotation_deg, self.rotation_deg) * (np.pi / 180.0)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            rot_matrix = np.array([
                [cos_a, -sin_a, 0.0],
                [sin_a,  cos_a, 0.0],
                [0.0,    0.0,   1.0],
            ], dtype=np.float32)
            xyz = xyz @ rot_matrix.T

        # 2. Random Scaling
        if self.scale_range[0] < self.scale_range[1]:
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            xyz = xyz * np.float32(scale)

        # 3. Random Gaussian Coordinate Jitter
        if self.jitter_std > 0:
            noise = np.clip(
                np.random.normal(0.0, self.jitter_std, size=xyz.shape).astype(np.float32),
                -self.jitter_clip,
                self.jitter_clip,
            )
            xyz = xyz + noise

        # 4. Random Translation
        if self.translation_max > 0:
            shift = np.random.uniform(-self.translation_max, self.translation_max, size=(1, 3)).astype(np.float32)
            xyz = xyz + shift

        # 5. Random Flip X / Y
        if np.random.rand() < self.flip_x_prob:
            xyz[:, 0] = -xyz[:, 0]
        if np.random.rand() < self.flip_y_prob:
            xyz[:, 1] = -xyz[:, 1]

        # Reconstruct output points array
        if intensity is not None:
            aug_pts = np.hstack([xyz, intensity]).astype(np.float32)
        else:
            aug_pts = xyz.astype(np.float32)

        if is_torch:
            return torch.from_numpy(aug_pts).to(points.device), labels
        else:
            return aug_pts, labels
