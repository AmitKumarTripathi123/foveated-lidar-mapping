"""Training-Only 3D Point Cloud Data Augmentation Engine (Phase 5).

Applies rigid geometric transformations exclusively to spatial coordinates (X, Y, Z):
  - Random yaw rotation around Z-axis
  - Random anisotropic scaling
  - Gaussian coordinate jitter

Guarantees:
  - Labels are NEVER modified
  - Intensity channel is preserved
  - Validation/test sets are NEVER augmented
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch


class LidarAugmentor:
    """Rigid 3D geometric augmentation pipeline for training point clouds."""

    def __init__(
        self,
        enabled: bool = True,
        rotation_range: Tuple[float, float] = (-15.0, 15.0),
        scale_range: Tuple[float, float] = (0.95, 1.05),
        jitter_std: float = 0.01,
        seed: Optional[int] = None,
    ):
        """Initialize augmentor.

        Args:
            enabled: Whether augmentation is active.
            rotation_range: (min_deg, max_deg) for yaw rotation around Z-axis.
            scale_range: (min_scale, max_scale) for point cloud scaling.
            jitter_std: Standard deviation of additive Gaussian noise on coordinates.
            seed: Optional random seed for reproducible augmentation testing.
        """
        self.enabled = enabled
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.jitter_std = jitter_std
        self.rng = np.random.RandomState(seed)

    def __call__(
        self, points: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Apply geometric augmentations to input point cloud.

        Args:
            points: Input array or tensor of shape (N, 4) with [x, y, z, intensity].

        Returns:
            Augmented point cloud of shape (N, 4).
        """
        if not self.enabled:
            return points

        is_torch = isinstance(points, torch.Tensor)
        if is_torch:
            pts_np = points.detach().cpu().numpy().copy()
        else:
            pts_np = points.copy()

        xyz = pts_np[:, :3]
        intensity = pts_np[:, 3:]

        # 1. Random Yaw Rotation around Z-axis (upward)
        if self.rotation_range[0] != 0.0 or self.rotation_range[1] != 0.0:
            angle_deg = self.rng.uniform(self.rotation_range[0], self.rotation_range[1])
            angle_rad = np.radians(angle_deg)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            rot_matrix = np.array([
                [cos_a, -sin_a, 0.0],
                [sin_a,  cos_a, 0.0],
                [0.0,    0.0,   1.0]
            ], dtype=np.float32)
            xyz = xyz @ rot_matrix.T

        # 2. Random Global Scaling
        if self.scale_range[0] != 1.0 or self.scale_range[1] != 1.0:
            scale = self.rng.uniform(self.scale_range[0], self.scale_range[1])
            xyz = xyz * scale

        # 3. Additive Gaussian Jitter
        if self.jitter_std > 0.0:
            jitter = self.rng.normal(0.0, self.jitter_std, size=xyz.shape).astype(np.float32)
            xyz = xyz + jitter

        aug_points = np.concatenate([xyz.astype(np.float32), intensity], axis=1)

        if is_torch:
            return torch.from_numpy(aug_points).to(points.device).type_as(points)
        return aug_points
