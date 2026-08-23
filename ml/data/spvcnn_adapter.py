"""SPVCNN Input Adapter and Point-Voxel Index Mapping (Phase 12).

Converts raw and foveated 3D LiDAR point clouds into SPVCNN-compatible sparse
point-voxel representations while maintaining strict bidirectional index maps
between input points and voxelized features.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch


class SPVCNNInputAdapter:
    """Adapter converting (N, 4) point clouds to SPVCNN sparse point-voxel inputs."""

    def __init__(self, voxel_size: float = 0.05):
        """Initialize adapter.

        Args:
            voxel_size: Spatial quantization resolution in meters (default: 0.05m = 5cm).
        """
        self.voxel_size = float(voxel_size)

    def prepare_input(
        self,
        points: Union[np.ndarray, torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, Any]:
        """Convert input point cloud to SPVCNN input bundle.

        Args:
            points: Array or Tensor of shape (N, 4) with [x, y, z, intensity].
            device: Optional torch device.

        Returns:
            Dict containing:
                - 'points': Tensor of shape (N, 4) with [x, y, z, intensity]
                - 'xyz': Tensor of shape (N, 3)
                - 'features': Tensor of shape (N, 4)
                - 'voxel_coords': Tensor of shape (M, 3) quantized voxel coordinates
                - 'point_to_voxel_idx': Tensor of shape (N,) mapping each point to voxel index in [0, M-1]
                - 'voxel_to_point_idx': Tensor of shape (M,) mapping each voxel to representative point
                - 'num_points': int N
                - 'num_voxels': int M
        """
        if isinstance(points, torch.Tensor):
            pts_np = points.detach().cpu().numpy().astype(np.float32)
        else:
            pts_np = np.asarray(points, dtype=np.float32)

        if pts_np.ndim != 2 or pts_np.shape[1] < 3:
            raise ValueError(f"Expected points array of shape (N, 3) or (N, 4), got {pts_np.shape}")

        if pts_np.shape[1] == 3:
            # Add zero intensity if only XYZ is provided
            intensity = np.zeros((pts_np.shape[0], 1), dtype=np.float32)
            pts_np = np.hstack([pts_np, intensity])

        n_points = pts_np.shape[0]
        xyz = pts_np[:, :3]

        # 1. Quantize 3D coordinates into integer voxel grid
        v_coords = np.floor(xyz / self.voxel_size).astype(np.int64)

        # 2. Extract unique voxels and inverse point-to-voxel mapping (Accelerated 64-bit packed hash)
        v_min = np.min(v_coords, axis=0)
        v_shifted = v_coords - v_min
        v_max = np.max(v_shifted, axis=0) + 1

        max_idx = int(v_max[0]) * int(v_max[1]) * int(v_max[2])
        if max_idx < (1 << 62):
            keys = (
                v_shifted[:, 0]
                + v_shifted[:, 1] * v_max[0]
                + v_shifted[:, 2] * (v_max[0] * v_max[1])
            )
            _, voxel_to_pt, pt_to_voxel = np.unique(
                keys, return_index=True, return_inverse=True
            )
            unique_voxels = v_coords[voxel_to_pt]
        else:
            unique_voxels, voxel_to_pt, pt_to_voxel = np.unique(
                v_coords, axis=0, return_index=True, return_inverse=True
            )
        n_voxels = unique_voxels.shape[0]

        # 3. Build PyTorch tensors
        dev = device if device is not None else torch.device("cpu")
        points_tensor = torch.from_numpy(pts_np).float().to(dev)
        xyz_tensor = torch.from_numpy(xyz).float().to(dev)
        features_tensor = torch.from_numpy(pts_np).float().to(dev)
        voxel_coords_tensor = torch.from_numpy(unique_voxels).long().to(dev)
        pt_to_voxel_tensor = torch.from_numpy(pt_to_voxel).long().to(dev)
        voxel_to_pt_tensor = torch.from_numpy(voxel_to_pt).long().to(dev)

        return {
            "points": points_tensor,
            "xyz": xyz_tensor,
            "features": features_tensor,
            "voxel_coords": voxel_coords_tensor,
            "point_to_voxel_idx": pt_to_voxel_tensor,
            "voxel_to_point_idx": voxel_to_pt_tensor,
            "num_points": n_points,
            "num_voxels": n_voxels,
            "raw_xyz": xyz,
        }

    def project_voxel_predictions_to_points(
        self,
        voxel_predictions: Union[np.ndarray, torch.Tensor],
        point_to_voxel_idx: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Project voxel-level predictions or features back to per-point representations.

        Args:
            voxel_predictions: Array or Tensor of shape (M, ...) containing voxel-level predictions.
            point_to_voxel_idx: Index mapping of shape (N,) where each entry is in [0, M-1].

        Returns:
            Array or Tensor of shape (N, ...) with point-level predictions.
        """
        if isinstance(voxel_predictions, torch.Tensor):
            idx = point_to_voxel_idx if isinstance(point_to_voxel_idx, torch.Tensor) else torch.from_numpy(point_to_voxel_idx).to(voxel_predictions.device)
            return voxel_predictions[idx]
        else:
            return voxel_predictions[point_to_voxel_idx]
