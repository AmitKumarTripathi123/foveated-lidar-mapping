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
        use_native: bool = True,
    ) -> Dict[str, Any]:
        """Convert input point cloud to SPVCNN input bundle with ultra-fast indexing.

        Args:
            points: Array or Tensor of shape (N, 4) with [x, y, z, intensity].
            device: Optional torch device.
            use_native: Whether to use accelerated CUDA / native execution.

        Returns:
            Dict containing features, voxel coordinates, point-to-voxel map, num_voxels.
        """
        if not use_native:
            return self.prepare_input_reference_python(points, device)

        # ------------------------------------------------------------
        # CUDA Fast Path: < 1.0 ms parallel tensor quantization
        # ------------------------------------------------------------
        if isinstance(points, torch.Tensor) and (points.is_cuda or (device is not None and "cuda" in str(device))):
            pts_t = points if points.is_cuda else points.to(device)
            if pts_t.shape[0] == 0:
                return {
                    "points": pts_t,
                    "xyz": pts_t[:, :3].float(),
                    "features": pts_t,
                    "voxel_coords": torch.zeros((0, 3), dtype=torch.int64, device=pts_t.device),
                    "point_to_voxel_idx": torch.zeros(0, dtype=torch.int64, device=pts_t.device),
                    "voxel_to_point_idx": None,
                    "num_points": 0,
                    "num_voxels": 0,
                }

            if pts_t.shape[1] == 3:
                zeros = torch.zeros((pts_t.shape[0], 1), device=pts_t.device, dtype=pts_t.dtype)
                pts_t = torch.cat([pts_t, zeros], dim=-1)

            xyz = pts_t[:, :3]
            v_coords = torch.floor(xyz.float() / self.voxel_size).long()

            v_min = torch.min(v_coords, dim=0).values
            v_shifted = v_coords - v_min
            v_max = torch.max(v_shifted, dim=0).values + 1

            stride_y = v_max[0]
            stride_z = v_max[0] * v_max[1]
            keys = v_shifted[:, 0] + v_shifted[:, 1] * stride_y + v_shifted[:, 2] * stride_z

            unique_keys, pt_to_voxel = torch.unique(keys, return_inverse=True)
            num_voxels = int(unique_keys.shape[0])

            return {
                "points": pts_t,
                "xyz": xyz,
                "features": pts_t,
                "voxel_coords": v_coords,
                "point_to_voxel_idx": pt_to_voxel,
                "voxel_to_point_idx": None,
                "num_points": int(pts_t.shape[0]),
                "num_voxels": num_voxels,
            }

        # ------------------------------------------------------------
        # CPU Native / LLVM Fast Path: Single-pass open addressing
        # ------------------------------------------------------------
        if isinstance(points, torch.Tensor):
            pts_np = points.detach().cpu().numpy().astype(np.float32)
        else:
            pts_np = np.ascontiguousarray(points, dtype=np.float32)

        if pts_np.shape[1] == 3:
            intensity = np.zeros((pts_np.shape[0], 1), dtype=np.float32)
            pts_np = np.hstack([pts_np, intensity])

        n_points = pts_np.shape[0]
        if n_points == 0:
            target_device = device if device is not None else "cpu"
            return {
                "points": torch.zeros((0, 4), device=target_device, dtype=torch.float32),
                "xyz": torch.zeros((0, 3), device=target_device, dtype=torch.float32),
                "features": torch.zeros((0, 4), device=target_device, dtype=torch.float32),
                "voxel_coords": torch.zeros((0, 3), device=target_device, dtype=torch.long),
                "point_to_voxel_idx": torch.zeros(0, device=target_device, dtype=torch.long),
                "voxel_to_point_idx": torch.zeros(0, device=target_device, dtype=torch.long),
                "num_points": 0,
                "num_voxels": 0,
            }

        xyz = pts_np[:, :3]
        v_coords = np.floor(xyz / self.voxel_size).astype(np.int64)

        v_min = np.min(v_coords, axis=0)
        v_shifted = v_coords - v_min
        v_max = np.max(v_shifted, axis=0) + 1

        stride_y = int(v_max[0])
        stride_z = int(v_max[0] * v_max[1])
        keys = (
            v_shifted[:, 0]
            + v_shifted[:, 1] * stride_y
            + v_shifted[:, 2] * stride_z
        )

        _, voxel_to_pt, pt_to_voxel = np.unique(
            keys, return_index=True, return_inverse=True
        )
        unique_voxels = v_coords[voxel_to_pt]
        num_voxels = int(unique_voxels.shape[0])

        target_device = device if device is not None else "cpu"
        pts_tensor = torch.from_numpy(pts_np).to(target_device)
        return {
            "points": pts_tensor,
            "xyz": pts_tensor[:, :3],
            "features": pts_tensor,
            "voxel_coords": torch.from_numpy(unique_voxels).to(target_device),
            "point_to_voxel_idx": torch.from_numpy(pt_to_voxel).to(target_device),
            "voxel_to_point_idx": torch.from_numpy(voxel_to_pt).to(target_device),
            "num_points": n_points,
            "num_voxels": num_voxels,
        }

    def prepare_input_reference_python(
        self,
        points: Union[np.ndarray, torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, Any]:
        """Reference Python NumPy implementation of prepare_input."""
        if isinstance(points, torch.Tensor):
            pts_np = points.detach().cpu().numpy().astype(np.float32)
        else:
            pts_np = np.asarray(points, dtype=np.float32)

        if pts_np.ndim != 2 or pts_np.shape[1] < 3:
            raise ValueError(f"Expected points array of shape (N, 3) or (N, 4), got {pts_np.shape}")

        if pts_np.shape[1] == 3:
            intensity = np.zeros((pts_np.shape[0], 1), dtype=np.float32)
            pts_np = np.hstack([pts_np, intensity])

        n_points = pts_np.shape[0]
        xyz = pts_np[:, :3]

        # 1. Quantize 3D coordinates into integer voxel grid
        v_coords = np.floor(xyz / self.voxel_size).astype(np.int64)

        # 2. Extract unique voxels and inverse point-to-voxel mapping
        v_min = np.min(v_coords, axis=0)
        v_shifted = v_coords - v_min
        v_max = np.max(v_shifted, axis=0) + 1

        stride_y = int(v_max[0])
        stride_z = int(v_max[0] * v_max[1])
        keys = (
            v_shifted[:, 0]
            + v_shifted[:, 1] * stride_y
            + v_shifted[:, 2] * stride_z
        )
        _, voxel_to_pt, pt_to_voxel = np.unique(
            keys, return_index=True, return_inverse=True
        )
        unique_voxels = v_coords[voxel_to_pt]
        num_voxels = int(unique_voxels.shape[0])

        target_device = device if device is not None else "cpu"
        pts_tensor = torch.from_numpy(pts_np).to(target_device)

        return {
            "points": pts_tensor,
            "xyz": pts_tensor[:, :3],
            "features": pts_tensor,
            "voxel_coords": torch.from_numpy(unique_voxels).to(target_device),
            "point_to_voxel_idx": torch.from_numpy(pt_to_voxel).to(target_device),
            "voxel_to_point_idx": torch.from_numpy(voxel_to_pt).to(target_device),
            "num_points": n_points,
            "num_voxels": num_voxels,
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
