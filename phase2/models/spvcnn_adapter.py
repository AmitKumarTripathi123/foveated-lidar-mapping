"""
SPVCNN Input and Label Adapters for Phase 2 Perception Pipeline.
Ensures seamless conversion from PointCloudFrame to Point-Voxel tensors
and maps native class predictions to the 4 SIH navigation super-classes:
  0: drivable_terrain
  1: non_drivable_terrain
  2: static_obstacle
  3: dynamic_object
  255: IGNORE_LABEL
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

from src.types import SuperClass


# Authoritative SemanticKITTI 19-class to SIH 4-class ontology mapping
SEMANTICKITTI_TO_SIH: Dict[int, int] = {
    0: SuperClass.DYNAMIC_OBJECT,        # car -> dynamic_object (3)
    1: SuperClass.DYNAMIC_OBJECT,        # bicycle -> dynamic_object (3)
    2: SuperClass.DYNAMIC_OBJECT,        # motorcycle -> dynamic_object (3)
    3: SuperClass.DYNAMIC_OBJECT,        # truck -> dynamic_object (3)
    4: SuperClass.DYNAMIC_OBJECT,        # other-vehicle -> dynamic_object (3)
    5: SuperClass.DYNAMIC_OBJECT,        # person -> dynamic_object (3)
    6: SuperClass.DYNAMIC_OBJECT,        # bicyclist -> dynamic_object (3)
    7: SuperClass.DYNAMIC_OBJECT,        # motorcyclist -> dynamic_object (3)
    8: SuperClass.DRIVABLE_TERRAIN,      # road -> drivable_terrain (0)
    9: SuperClass.DRIVABLE_TERRAIN,      # parking -> drivable_terrain (0)
    10: SuperClass.NON_DRIVABLE_TERRAIN, # sidewalk -> non_drivable_terrain (1)
    11: SuperClass.NON_DRIVABLE_TERRAIN, # other-ground -> non_drivable_terrain (1)
    12: SuperClass.STATIC_OBSTACLE,      # building -> static_obstacle (2)
    13: SuperClass.STATIC_OBSTACLE,      # fence -> static_obstacle (2)
    14: SuperClass.STATIC_OBSTACLE,      # vegetation -> static_obstacle (2)
    15: SuperClass.STATIC_OBSTACLE,      # trunk -> static_obstacle (2)
    16: SuperClass.NON_DRIVABLE_TERRAIN, # terrain -> non_drivable_terrain (1)
    17: SuperClass.STATIC_OBSTACLE,      # pole -> static_obstacle (2)
    18: SuperClass.STATIC_OBSTACLE,      # traffic-sign -> static_obstacle (2)
}


class SPVCNNInputAdapter:
    """Adapter converting (N, 4) point clouds to SPVCNN sparse point-voxel inputs."""

    def __init__(self, voxel_size: float = 0.05):
        self.voxel_size = float(voxel_size)

    def prepare_input(
        self,
        points: Union[np.ndarray, torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, Any]:
        """Convert input point cloud to SPVCNN input bundle with strict 1:1 point tracking."""
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

        if n_points == 0:
            dev = device if device is not None else torch.device("cpu")
            return {
                "points": torch.empty((0, 4), dtype=torch.float32, device=dev),
                "xyz": torch.empty((0, 3), dtype=torch.float32, device=dev),
                "features": torch.empty((0, 4), dtype=torch.float32, device=dev),
                "voxel_coords": torch.empty((0, 3), dtype=torch.long, device=dev),
                "point_to_voxel_idx": torch.empty((0,), dtype=torch.long, device=dev),
                "voxel_to_point_idx": torch.empty((0,), dtype=torch.long, device=dev),
                "num_points": 0,
                "num_voxels": 0,
                "raw_xyz": xyz,
            }

        # 1. Quantize 3D coordinates into integer voxel grid with 64-bit integer packing
        OFFSET = 50000
        vx = np.floor(xyz[:, 0] / self.voxel_size).astype(np.int64) + OFFSET
        vy = np.floor(xyz[:, 1] / self.voxel_size).astype(np.int64) + OFFSET
        vz = np.floor(xyz[:, 2] / self.voxel_size).astype(np.int64) + OFFSET
        packed = (vx << 42) | (vy << 21) | vz

        # 2. Extract unique voxels and inverse point-to-voxel mapping
        unique_packed, voxel_to_pt, pt_to_voxel = np.unique(
            packed, return_index=True, return_inverse=True
        )
        n_voxels = unique_packed.shape[0]

        u_vx = (unique_packed >> 42) - OFFSET
        u_vy = ((unique_packed >> 21) & 0x1FFFFF) - OFFSET
        u_vz = (unique_packed & 0x1FFFFF) - OFFSET
        unique_voxels = np.column_stack([u_vx, u_vy, u_vz])

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


class SPVCNNLabelAdapter:
    """Adapts SPVCNN native class logits and predictions to the frozen SIH 4-class ontology."""

    def __init__(
        self,
        native_source: str = "semantickitti",
        custom_mapping: Optional[Dict[int, int]] = None,
    ):
        self.native_source = native_source.lower()
        if custom_mapping is not None:
            self.mapping = custom_mapping
        elif self.native_source == "semantickitti":
            self.mapping = SEMANTICKITTI_TO_SIH
        elif self.native_source == "sih_direct":
            self.mapping = {0: 0, 1: 1, 2: 2, 3: 3}
        else:
            self.mapping = SEMANTICKITTI_TO_SIH

        # Fast lookup table
        max_class = max(self.mapping.keys()) if self.mapping else 255
        self.lut = np.full(max(max_class + 1, 256), SuperClass.IGNORE_LABEL, dtype=np.int64)
        for k, v in self.mapping.items():
            if 0 <= k < len(self.lut):
                self.lut[k] = v

    def remap_predictions(self, native_classes: np.ndarray) -> np.ndarray:
        raw = np.asarray(native_classes, dtype=np.int64)
        valid_mask = (raw >= 0) & (raw < len(self.lut))
        sih_classes = np.full_like(raw, SuperClass.IGNORE_LABEL)
        sih_classes[valid_mask] = self.lut[raw[valid_mask]]
        return sih_classes

    def process_logits(
        self,
        logits: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert raw network logits to:
          - sih_predicted_classes: int64[N] in {0, 1, 2, 3, 255}
          - superclass_probabilities: float32[N, 4]
          - confidences: float32[N] in [0.0, 1.0]
        """
        if isinstance(logits, torch.Tensor):
            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)
                confs, native_preds = torch.max(probs, dim=-1)
                probs_np = probs.detach().cpu().numpy().astype(np.float32)
                confs_np = confs.detach().cpu().numpy().astype(np.float32)
                native_preds_np = native_preds.detach().cpu().numpy().astype(np.int64)
        else:
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs_np = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            native_preds_np = np.argmax(probs_np, axis=-1).astype(np.int64)
            confs_np = np.max(probs_np, axis=-1).astype(np.float32)

        sih_preds_np = self.remap_predictions(native_preds_np)

        # Aggregate native class probabilities into 4 super-classes
        if probs_np.shape[1] == 4 and self.native_source == "sih_direct":
            super_probs = probs_np
        else:
            N = len(native_preds_np)
            super_probs = np.zeros((N, 4), dtype=np.float32)
            for native_c, sih_c in self.mapping.items():
                if native_c < probs_np.shape[1] and sih_c in (0, 1, 2, 3):
                    super_probs[:, sih_c] += probs_np[:, native_c]

        # Normalize superclass probabilities where sum > 0
        p_sums = np.sum(super_probs, axis=1, keepdims=True)
        valid_sums = p_sums > 0
        super_probs = np.where(valid_sums, super_probs / np.maximum(p_sums, 1e-6), 0.25)

        return sih_preds_np, super_probs.astype(np.float32), confs_np
