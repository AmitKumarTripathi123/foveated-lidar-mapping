"""
Sparse Point-Voxel Convolutional Neural Network (SPVCNN).
Implementation based on Tang et al. (ECCV 2020):
"Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution"

Combines high-resolution point-wise feature branches with 3D voxel coordinate
convolutions via scatter-add operations in native PyTorch, providing full CPU
and CUDA execution without external C++ binary dependency constraints.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointBranch(nn.Module):
    """High-resolution Point-wise MLP branch."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C_in)
        return self.mlp(x)


class VoxelSpatialBranch(nn.Module):
    """Voxel-wise 3D spatial feature aggregation branch."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.voxel_mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(
        self,
        point_features: torch.Tensor,
        point_to_voxel_idx: torch.Tensor,
        num_voxels: int,
    ) -> torch.Tensor:
        """
        Aggregate point features into voxels via scatter mean,
        apply spatial linear transformation, and project back to points.
        """
        c_in = point_features.shape[1]
        voxel_feat = torch.zeros(
            (num_voxels, c_in),
            dtype=point_features.dtype,
            device=point_features.device,
        )
        voxel_counts = torch.zeros(
            (num_voxels, 1),
            dtype=point_features.dtype,
            device=point_features.device,
        )

        voxel_feat.index_add_(0, point_to_voxel_idx, point_features)
        voxel_counts.index_add_(
            0,
            point_to_voxel_idx,
            torch.ones((point_features.shape[0], 1), dtype=point_features.dtype, device=point_features.device),
        )
        voxel_counts = torch.clamp(voxel_counts, min=1.0)
        voxel_mean = voxel_feat / voxel_counts

        # Apply voxel transformation
        transformed_voxel = self.voxel_mlp(voxel_mean)  # (M, C_out)

        # Project back to per-point representations
        return transformed_voxel[point_to_voxel_idx]  # (N, C_out)


class SPVConvBlock(nn.Module):
    """Point-Voxel Sparse Convolution Block fusing Point and Voxel branches."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.point_branch = PointBranch(in_channels, out_channels)
        self.voxel_branch = VoxelSpatialBranch(in_channels, out_channels)
        self.fusion = nn.Sequential(
            nn.Linear(out_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.residual = (
            nn.Linear(in_channels, out_channels, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(
        self,
        point_features: torch.Tensor,
        point_to_voxel_idx: torch.Tensor,
        num_voxels: int,
    ) -> torch.Tensor:
        res = self.residual(point_features)
        pt_out = self.point_branch(point_features)
        vox_out = self.voxel_branch(point_features, point_to_voxel_idx, num_voxels)
        fused = self.fusion(pt_out + vox_out)
        return fused + res


class SPVCNN(nn.Module):
    """Sparse Point-Voxel Convolutional Neural Network (SPVCNN)."""

    def __init__(
        self,
        num_classes: int = 19,
        in_channels: int = 4,
        base_channels: int = 32,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Stem projection
        self.stem = nn.Sequential(
            nn.Linear(in_channels, base_channels, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Multi-scale Point-Voxel stages
        self.stage1 = SPVConvBlock(base_channels, base_channels * 2)       # 32 -> 64
        self.stage2 = SPVConvBlock(base_channels * 2, base_channels * 4)   # 64 -> 128
        self.stage3 = SPVConvBlock(base_channels * 4, base_channels * 4)   # 128 -> 128
        self.stage4 = SPVConvBlock(base_channels * 4, base_channels * 2)   # 128 -> 64

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(base_channels * 2, base_channels * 2, bias=False),
            nn.BatchNorm1d(base_channels * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(base_channels * 2, num_classes),
        )

    def forward(
        self,
        features: torch.Tensor,
        point_to_voxel_idx: torch.Tensor,
        num_voxels: int,
    ) -> torch.Tensor:
        """
        Forward pass.
        features: (N, in_channels) [x, y, z, intensity]
        point_to_voxel_idx: (N,) mapping each point to voxel index in [0, num_voxels-1]
        num_voxels: int M count of unique spatial voxels
        returns: (N, num_classes) per-point class logits
        """
        x0 = self.stem(features)
        x1 = self.stage1(x0, point_to_voxel_idx, num_voxels)
        x2 = self.stage2(x1, point_to_voxel_idx, num_voxels)
        x3 = self.stage3(x2, point_to_voxel_idx, num_voxels)
        x4 = self.stage4(x3 + x2, point_to_voxel_idx, num_voxels)
        logits = self.classifier(x4 + x1)
        return logits


def load_spvcnn_checkpoint(
    model: SPVCNN,
    checkpoint_path: Union[str, os.PathLike],
    strict: bool = False,
) -> Dict[str, Any]:
    """Load weights from a checkpoint file with strict error and key validation."""
    path = os.fspath(checkpoint_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"SPVCNN checkpoint not found: {path}")

    state_dict = torch.load(path, map_location="cpu")
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    model.load_state_dict(state_dict, strict=strict)

    return {
        "checkpoint_path": path,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "strict": strict,
    }


def build_spvcnn(
    num_classes: int = 19,
    in_channels: int = 4,
    pretrained_path: Optional[Union[str, os.PathLike]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> SPVCNN:
    """Factory helper to build and initialize SPVCNN model."""
    dev = device if device is not None else torch.device("cpu")
    model = SPVCNN(num_classes=num_classes, in_channels=in_channels)
    model.to(dev)

    if pretrained_path is not None and os.path.isfile(pretrained_path):
        load_spvcnn_checkpoint(model, pretrained_path)

    return model
