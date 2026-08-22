"""PointNet++ Point-Wise Semantic Segmentation Architecture (Phase 4).

Pure PyTorch implementation supporting hierarchical Set Abstraction (SA),
Feature Propagation (FP) with 3-NN interpolation, and point-wise classification head.
Predicts exactly 4 SIH semantic classes:
    0: drivable_terrain
    1: non_drivable_terrain
    2: static_obstacle
    3: dynamic_object
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Calculate Euclidean squared distance between each two points.

    Args:
        src: Source points tensor, [B, N, C]
        dst: Target points tensor, [B, M, C]

    Returns:
        torch.Tensor: Distance matrix, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather point features given batch index tensor.

    Args:
        points: Input points data, [B, N, C]
        idx: Sample index data, [B, S] or [B, S, K]

    Returns:
        torch.Tensor: Indexed points data, [B, S, C] or [B, S, K, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest Point Sampling (FPS) algorithm.

    Args:
        xyz: Point cloud coordinates, [B, N, 3]
        npoint: Number of samples

    Returns:
        torch.Tensor: Sampled points indices, [B, npoint]
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """Find points within ball radius around query centroids.

    Args:
        radius: Local sphere radius
        nsample: Maximum number of points in each local region
        xyz: All input points coordinates, [B, N, 3]
        new_xyz: Query centroids coordinates, [B, S, 3]

    Returns:
        torch.Tensor: Grouped points indices, [B, S, nsample]
    """
    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(
    npoint: Optional[int],
    radius: float,
    nsample: int,
    xyz: torch.Tensor,
    points: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample centroids and group neighbor points with relative coordinates.

    Args:
        npoint: Number of centroids to sample (if None, use all points)
        radius: Ball query radius
        nsample: Number of points in each local region
        xyz: Input points coordinates, [B, N, 3]
        points: Input point features, [B, N, D] or None

    Returns:
        Tuple: (new_xyz [B, S, 3], new_points [B, S, nsample, 3 + D])
    """
    B, N, C = xyz.shape
    if npoint is not None and npoint < N:
        fps_idx = farthest_point_sample(xyz, npoint)
        new_xyz = index_points(xyz, fps_idx)
    else:
        new_xyz = xyz

    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, S, nsample, 3]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, -1, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, S, nsample, 3 + D]
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    """PointNet++ Set Abstraction (SA) layer with shared 2D convolution."""

    def __init__(
        self,
        npoint: Optional[int],
        radius: float,
        nsample: int,
        in_channel: int,
        mlp: List[int],
        group_all: bool = False,
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(
        self, xyz: torch.Tensor, points: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            xyz: Input coordinates, [B, N, 3]
            points: Input features, [B, C, N] or None

        Returns:
            Tuple: (new_xyz [B, S, 3], new_points [B, C_out, S])
        """
        if points is not None:
            points = points.transpose(1, 2)  # [B, N, C]

        new_xyz, new_points = sample_and_group(
            self.npoint, self.radius, self.nsample, xyz, points
        )

        # [B, S, nsample, 3+D] -> [B, 3+D, nsample, S]
        new_points = new_points.permute(0, 3, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]  # [B, C_out, S]
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    """PointNet++ Feature Propagation (FP) layer with 3-NN inverse-distance interpolation."""

    def __init__(self, in_channel: int, mlp: List[int]):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(
        self,
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
        points1: Optional[torch.Tensor],
        points2: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            xyz1: Higher-resolution points coordinates, [B, N, 3]
            xyz2: Lower-resolution points coordinates, [B, S, 3]
            points1: Skip-connection features from higher level, [B, C1, N] or None
            points2: Coarser features from lower level, [B, C2, S]

        Returns:
            torch.Tensor: Interpolated and fused per-point features, [B, C_out, N]
        """
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, 1, N)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-10)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_points = torch.sum(
                index_points(points2.transpose(1, 2), idx) * weight.view(B, N, 3, 1),
                dim=2,
            ).transpose(1, 2)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=1)
        else:
            new_points = interpolated_points

        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))

        return new_points


class PointNet2SemSeg(nn.Module):
    """PointNet++ Model for Point-Wise 3D Semantic Segmentation.

    Input:
        points: [B, N, 4] where channels 0:3 are XYZ and channel 3 is intensity.

    Output:
        logits: [B, N, 4] raw class logits (num_classes=4).
    """

    def __init__(self, num_classes: int = 4, in_channels: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Additional feature channels beyond 3D coordinates (intensity = 1)
        additional_channel = in_channels - 3

        # Set Abstraction Hierarchy
        self.sa1 = PointNetSetAbstraction(
            npoint=1024, radius=0.2, nsample=32, in_channel=additional_channel + 3, mlp=[32, 32, 64]
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=256, radius=0.4, nsample=32, in_channel=64 + 3, mlp=[64, 64, 128]
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=64, radius=0.8, nsample=32, in_channel=128 + 3, mlp=[128, 128, 256]
        )
        self.sa4 = PointNetSetAbstraction(
            npoint=16, radius=1.6, nsample=32, in_channel=256 + 3, mlp=[256, 256, 512]
        )

        # Feature Propagation Hierarchy
        self.fp4 = PointNetFeaturePropagation(in_channel=512 + 256, mlp=[256, 256])
        self.fp3 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 64, mlp=[128, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + additional_channel, mlp=[128, 128, 128])

        # Segmentation Head
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            points: Input tensor of shape [B, N, 4] with [x, y, z, intensity]

        Returns:
            torch.Tensor: Per-point logits of shape [B, N, num_classes]
        """
        B, N, C = points.shape
        xyz = points[:, :, :3].contiguous()

        if C > 3:
            features = points[:, :, 3:].transpose(1, 2).contiguous()  # [B, C-3, N]
        else:
            features = None

        # Set Abstraction Hierarchy (Downsampling & Feature Extraction)
        l1_xyz, l1_points = self.sa1(xyz, features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # Feature Propagation Hierarchy (Upsampling & Skip Connections)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, features, l1_points)  # [B, 128, N]

        # Segmentation Head
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)  # [B, num_classes, N]

        logits = x.transpose(1, 2).contiguous()  # [B, N, num_classes]
        return logits


def build_model(
    name: str = "pointnet2_semseg",
    num_classes: int = 4,
    in_channels: int = 4,
    **kwargs,
) -> nn.Module:
    """Model factory for building 3D point cloud segmentation models.

    Args:
        name: Model architecture name.
        num_classes: Number of target semantic classes (default: 4).
        in_channels: Total input channels (default: 4 for [x, y, z, intensity]).

    Returns:
        nn.Module: Initialized PyTorch model.
    """
    if name.lower() in ("pointnet2", "pointnet2_semseg", "pointnet++"):
        return PointNet2SemSeg(num_classes=num_classes, in_channels=in_channels)
    else:
        raise ValueError(f"Unknown model name '{name}'. Supported: 'pointnet2_semseg'.")
