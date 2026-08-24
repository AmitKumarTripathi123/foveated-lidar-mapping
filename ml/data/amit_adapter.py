"""Amit Foveated Voxel Sampling Adapter (Phase 2 -> Phase 3 Bridge).

Implements 3-Zone Variable-Resolution Foveated Voxel Downsampling:
  - Zone 1 (Near-Field, 0m <= d < 10m)   : voxel_size = 0.05m
  - Zone 2 (Mid-Field,  10m <= d < 40m)  : voxel_size = 0.15m
  - Zone 3 (Far-Field,  40m <= d <= 100m): voxel_size = 0.50m
  - Outer Boundary (d > 100m)            : Filtered out

Distance Policy:
  - 3D Euclidean distance: d = sqrt(x^2 + y^2 + z^2)

Guarantees:
  - Strict 1-to-1 point-label alignment: len(points) == len(labels)
  - Comprehensive zone reduction statistics reporting
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from ml.data.dataset import validate_point_label_alignment


@dataclass
class FoveatedZoneStats:
    """Statistics for a single foveated distance zone."""
    zone_name: str
    min_dist: float
    max_dist: float
    voxel_size: float
    input_count: int
    output_count: int
    reduction_pct: float


@dataclass
class FoveatedSamplingReport:
    """Comprehensive execution report for 3-zone foveated voxel sampling."""
    original_count: int
    foveated_count: int
    overall_reduction_pct: float
    zone_stats: List[FoveatedZoneStats]
    filtered_out_count: int
    alignment_pass: bool


def voxel_grid_downsample(
    points: np.ndarray,
    labels: Optional[np.ndarray],
    voxel_size: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Downsample point cloud and optional labels using uniform 3D voxel grid.

    Selects the first point and label occurring within each non-empty voxel.

    Args:
        points: Point cloud array of shape (N, 4).
        labels: Optional label array of shape (N,).
        voxel_size: Edge length of cubic voxel in meters.

    Returns:
        Tuple: (downsampled_points, downsampled_labels)
    """
    if points.shape[0] == 0:
        return points, labels

    xyz = points[:, :3]
    min_bound = np.min(xyz, axis=0)

    # Compute integer 3D voxel grid coordinates
    voxel_coords = np.floor((xyz - min_bound) / voxel_size).astype(np.int64)

    # Fast structured hashing of 3D integer coordinates
    # Sort voxel coordinates to find unique cells
    # Convert 3D coords to structured 1D void view for fast unique selection
    v_min = np.min(voxel_coords, axis=0)
    v_shifted = voxel_coords - v_min
    v_max = np.max(v_shifted, axis=0) + 1

    # Linear cell index: c = x + y*dx + z*dx*dy
    linear_idx = (
        v_shifted[:, 0]
        + v_shifted[:, 1] * v_max[0]
        + v_shifted[:, 2] * (v_max[0] * v_max[1])
    )

    _, unique_indices = np.unique(linear_idx, return_index=True)
    unique_indices.sort()  # Maintain temporal/spatial sequence order

    down_points = points[unique_indices]
    down_labels = labels[unique_indices] if labels is not None else None

    return down_points, down_labels


class FoveatedVoxelSampler:
    """Amit's 3-Zone Foveated Voxel Downsampler."""

    def __init__(
        self,
        near_dist: float = 10.0,
        near_voxel: float = 0.05,
        mid_dist: float = 40.0,
        mid_voxel: float = 0.15,
        far_dist: float = 100.0,
        far_voxel: float = 0.50,
    ):
        """Initialize 3-zone parameters.

        Args:
            near_dist: Outer boundary of near-field zone in meters (default: 10m).
            near_voxel: Voxel size for near-field zone in meters (default: 0.05m).
            mid_dist: Outer boundary of mid-field zone in meters (default: 40m).
            mid_voxel: Voxel size for mid-field zone in meters (default: 0.15m).
            far_dist: Outer boundary of far-field zone in meters (default: 100m).
            far_voxel: Voxel size for far-field zone in meters (default: 0.50m).
        """
        self.near_dist = near_dist
        self.near_voxel = near_voxel
        self.mid_dist = mid_dist
        self.mid_voxel = mid_voxel
        self.far_dist = far_dist
        self.far_voxel = far_voxel

        from src.core.native_foveation import NativeFoveationAccelerator
        self.native_accelerator = NativeFoveationAccelerator(
            near_dist=near_dist,
            near_voxel=near_voxel,
            mid_dist=mid_dist,
            mid_voxel=mid_voxel,
            far_dist=far_dist,
            far_voxel=far_voxel,
        )

    def sample(
        self,
        points: np.ndarray,
        labels: Optional[np.ndarray] = None,
        use_native: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], FoveatedSamplingReport]:
        """Apply 3-zone foveated downsampling to points and corresponding labels.

        Args:
            points: Point cloud array of shape (N, 4) with [x, y, z, intensity].
            labels: Optional label array of shape (N,).
            use_native: Whether to use native C++/LLVM acceleration (default: True).

        Returns:
            Tuple: (foveated_points, foveated_labels, report)
        """
        if use_native:
            return self.native_accelerator.sample(points, labels)
        return self.sample_reference_python(points, labels)

    def sample_reference_python(
        self,
        points: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], FoveatedSamplingReport]:
        """Reference Python NumPy implementation of 3-zone foveated downsampling."""
        orig_count = points.shape[0]
        if labels is not None:
            validate_point_label_alignment(points, labels)

        if orig_count == 0:
            empty_report = FoveatedSamplingReport(
                original_count=0,
                foveated_count=0,
                overall_reduction_pct=0.0,
                zone_stats=[],
                filtered_out_count=0,
                alignment_pass=True,
            )
            return points, labels, empty_report

        # Compute squared 3D Euclidean distances (avoids expensive sqrt)
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        d2 = x * x + y * y + z * z

        n_d2 = self.near_dist * self.near_dist
        m_d2 = self.mid_dist * self.mid_dist
        f_d2 = self.far_dist * self.far_dist

        # 1. Near-Field Zone (0 <= d < 10m)
        near_mask = (d2 >= 0.0) & (d2 < n_d2)
        near_pts = points[near_mask]
        near_lbls = labels[near_mask] if labels is not None else None
        near_down_pts, near_down_lbls = voxel_grid_downsample(
            near_pts, near_lbls, self.near_voxel
        )

        # 2. Mid-Field Zone (10m <= d < 40m)
        mid_mask = (d2 >= n_d2) & (d2 < m_d2)
        mid_pts = points[mid_mask]
        mid_lbls = labels[mid_mask] if labels is not None else None
        mid_down_pts, mid_down_lbls = voxel_grid_downsample(
            mid_pts, mid_lbls, self.mid_voxel
        )

        # 3. Far-Field Zone (40m <= d <= 100m)
        far_mask = (d2 >= m_d2) & (d2 <= f_d2)
        far_pts = points[far_mask]
        far_lbls = labels[far_mask] if labels is not None else None
        far_down_pts, far_down_lbls = voxel_grid_downsample(
            far_pts, far_lbls, self.far_voxel
        )

        # 4. Out-of-bounds (> 100m)
        out_mask = d2 > f_d2
        out_count = int(out_mask.sum())

        # Concatenate foveated zones
        foveated_points = np.vstack([near_down_pts, mid_down_pts, far_down_pts])
        if labels is not None:
            foveated_labels = np.concatenate([near_down_lbls, mid_down_lbls, far_down_lbls])
            validate_point_label_alignment(foveated_points, foveated_labels)
        else:
            foveated_labels = None

        final_count = foveated_points.shape[0]
        reduction_pct = ((orig_count - final_count) / orig_count) * 100.0 if orig_count > 0 else 0.0

        # Calculate per-zone metrics
        def _get_reduc(in_c, out_c):
            return ((in_c - out_c) / in_c) * 100.0 if in_c > 0 else 0.0

        zone_stats = [
            FoveatedZoneStats(
                zone_name="Near-Field (0-10m)",
                min_dist=0.0,
                max_dist=self.near_dist,
                voxel_size=self.near_voxel,
                input_count=int(near_mask.sum()),
                output_count=near_down_pts.shape[0],
                reduction_pct=_get_reduc(int(near_mask.sum()), near_down_pts.shape[0]),
            ),
            FoveatedZoneStats(
                zone_name="Mid-Field (10-40m)",
                min_dist=self.near_dist,
                max_dist=self.mid_dist,
                voxel_size=self.mid_voxel,
                input_count=int(mid_mask.sum()),
                output_count=mid_down_pts.shape[0],
                reduction_pct=_get_reduc(int(mid_mask.sum()), mid_down_pts.shape[0]),
            ),
            FoveatedZoneStats(
                zone_name="Far-Field (40-100m)",
                min_dist=self.mid_dist,
                max_dist=self.far_dist,
                voxel_size=self.far_voxel,
                input_count=int(far_mask.sum()),
                output_count=far_down_pts.shape[0],
                reduction_pct=_get_reduc(int(far_mask.sum()), far_down_pts.shape[0]),
            ),
        ]

        report = FoveatedSamplingReport(
            original_count=orig_count,
            foveated_count=final_count,
            overall_reduction_pct=reduction_pct,
            zone_stats=zone_stats,
            filtered_out_count=out_count,
            alignment_pass=True,
        )

        return foveated_points, foveated_labels, report
