"""
Canonical 3-Zone Foveated Voxel Downsampler (SIH PS 26130).
Integrates native C++/LLVM single-pass foveation accelerator with automatic fallback to reference Python.
"""

from typing import Optional, Tuple
import numpy as np

from ml.data.amit_adapter import (
    FoveatedSamplingReport,
    FoveatedZoneStats,
    FoveatedVoxelSampler as ReferenceFoveatedVoxelSampler,
    validate_point_label_alignment,
)
from src.core.native_foveation import NativeFoveationAccelerator


class CanonicalFoveatedSampler:
    """Canonical 3-Zone Foveated Voxel Downsampling Engine."""

    def __init__(
        self,
        near_dist: float = 10.0,
        near_voxel: float = 0.05,
        mid_dist: float = 40.0,
        mid_voxel: float = 0.15,
        far_dist: float = 100.0,
        far_voxel: float = 0.50,
        use_native: bool = True,
    ):
        self.near_dist = near_dist
        self.near_voxel = near_voxel
        self.mid_dist = mid_dist
        self.mid_voxel = mid_voxel
        self.far_dist = far_dist
        self.far_voxel = far_voxel
        self.use_native = use_native

        self.native_engine = NativeFoveationAccelerator(
            near_dist=near_dist,
            near_voxel=near_voxel,
            mid_dist=mid_dist,
            mid_voxel=mid_voxel,
            far_dist=far_dist,
            far_voxel=far_voxel,
        )
        self.reference_engine = ReferenceFoveatedVoxelSampler(
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
        use_native: Optional[bool] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], FoveatedSamplingReport]:
        """Apply 3-zone foveated downsampling to points and labels."""
        run_native = self.use_native if use_native is None else use_native
        if run_native:
            return self.native_engine.sample(points, labels)
        return self.reference_engine.sample(points, labels)
