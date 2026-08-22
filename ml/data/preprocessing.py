"""LiDAR Preprocessing and Model-Ready Data Pipeline for Phase 2 & 3.

Provides modular, label-preserving preprocessing stages including:
  1. Invalid point removal (NaN/Inf filtering with shared boolean masks)
  2. Optional spatial range filtering (metric 3D bounding box)
  3. Configurable point-count handling & sampling (keep_all, random, deterministic, with replacement, pad)
  4. Coordinate and intensity handling
  5. Optional SIH 4-Class label remapping
  6. Strict point-label correspondence verification after EVERY transformation stage
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import yaml

from ml.data.dataset import LiDARError, validate_point_label_alignment


class PreprocessingError(LiDARError):
    """Raised when an error occurs during preprocessing."""
    pass


@dataclass
class InvalidPointsConfig:
    """Configuration for invalid point removal."""
    remove: bool = True


@dataclass
class RangeFilterConfig:
    """Configuration for spatial range filtering."""
    enabled: bool = False
    min_x: Optional[float] = None
    max_x: Optional[float] = None
    min_y: Optional[float] = None
    max_y: Optional[float] = None
    min_z: Optional[float] = None
    max_z: Optional[float] = None


@dataclass
class SamplingConfig:
    """Configuration for point cloud sampling and point-count handling."""
    strategy: str = "keep_all"  # keep_all, random, deterministic, random_with_replacement, pad
    num_points: Optional[int] = None
    seed: Optional[int] = 42


@dataclass
class CoordinatesConfig:
    """Configuration for coordinate normalization."""
    normalization: str = "none"  # none, minmax, standard


@dataclass
class IntensityConfig:
    """Configuration for intensity normalization."""
    normalization: str = "none"  # none, minmax, standard


@dataclass
class PreprocessingConfig:
    """Complete preprocessing configuration."""
    invalid_points: InvalidPointsConfig = field(default_factory=InvalidPointsConfig)
    range_filter: RangeFilterConfig = field(default_factory=RangeFilterConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    coordinates: CoordinatesConfig = field(default_factory=CoordinatesConfig)
    intensity: IntensityConfig = field(default_factory=IntensityConfig)
    remap_labels: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreprocessingConfig":
        """Build configuration from a nested dictionary."""
        p_data = data.get("preprocessing", data)

        inv_data = p_data.get("invalid_points", {})
        rf_data = p_data.get("range_filter", {})
        samp_data = p_data.get("sampling", {})
        coord_data = p_data.get("coordinates", {})
        int_data = p_data.get("intensity", {})
        remap_labels = p_data.get("remap_labels", False)

        return cls(
            invalid_points=InvalidPointsConfig(**inv_data),
            range_filter=RangeFilterConfig(**rf_data),
            sampling=SamplingConfig(**samp_data),
            coordinates=CoordinatesConfig(**coord_data),
            intensity=IntensityConfig(**int_data),
            remap_labels=remap_labels,
        )

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "PreprocessingConfig":
        """Load configuration from a YAML file."""
        path = Path(yaml_path)
        if not path.is_file():
            raise PreprocessingError(f"Configuration file not found: {path.resolve()}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)


@dataclass
class ProcessingReport:
    """Execution summary report for a preprocessed scan."""
    sequence: Optional[str]
    frame: Optional[str]
    original_point_count: int
    invalid_points_removed: int
    range_filtered_points: int
    sampled_points_count: int
    final_point_count: int
    final_label_count: Optional[int]
    input_dtype: str
    output_dtype: str
    sampling_strategy: str
    sampling_seed: Optional[int]
    alignment_pass: bool
    passed: bool


@dataclass
class PreprocessedSample:
    """Encapsulates preprocessed points, labels, metadata, and quality report."""
    points: np.ndarray
    labels: Optional[np.ndarray]
    metadata: Dict[str, Any]
    report: ProcessingReport


def filter_invalid_points(
    points: np.ndarray, labels: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
    """Remove points (and corresponding labels) containing NaN or Inf values.

    Uses a single shared boolean mask to guarantee point-label alignment.

    Args:
        points: Point cloud array of shape (N, 4).
        labels: Optional label array of shape (N,).

    Returns:
        Tuple: (filtered_points, filtered_labels, num_removed)
    """
    if points.size == 0:
        return points, labels, 0

    valid_mask = ~(np.isnan(points).any(axis=1) | np.isinf(points).any(axis=1))
    num_invalid = int((~valid_mask).sum())

    if num_invalid == 0:
        return points, labels, 0

    filtered_points = points[valid_mask]
    filtered_labels = labels[valid_mask] if labels is not None else None

    if filtered_labels is not None:
        validate_point_label_alignment(filtered_points, filtered_labels)

    return filtered_points, filtered_labels, num_invalid


def apply_range_filter(
    points: np.ndarray,
    labels: Optional[np.ndarray] = None,
    min_x: Optional[float] = None,
    max_x: Optional[float] = None,
    min_y: Optional[float] = None,
    max_y: Optional[float] = None,
    min_z: Optional[float] = None,
    max_z: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
    """Filter points and labels within a 3D spatial bounding box.

    Args:
        points: Point cloud array of shape (N, 4) with columns [x, y, z, intensity].
        labels: Optional label array of shape (N,).
        min_x, max_x, min_y, max_y, min_z, max_z: Spatial bounds in meters.

    Returns:
        Tuple: (filtered_points, filtered_labels, num_filtered_out)
    """
    if points.size == 0:
        return points, labels, 0

    mask = np.ones(points.shape[0], dtype=bool)

    if min_x is not None:
        mask &= points[:, 0] >= min_x
    if max_x is not None:
        mask &= points[:, 0] <= max_x
    if min_y is not None:
        mask &= points[:, 1] >= min_y
    if max_y is not None:
        mask &= points[:, 1] <= max_y
    if min_z is not None:
        mask &= points[:, 2] >= min_z
    if max_z is not None:
        mask &= points[:, 2] <= max_z

    num_filtered = int((~mask).sum())

    filtered_points = points[mask]
    filtered_labels = labels[mask] if labels is not None else None

    if filtered_labels is not None:
        validate_point_label_alignment(filtered_points, filtered_labels)

    return filtered_points, filtered_labels, num_filtered


def sample_points(
    points: np.ndarray,
    labels: Optional[np.ndarray] = None,
    num_points: Optional[int] = None,
    strategy: str = "keep_all",
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Sample point cloud and labels using a specified strategy while preserving correspondence.

    Args:
        points: Point cloud array of shape (N, 4).
        labels: Optional label array of shape (N,).
        num_points: Target number of points to sample/pad to.
        strategy: Sampling strategy name.
        seed: Random seed for reproducible sampling.

    Returns:
        Tuple: (sampled_points, sampled_labels, sampling_metadata)
    """
    n_pts = points.shape[0]
    metadata = {
        "strategy": strategy,
        "requested_points": num_points,
        "input_points": n_pts,
        "seed": seed,
    }

    if strategy in ("keep_all", "none") or num_points is None:
        return points, labels, metadata

    if num_points <= 0:
        raise PreprocessingError(f"Target num_points must be positive, got {num_points}")

    rng = np.random.RandomState(seed)

    if strategy in ("random", "random_sample"):
        if num_points > n_pts:
            raise PreprocessingError(
                f"Cannot perform random sampling without replacement: "
                f"requested {num_points} points, but point cloud has only {n_pts} points. "
                f"Use strategy='random_with_replacement' or 'pad' instead."
            )
        indices = rng.choice(n_pts, size=num_points, replace=False)

    elif strategy in ("deterministic", "deterministic_sample"):
        if num_points > n_pts:
            raise PreprocessingError(
                f"Cannot perform deterministic sampling without replacement: "
                f"requested {num_points} points, but point cloud has only {n_pts} points."
            )
        indices = np.linspace(0, n_pts - 1, num_points, dtype=int)

    elif strategy in ("random_with_replacement", "replace"):
        indices = rng.choice(n_pts, size=num_points, replace=True)

    elif strategy in ("pad", "zero_pad"):
        if num_points <= n_pts:
            indices = rng.choice(n_pts, size=num_points, replace=False)
            sampled_points = points[indices]
            sampled_labels = labels[indices] if labels is not None else None
            return sampled_points, sampled_labels, metadata
        else:
            padded_points = np.zeros((num_points, 4), dtype=points.dtype)
            padded_points[:n_pts] = points

            if labels is not None:
                padded_labels = np.full((num_points,), 255, dtype=labels.dtype)
                padded_labels[:n_pts] = labels
            else:
                padded_labels = None

            return padded_points, padded_labels, metadata

    else:
        raise PreprocessingError(
            f"Unknown sampling strategy '{strategy}'. "
            f"Supported: 'keep_all', 'random', 'deterministic', 'random_with_replacement', 'pad'."
        )

    sampled_points = points[indices]
    sampled_labels = labels[indices] if labels is not None else None

    if sampled_labels is not None:
        validate_point_label_alignment(sampled_points, sampled_labels)

    return sampled_points, sampled_labels, metadata


def handle_coordinates(points: np.ndarray, normalization: str = "none") -> np.ndarray:
    """Apply optional coordinate normalization while preserving array layout."""
    if normalization in ("none", "", None) or points.size == 0:
        return points

    processed = points.copy()
    xyz = processed[:, :3]

    if normalization == "minmax":
        min_val = np.min(xyz, axis=0)
        max_val = np.max(xyz, axis=0)
        diff = max_val - min_val
        diff[diff == 0] = 1.0
        processed[:, :3] = 2.0 * ((xyz - min_val) / diff) - 1.0

    elif normalization == "standard":
        mean = np.mean(xyz, axis=0)
        std = np.std(xyz, axis=0)
        std[std == 0] = 1.0
        processed[:, :3] = (xyz - mean) / std

    else:
        raise PreprocessingError(f"Unknown coordinate normalization '{normalization}'")

    return processed


def handle_intensity(points: np.ndarray, normalization: str = "none") -> np.ndarray:
    """Apply optional intensity normalization while preserving array layout."""
    if normalization in ("none", "", None) or points.size == 0:
        return points

    processed = points.copy()
    intensity = processed[:, 3]

    if normalization == "minmax":
        min_val = np.min(intensity)
        max_val = np.max(intensity)
        diff = max_val - min_val if max_val != min_val else 1.0
        processed[:, 3] = (intensity - min_val) / diff

    elif normalization == "standard":
        mean = np.mean(intensity)
        std = np.std(intensity)
        std = std if std != 0 else 1.0
        processed[:, 3] = (intensity - mean) / std

    else:
        raise PreprocessingError(f"Unknown intensity normalization '{normalization}'")

    return processed


class LidarPreprocessor:
    """Full modular LiDAR preprocessor orchestrating all Phase 2 & Phase 3 stages."""

    def __init__(
        self,
        config: Optional[Union[PreprocessingConfig, Dict[str, Any], Path, str]] = None,
        label_remapper: Optional[Any] = None,
    ):
        """Initialize preprocessor with configuration and optional label remapper.

        Args:
            config: PreprocessingConfig object, dict, or path to YAML config.
            label_remapper: Optional callable / SemanticLabelRemapper instance.
        """
        if config is None:
            self.config = PreprocessingConfig()
        elif isinstance(config, PreprocessingConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = PreprocessingConfig.from_dict(config)
        elif isinstance(config, (str, Path)):
            self.config = PreprocessingConfig.from_yaml(config)
        else:
            raise TypeError(f"Invalid config type: {type(config)}")

        self.label_remapper = label_remapper

    def __call__(
        self,
        points: np.ndarray,
        labels: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PreprocessedSample:
        """Run complete preprocessing pipeline on a point cloud and optional labels.

        Args:
            points: Raw point cloud array (N, 4) with dtype float32.
            labels: Optional semantic label array (N,) with integer dtype.
            metadata: Optional dictionary with sequence/frame identifiers.

        Returns:
            PreprocessedSample: Processed points, labels, updated metadata, and report.
        """
        meta = dict(metadata) if metadata is not None else {}
        orig_count = points.shape[0]
        orig_dtype = str(points.dtype)

        # Initial alignment validation if labels provided
        if labels is not None:
            validate_point_label_alignment(points, labels)

        cur_points = points.copy()
        cur_labels = labels.copy() if labels is not None else None

        # Stage 1: Invalid Point Removal
        num_invalid = 0
        if self.config.invalid_points.remove:
            cur_points, cur_labels, num_invalid = filter_invalid_points(
                cur_points, cur_labels
            )

        # Stage 2: Spatial Range Filter
        num_range_filtered = 0
        if self.config.range_filter.enabled:
            rf = self.config.range_filter
            cur_points, cur_labels, num_range_filtered = apply_range_filter(
                cur_points,
                cur_labels,
                min_x=rf.min_x,
                max_x=rf.max_x,
                min_y=rf.min_y,
                max_y=rf.max_y,
                min_z=rf.min_z,
                max_z=rf.max_z,
            )

        # Stage 3: Sampling / Point-Count Handling
        samp = self.config.sampling
        cur_points, cur_labels, samp_meta = sample_points(
            cur_points,
            cur_labels,
            num_points=samp.num_points,
            strategy=samp.strategy,
            seed=samp.seed,
        )

        # Stage 4: Coordinate Normalization
        cur_points = handle_coordinates(
            cur_points, normalization=self.config.coordinates.normalization
        )

        # Stage 5: Intensity Normalization
        cur_points = handle_intensity(
            cur_points, normalization=self.config.intensity.normalization
        )

        # Stage 6: Optional SIH Label Remapping
        if (self.config.remap_labels or self.label_remapper is not None) and cur_labels is not None:
            if self.label_remapper is not None:
                if hasattr(self.label_remapper, "remap"):
                    cur_labels = self.label_remapper.remap(cur_labels)
                elif callable(self.label_remapper):
                    cur_labels = self.label_remapper(cur_labels)

        # Stage 7: Final strict alignment check
        alignment_pass = True
        if cur_labels is not None:
            validate_point_label_alignment(cur_points, cur_labels)

        final_count = cur_points.shape[0]
        final_label_count = cur_labels.shape[0] if cur_labels is not None else None

        # Assemble rich metadata
        meta.update({
            "original_point_count": orig_count,
            "invalid_points_removed": num_invalid,
            "range_filtered_points": num_range_filtered,
            "final_point_count": final_count,
            "sampling_strategy": samp.strategy,
            "sampling_seed": samp.seed,
        })

        report = ProcessingReport(
            sequence=meta.get("sequence"),
            frame=meta.get("frame"),
            original_point_count=orig_count,
            invalid_points_removed=num_invalid,
            range_filtered_points=num_range_filtered,
            sampled_points_count=final_count,
            final_point_count=final_count,
            final_label_count=final_label_count,
            input_dtype=orig_dtype,
            output_dtype=str(cur_points.dtype),
            sampling_strategy=samp.strategy,
            sampling_seed=samp.seed,
            alignment_pass=alignment_pass,
            passed=True,
        )

        return PreprocessedSample(
            points=cur_points,
            labels=cur_labels,
            metadata=meta,
            report=report,
        )
