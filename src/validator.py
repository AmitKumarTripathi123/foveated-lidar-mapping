"""
Point Cloud Validation Module.
Performs rigorous coordinate, range, intensity, and coordinate-system diagnostic checks.
STRICT RULE: Validation never modifies or filters data. It only inspects and reports.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from src.types import PointCloudFrame


@dataclass
class CoordinateValidationResult:
    total_points: int
    nan_count: int
    pos_inf_count: int
    neg_inf_count: int
    invalid_point_count: int
    invalid_point_percentage: float
    is_clean: bool


@dataclass
class RangeValidationResult:
    min_range: float
    max_range: float
    mean_range: float
    median_range: float
    p95_range: float
    p99_range: float
    points_within_100m: int
    points_beyond_100m: int
    percentage_within_100m: float


@dataclass
class IntensityValidationResult:
    min_intensity: float
    max_intensity: float
    mean_intensity: float
    median_intensity: float
    p25_intensity: float
    p75_intensity: float
    p90_intensity: float
    p99_intensity: float
    detected_format: str  # 'normalized_0_1', 'integer_0_255', 'integer_0_65535', or 'unbounded_float'
    is_normalized: bool


@dataclass
class CoordinateDistributionResult:
    x_stats: Dict[str, float]
    y_stats: Dict[str, float]
    z_stats: Dict[str, float]
    forward_x_percentage: float   # % with x > 0
    lateral_y_symmetry_error: float # abs(mean(y))
    coordinate_convention_status: str = "Machine Checked + Human Confirmation Required"
    expected_convention: str = "+X forward, +Y left, +Z upward"


@dataclass
class FrameValidationSummary:
    frame_id: str
    is_valid_frame: bool
    coordinate_validation: CoordinateValidationResult
    range_validation: RangeValidationResult
    intensity_validation: IntensityValidationResult
    coordinate_distribution: CoordinateDistributionResult
    warnings: List[str] = field(default_factory=list)


class PointCloudValidator:
    """
    Performs comprehensive validation on LiDAR frames without mutating inputs.
    """

    def __init__(self, max_allowed_range: float = 100.0):
        self.max_allowed_range = max_allowed_range

    def validate_coordinates(self, points: np.ndarray) -> CoordinateValidationResult:
        """Checks for NaN, +Inf, -Inf in points [N, 4]."""
        if points is None or len(points) == 0:
            return CoordinateValidationResult(0, 0, 0, 0, 0, 0.0, True)

        total = len(points)
        nan_mask = np.isnan(points).any(axis=1)
        pos_inf_mask = np.isposinf(points).any(axis=1)
        neg_inf_mask = np.isneginf(points).any(axis=1)
        invalid_mask = nan_mask | pos_inf_mask | neg_inf_mask

        nan_count = int(np.sum(nan_mask))
        pos_inf_count = int(np.sum(pos_inf_mask))
        neg_inf_count = int(np.sum(neg_inf_mask))
        invalid_count = int(np.sum(invalid_mask))
        pct = (invalid_count / total) * 100.0 if total > 0 else 0.0

        return CoordinateValidationResult(
            total_points=total,
            nan_count=nan_count,
            pos_inf_count=pos_inf_count,
            neg_inf_count=neg_inf_count,
            invalid_point_count=invalid_count,
            invalid_point_percentage=round(pct, 4),
            is_clean=(invalid_count == 0)
        )

    def validate_ranges(self, points: np.ndarray) -> RangeValidationResult:
        """
        Calculates radial distance r = sqrt(x^2 + y^2) statistics.
        Does NOT delete out-of-range points.
        """
        if points is None or len(points) == 0:
            return RangeValidationResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0)

        # Only compute on valid finite points
        valid_mask = np.isfinite(points[:, :3]).all(axis=1)
        if not np.any(valid_mask):
            return RangeValidationResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0)

        valid_pts = points[valid_mask]
        x = valid_pts[:, 0]
        y = valid_pts[:, 1]
        r = np.sqrt(x * x + y * y)

        min_r = float(np.min(r))
        max_r = float(np.max(r))
        mean_r = float(np.mean(r))
        median_r = float(np.median(r))
        p95_r = float(np.percentile(r, 95))
        p99_r = float(np.percentile(r, 99))

        within_max = int(np.sum(r <= self.max_allowed_range))
        beyond_max = int(np.sum(r > self.max_allowed_range))
        pct_within = (within_max / len(r)) * 100.0 if len(r) > 0 else 0.0

        return RangeValidationResult(
            min_range=round(min_r, 4),
            max_range=round(max_r, 4),
            mean_range=round(mean_r, 4),
            median_range=round(median_r, 4),
            p95_range=round(p95_r, 4),
            p99_range=round(p99_r, 4),
            points_within_100m=within_max,
            points_beyond_100m=beyond_max,
            percentage_within_100m=round(pct_within, 2)
        )

    def validate_intensity(self, points: np.ndarray) -> IntensityValidationResult:
        """
        Determines intensity range, distribution, and detects format.
        Does NOT silently normalize.
        """
        if points is None or len(points) == 0 or points.shape[1] < 4:
            return IntensityValidationResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "unknown", False)

        valid_mask = np.isfinite(points[:, 3])
        if not np.any(valid_mask):
            return IntensityValidationResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "all_invalid", False)

        intensity = points[valid_mask, 3]
        min_i = float(np.min(intensity))
        max_i = float(np.max(intensity))
        mean_i = float(np.mean(intensity))
        median_i = float(np.median(intensity))
        p25_i = float(np.percentile(intensity, 25))
        p75_i = float(np.percentile(intensity, 75))
        p90_i = float(np.percentile(intensity, 90))
        p99_i = float(np.percentile(intensity, 99))

        # Format detection
        if 0.0 <= min_i and max_i <= 1.0001:
            detected_format = "normalized_0_1"
            is_normalized = True
        elif 0.0 <= min_i and max_i <= 255.0 and np.all(intensity == np.floor(intensity)):
            detected_format = "integer_0_255"
            is_normalized = False
        elif 0.0 <= min_i and max_i <= 65535.0:
            detected_format = "integer_0_65535"
            is_normalized = False
        else:
            detected_format = "unbounded_float"
            is_normalized = False

        return IntensityValidationResult(
            min_intensity=round(min_i, 4),
            max_intensity=round(max_i, 4),
            mean_intensity=round(mean_i, 4),
            median_intensity=round(median_i, 4),
            p25_intensity=round(p25_i, 4),
            p75_intensity=round(p75_i, 4),
            p90_intensity=round(p90_i, 4),
            p99_intensity=round(p99_i, 4),
            detected_format=detected_format,
            is_normalized=is_normalized
        )

    def validate_coordinate_distribution(self, points: np.ndarray) -> CoordinateDistributionResult:
        """
        Diagnostics for X, Y, Z coordinate distribution to check sensor conventions.
        """
        if points is None or len(points) == 0:
            empty_stat = {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
            return CoordinateDistributionResult(empty_stat, empty_stat, empty_stat, 0.0, 0.0)

        valid_mask = np.isfinite(points[:, :3]).all(axis=1)
        valid_pts = points[valid_mask]

        def _calc_stats(arr: np.ndarray) -> Dict[str, float]:
            return {
                "min": round(float(np.min(arr)), 3),
                "max": round(float(np.max(arr)), 3),
                "mean": round(float(np.mean(arr)), 3),
                "std": round(float(np.std(arr)), 3),
                "p25": round(float(np.percentile(arr, 25)), 3),
                "p75": round(float(np.percentile(arr, 75)), 3)
            }

        x_stats = _calc_stats(valid_pts[:, 0])
        y_stats = _calc_stats(valid_pts[:, 1])
        z_stats = _calc_stats(valid_pts[:, 2])

        forward_pct = (np.sum(valid_pts[:, 0] > 0) / len(valid_pts)) * 100.0
        lateral_sym_err = abs(float(np.mean(valid_pts[:, 1])))

        return CoordinateDistributionResult(
            x_stats=x_stats,
            y_stats=y_stats,
            z_stats=z_stats,
            forward_x_percentage=round(forward_pct, 2),
            lateral_y_symmetry_error=round(lateral_sym_err, 4),
            coordinate_convention_status="Machine Checked + Human Confirmation Required",
            expected_convention="+X forward, +Y left, +Z upward"
        )

    def validate_frame(self, frame: PointCloudFrame) -> FrameValidationSummary:
        """Runs complete validation suite on a PointCloudFrame."""
        warnings = []
        if not frame.is_valid:
            warnings.extend(frame.validation_notes)

        coord_res = self.validate_coordinates(frame.points)
        if not coord_res.is_clean:
            warnings.append(
                f"Coordinate warning: {coord_res.invalid_point_count} invalid points "
                f"({coord_res.invalid_point_percentage}%)."
            )

        range_res = self.validate_ranges(frame.points)
        if range_res.points_beyond_100m > 0:
            warnings.append(
                f"Range note: {range_res.points_beyond_100m} points exceed {self.max_allowed_range}m."
            )

        intensity_res = self.validate_intensity(frame.points)
        if not intensity_res.is_normalized:
            warnings.append(
                f"Intensity format detected as {intensity_res.detected_format} "
                f"(range [{intensity_res.min_intensity}, {intensity_res.max_intensity}]). "
                f"Requires explicit normalization stage."
            )

        dist_res = self.validate_coordinate_distribution(frame.points)

        return FrameValidationSummary(
            frame_id=frame.frame_id,
            is_valid_frame=frame.is_valid and coord_res.is_clean,
            coordinate_validation=coord_res,
            range_validation=range_res,
            intensity_validation=intensity_res,
            coordinate_distribution=dist_res,
            warnings=warnings
        )

    @staticmethod
    def normalize_intensity(points: np.ndarray, detected_format: str = "auto") -> np.ndarray:
        """
        Explicit, non-destructive normalization of intensity channel to float32 [0, 1].
        Returns a new array.
        """
        if points is None or len(points) == 0:
            return points.copy()

        norm_points = points.copy()
        intensity = norm_points[:, 3]
        valid_mask = np.isfinite(intensity)

        if not np.any(valid_mask):
            return norm_points

        valid_i = intensity[valid_mask]
        min_v = np.min(valid_i)
        max_v = np.max(valid_i)

        if detected_format == "auto":
            if 0.0 <= min_v and max_v <= 1.0001:
                # Already normalized
                np.clip(intensity, 0.0, 1.0, out=intensity)
            elif max_v <= 255.0:
                intensity[valid_mask] = valid_i / 255.0
            elif max_v <= 65535.0:
                intensity[valid_mask] = valid_i / 65535.0
            else:
                if max_v > min_v:
                    intensity[valid_mask] = (valid_i - min_v) / (max_v - min_v)
                else:
                    intensity[valid_mask] = 0.0
        elif detected_format == "integer_0_255":
            intensity[valid_mask] = np.clip(valid_i / 255.0, 0.0, 1.0)
        elif detected_format == "integer_0_65535":
            intensity[valid_mask] = np.clip(valid_i / 65535.0, 0.0, 1.0)
        elif detected_format == "min_max":
            if max_v > min_v:
                intensity[valid_mask] = (valid_i - min_v) / (max_v - min_v)
            else:
                intensity[valid_mask] = 0.0

        norm_points[:, 3] = intensity.astype(np.float32)
        return norm_points
