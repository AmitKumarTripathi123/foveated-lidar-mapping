"""
Range Filtering Module.
Performs non-destructive range and validity filtering on LiDAR frames.
Limits processing to 0 <= r <= max_range (default 100m).
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from src.types import PointCloudFrame


@dataclass
class RangeFilterReport:
    input_points: int
    removed_invalid_points: int
    removed_out_of_range_points: int
    output_points: int
    retention_percentage: float
    min_range: float
    max_range: float


class RangeFilter:
    """
    Filters points by radial horizontal distance r = sqrt(x^2 + y^2) and removes NaNs/Infs.
    Never modifies input data.
    """

    def __init__(self, min_range: float = 0.0, max_range: float = 100.0):
        self.min_range = float(min_range)
        self.max_range = float(max_range)

    def filter_frame(self, frame: PointCloudFrame) -> Tuple[PointCloudFrame, RangeFilterReport]:
        """
        Filters points in the frame. Returns (new_filtered_frame, report).
        """
        if frame.points is None or len(frame.points) == 0:
            empty_report = RangeFilterReport(
                input_points=0,
                removed_invalid_points=0,
                removed_out_of_range_points=0,
                output_points=0,
                retention_percentage=0.0,
                min_range=self.min_range,
                max_range=self.max_range
            )
            return frame.copy(), empty_report

        pts = frame.points
        input_count = len(pts)

        # 1. Check finite valid coordinates
        valid_coords = np.isfinite(pts[:, :3]).all(axis=1) & np.isfinite(pts[:, 3])
        invalid_count = int(np.sum(~valid_coords))

        pts_valid = pts[valid_coords]
        labels_valid = frame.labels[valid_coords] if frame.labels is not None else None
        conf_valid = frame.confidences[valid_coords] if frame.confidences is not None else None

        # 2. Compute radial horizontal distance r = sqrt(x^2 + y^2)
        x = pts_valid[:, 0]
        y = pts_valid[:, 1]
        r = np.sqrt(x * x + y * y)

        range_mask = (r >= self.min_range) & (r <= self.max_range)
        out_of_range_count = int(np.sum(~range_mask))

        filtered_pts = pts_valid[range_mask].astype(np.float32)
        filtered_labels = labels_valid[range_mask].astype(np.uint32) if labels_valid is not None else None
        filtered_conf = conf_valid[range_mask].astype(np.float32) if conf_valid is not None else None

        output_count = len(filtered_pts)
        retention = (output_count / input_count) * 100.0 if input_count > 0 else 0.0

        report = RangeFilterReport(
            input_points=input_count,
            removed_invalid_points=invalid_count,
            removed_out_of_range_points=out_of_range_count,
            output_points=output_count,
            retention_percentage=round(retention, 2),
            min_range=self.min_range,
            max_range=self.max_range
        )

        filtered_frame = PointCloudFrame(
            points=filtered_pts,
            labels=filtered_labels,
            confidences=filtered_conf,
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            sequence_id=frame.sequence_id,
            is_valid=frame.is_valid,
            validation_notes=list(frame.validation_notes),
            metadata=dict(frame.metadata, range_filtered=True, max_range=self.max_range)
        )

        return filtered_frame, report
