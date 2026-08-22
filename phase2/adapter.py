"""
Phase-2 ML to 2.5D Mapping Adapter (MLToMappingAdapter).
Connects AI semantic predictions to the 2.5D Foveated Grid Map representation,
ensuring strict spatial alignment, elevation preservation, and layer generation.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.types import SuperClass, PointCloudFrame
from src.foveated_grid import (
    FoveatedGrid25D,
    GridMap25D,
    GridCell25D,
    distance_to_band,
    xy_to_cell,
    DEFAULT_FROZEN_BANDS
)
from phase2.inference.predictor import SemanticPrediction


class MLToMappingAdapter:
    """
    Adapter converting Phase-2 SemanticPrediction into a multi-layer GridMap25D.
    Guarantees that point-wise predictions and spatial cell indices maintain 100% invariant alignment.
    """
    def __init__(self, bands: Optional[List[Any]] = None, max_range: float = 100.0):
        self.bands = bands if bands is not None else DEFAULT_FROZEN_BANDS
        self.max_range = float(max_range)
        self.grid_builder = FoveatedGrid25D(bands=self.bands, max_range=self.max_range)

    def prediction_to_grid(self, prediction: SemanticPrediction) -> GridMap25D:
        """
        Converts a SemanticPrediction into a fully populated 2.5D GridMap25D.
        """
        prediction.validate_interface()
        return self.grid_builder.build_grid(
            points=prediction.points,
            labels=prediction.predicted_class,
            confidences=prediction.confidence,
            frame_id=prediction.frame_id,
            timestamp=prediction.timestamp,
            sequence_id=prediction.sequence_id
        )

    def frame_and_pred_to_grid(
        self,
        frame: PointCloudFrame,
        predicted_class: np.ndarray,
        confidence: np.ndarray
    ) -> GridMap25D:
        """
        Builds GridMap25D from PointCloudFrame and array predictions.
        """
        return self.grid_builder.build_grid(
            points=frame.points,
            labels=predicted_class,
            confidences=confidence,
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            sequence_id=frame.sequence_id
        )

    def validate_spatial_alignment(self, points: np.ndarray, grid_map: GridMap25D) -> bool:
        """
        Verifies the fundamental Phase-2 spatial invariant:
        For every point p = (x, y, z) inside the operational range:
          1. It maps to exactly one distance band.
          2. It maps to cell (ix, iy) = (floor(x/s), floor(y/s)).
          3. ix * s <= x < (ix + 1) * s
          4. iy * s <= y < (iy + 1) * s
        """
        if points is None or len(points) == 0:
            return True

        for i in range(len(points)):
            x, y, z = float(points[i, 0]), float(points[i, 1]), float(points[i, 2])
            r = math.sqrt(x * x + y * y) if 'math' in globals() else np.sqrt(x * x + y * y)

            if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(z):
                continue
            if r < 0.0 or r >= self.max_range:
                continue

            band = distance_to_band(r, bands=self.bands)
            if band is None:
                continue

            ix, iy = xy_to_cell(x, y, band.voxel_size)
            cell = grid_map.get_cell(band.name, ix, iy)

            # Spatial boundary check
            min_x, max_x, min_y, max_y = cell.bounds
            if not (min_x <= x < max_x + 1e-7 and min_y <= y < max_y + 1e-7):
                return False

        return True
