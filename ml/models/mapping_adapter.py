"""ML -> 2.5D Mapping Adapter and Contract Translator (Phase 6 Foundation).

Connects Atul''s PointNet++ Semantic Segmentation output contract with
Amit''s 2.5D Elevation & Semantic Occupancy Grid Mapping System.

Architecture:
  Raw / Foveated Points [N, 4]
              │
              ▼
      PointNet2Predictor
              │
              ▼
   PredictionBatch [x, y, z, predicted_class, confidence]
              │
              ▼
     MLToMappingAdapter (Validation & Projection)
              │
              ▼
      Semantic25DGrid (Elevation, Semantics, Traversability)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class PredictionBatch:
    """Standardized Prediction Data Transfer Object (DTO) for ML -> Mapping."""
    xyz: np.ndarray             # (N, 3) float32 coordinates in sensor frame
    predicted_class: np.ndarray # (N,) int64 class indices strictly in {0, 1, 2, 3}
    confidence: np.ndarray      # (N,) float32 confidence scores strictly in [0.0, 1.0]
    intensity: Optional[np.ndarray] = None  # Optional (N, 1) float32 intensity


@dataclass
class GridMap25D:
    """2.5D Elevation and Semantic Grid Map representation."""
    bounds_x: Tuple[float, float]
    bounds_y: Tuple[float, float]
    resolution: float
    grid_shape: Tuple[int, int]
    elevation_min: np.ndarray      # (H, W) float32 min elevation (NaN if unobserved)
    elevation_max: np.ndarray      # (H, W) float32 max elevation (NaN if unobserved)
    elevation_mean: np.ndarray     # (H, W) float32 mean elevation (NaN if unobserved)
    semantic_layer: np.ndarray     # (H, W) int64 dominant semantic class (255 = unobserved)
    confidence_layer: np.ndarray   # (H, W) float32 average confidence (0.0 if unobserved)
    traversability_layer: np.ndarray # (H, W) float32 (-1.0=unobserved, 1.0=drivable, 0.0=blocked)
    point_count_layer: np.ndarray  # (H, W) int32 number of points per cell


class MLToMappingAdapter:
    """Validates and translates ML predictions into 2.5D grid mapping layers."""

    def __init__(
        self,
        bounds_x: Tuple[float, float] = (-50.0, 50.0),
        bounds_y: Tuple[float, float] = (-50.0, 50.0),
        resolution: float = 0.20,
        num_classes: int = 4,
        ignore_index: int = 255,
    ):
        """Initialize 2.5D mapping adapter.

        Args:
            bounds_x: (min_x, max_x) in meters.
            bounds_y: (min_y, max_y) in meters.
            resolution: Grid cell edge size in meters (default: 0.20m = 20cm).
            num_classes: Number of valid target classes (default: 4).
            ignore_index: Unobserved / ignore class index (default: 255).
        """
        self.bounds_x = bounds_x
        self.bounds_y = bounds_y
        self.resolution = resolution
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self.width = int(np.ceil((bounds_x[1] - bounds_x[0]) / resolution))
        self.height = int(np.ceil((bounds_y[1] - bounds_y[0]) / resolution))

    def validate_prediction(self, prediction: Union[PredictionBatch, Dict[str, np.ndarray]]) -> PredictionBatch:
        """Strictly validate incoming ML prediction batch contract.

        Args:
            prediction: PredictionBatch DTO or dict with 'xyz', 'predicted_class', 'confidence'.

        Returns:
            Validated PredictionBatch.

        Raises:
            ValueError: If shapes mismatch, NaN/Inf detected, or values out of bounds.
        """
        if isinstance(prediction, dict):
            if "xyz" not in prediction or "predicted_class" not in prediction or "confidence" not in prediction:
                raise ValueError("Prediction dictionary missing required keys: 'xyz', 'predicted_class', 'confidence'")
            batch = PredictionBatch(
                xyz=np.asarray(prediction["xyz"], dtype=np.float32),
                predicted_class=np.asarray(prediction["predicted_class"], dtype=np.int64),
                confidence=np.asarray(prediction["confidence"], dtype=np.float32),
                intensity=prediction.get("intensity"),
            )
        elif isinstance(prediction, PredictionBatch):
            batch = prediction
        else:
            raise TypeError(f"Unsupported prediction type: {type(prediction)}")

        xyz = batch.xyz
        pred_cls = batch.predicted_class
        conf = batch.confidence

        # 1. Dimensionality checks
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"XYZ coordinates must have shape (N, 3), got {xyz.shape}")
        if pred_cls.ndim != 1 or pred_cls.shape[0] != xyz.shape[0]:
            raise ValueError(f"predicted_class shape ({pred_cls.shape}) does not match points count ({xyz.shape[0]})")
        if conf.ndim != 1 or conf.shape[0] != xyz.shape[0]:
            raise ValueError(f"confidence shape ({conf.shape}) does not match points count ({xyz.shape[0]})")

        # 2. NaN / Inf checks
        if np.isnan(xyz).any() or np.isinf(xyz).any():
            raise ValueError("NaN or Inf detected in point coordinates!")
        if np.isnan(conf).any() or np.isinf(conf).any():
            raise ValueError("NaN or Inf detected in confidence scores!")

        # 3. Fast class value range checks strictly in [0, num_classes-1]
        c_min = int(pred_cls.min())
        c_max = int(pred_cls.max())
        if c_min < 0 or c_max >= self.num_classes:
            raise ValueError(f"Predicted class values out of bounds: [{c_min}, {c_max}] outside [0, {self.num_classes-1}]")

        # 4. Confidence range checks strictly in [0.0, 1.0]
        if (conf < 0.0).any() or (conf > 1.0).any():
            raise ValueError(f"Confidence values out of range [0.0, 1.0]: min={conf.min()}, max={conf.max()}")

        return batch

    def build_25d_grid(self, prediction: Union[PredictionBatch, Dict[str, np.ndarray]]) -> GridMap25D:
        """Project validated 3D semantic points into 2.5D elevation and semantic layers.

        Args:
            prediction: PredictionBatch or dict matching the frozen ML contract.

        Returns:
            GridMap25D: Fully populated 2.5D grid layers.
        """
        batch = self.validate_prediction(prediction)
        xyz = batch.xyz
        pred_cls = batch.predicted_class
        conf = batch.confidence
        n_pts = xyz.shape[0]

        # Allocate grid layers
        grid_shape = (self.height, self.width)
        elev_min = np.full(grid_shape, np.nan, dtype=np.float32)
        elev_max = np.full(grid_shape, np.nan, dtype=np.float32)
        elev_sum = np.zeros(grid_shape, dtype=np.float32)
        conf_sum = np.zeros(grid_shape, dtype=np.float32)
        pt_count = np.zeros(grid_shape, dtype=np.int32)
        class_votes = np.zeros((self.height, self.width, self.num_classes), dtype=np.int32)

        if n_pts == 0:
            return GridMap25D(
                bounds_x=self.bounds_x,
                bounds_y=self.bounds_y,
                resolution=self.resolution,
                grid_shape=grid_shape,
                elevation_min=elev_min,
                elevation_max=elev_max,
                elevation_mean=elev_min.copy(),
                semantic_layer=np.full(grid_shape, self.ignore_index, dtype=np.int64),
                confidence_layer=np.zeros(grid_shape, dtype=np.float32),
                traversability_layer=np.full(grid_shape, -1.0, dtype=np.float32),
                point_count_layer=pt_count,
            )

        # Spatial filter for grid bounding box
        in_x = (xyz[:, 0] >= self.bounds_x[0]) & (xyz[:, 0] < self.bounds_x[1])
        in_y = (xyz[:, 1] >= self.bounds_y[0]) & (xyz[:, 1] < self.bounds_y[1])
        valid_mask = in_x & in_y

        v_xyz = xyz[valid_mask]
        v_cls = pred_cls[valid_mask]
        v_conf = conf[valid_mask]

        if v_xyz.shape[0] > 0:
            grid_c = np.floor((v_xyz[:, 0] - self.bounds_x[0]) / self.resolution).astype(np.int64)
            grid_r = np.floor((v_xyz[:, 1] - self.bounds_y[0]) / self.resolution).astype(np.int64)

            # Clamp indices
            grid_c = np.clip(grid_c, 0, self.width - 1)
            grid_r = np.clip(grid_r, 0, self.height - 1)

            cell_idx = grid_r * self.width + grid_c
            total_cells = self.height * self.width
            z = v_xyz[:, 2]

            # 1. Point counts per cell
            pt_count_1d = np.bincount(cell_idx, minlength=total_cells)
            obs_1d = pt_count_1d > 0
            pt_count = pt_count_1d.reshape(grid_shape).astype(np.int32)

            # 2. Elevation Min / Max (vectorized in-place reduction)
            elev_min_1d = np.full(total_cells, np.inf, dtype=np.float32)
            elev_max_1d = np.full(total_cells, -np.inf, dtype=np.float32)
            np.minimum.at(elev_min_1d, cell_idx, z)
            np.maximum.at(elev_max_1d, cell_idx, z)
            elev_min_1d[~obs_1d] = np.nan
            elev_max_1d[~obs_1d] = np.nan
            elev_min = elev_min_1d.reshape(grid_shape)
            elev_max = elev_max_1d.reshape(grid_shape)

            # 3. Elevation Sum & Confidence Sum
            elev_sum_1d = np.bincount(cell_idx, weights=z, minlength=total_cells).astype(np.float32)
            conf_sum_1d = np.bincount(cell_idx, weights=v_conf, minlength=total_cells).astype(np.float32)
            elev_sum = elev_sum_1d.reshape(grid_shape)
            conf_sum = conf_sum_1d.reshape(grid_shape)

            # 4. Ultra-fast joint class vote calculation
            vote_keys = cell_idx * self.num_classes + v_cls
            vote_counts = np.bincount(vote_keys, minlength=total_cells * self.num_classes)
            class_votes = vote_counts.reshape((self.height, self.width, self.num_classes))

        # Compute mean elevation and average confidence
        observed_mask = pt_count > 0
        elev_mean = np.full(grid_shape, np.nan, dtype=np.float32)
        elev_mean[observed_mask] = elev_sum[observed_mask] / pt_count[observed_mask]

        conf_layer = np.zeros(grid_shape, dtype=np.float32)
        conf_layer[observed_mask] = conf_sum[observed_mask] / pt_count[observed_mask]

        # Determine dominant semantic class per cell
        sem_layer = np.full(grid_shape, self.ignore_index, dtype=np.int64)
        if np.any(observed_mask):
            dominant_c = np.argmax(class_votes, axis=-1)
            sem_layer[observed_mask] = dominant_c[observed_mask]

        # Traversability layer: 1.0 (drivable), 0.2 (non-drivable), 0.0 (obstacle/dynamic), -1.0 (unobserved)
        traversability = np.full(grid_shape, -1.0, dtype=np.float32)
        if np.any(observed_mask):
            is_drivable = (sem_layer == 0) & observed_mask
            is_non_drivable = (sem_layer == 1) & observed_mask
            is_obstacle = ((sem_layer == 2) | (sem_layer == 3)) & observed_mask

            traversability[is_drivable] = 1.0
            traversability[is_non_drivable] = 0.2
            traversability[is_obstacle] = 0.0

        return GridMap25D(
            bounds_x=self.bounds_x,
            bounds_y=self.bounds_y,
            resolution=self.resolution,
            grid_shape=grid_shape,
            elevation_min=elev_min,
            elevation_max=elev_max,
            elevation_mean=elev_mean,
            semantic_layer=sem_layer,
            confidence_layer=conf_layer,
            traversability_layer=traversability,
            point_count_layer=pt_count,
        )
