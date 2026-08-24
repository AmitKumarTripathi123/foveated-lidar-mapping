"""
Canonical End-to-End Foveated Pipeline Orchestrator (SIH PS 26130).
Single entry point executing:
  Input Ingest -> Range Filter -> 3-Zone Foveation -> SPVCNN Inference -> 2.5D Multilayer GridMap
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import yaml

from src.core.lidar_loader import load_lidar_points
from src.core.range_filter import RangeFilter
from src.core.foveated_grid import HierarchicalFoveatedGridEngine, GridMap25D
from ml.data.amit_adapter import FoveatedVoxelSampler
from src.inference.predictor import CanonicalPredictor
from src.inference.postprocess import validate_predictions


@dataclass
class PipelineResult:
    """Standard execution result returned by FoveatedPipeline."""
    grid_map: GridMap25D
    raw_points_count: int
    foveated_points_count: int
    total_latency_ms: float
    stage_latencies_ms: Dict[str, float] = field(default_factory=dict)
    foveated_xyz: Optional[np.ndarray] = None
    predicted_classes: Optional[np.ndarray] = None
    confidences: Optional[np.ndarray] = None


class FoveatedPipeline:
    """The Single Canonical Pipeline for Foveated LiDAR Mapping."""

    def __init__(self, config_path: Union[str, Path] = "configs/system_config.yaml"):
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        # 1. Range Filter
        l_cfg = self.cfg.get("lidar", {})
        self.range_filter = RangeFilter(
            min_range=float(l_cfg.get("min_range", 0.5)),
            max_range=float(l_cfg.get("max_range", 100.0)),
        )

        # 2. Foveated 3-Zone Sampler
        fov = self.cfg.get("foveation", {})
        self.foveated_sampler = FoveatedVoxelSampler(
            near_dist=float(fov.get("near", {}).get("radius", 10.0)),
            near_voxel=float(fov.get("near", {}).get("resolution", 0.05)),
            mid_dist=float(fov.get("mid", {}).get("radius", 40.0)),
            mid_voxel=float(fov.get("mid", {}).get("resolution", 0.15)),
            far_dist=float(fov.get("far", {}).get("radius", 100.0)),
            far_voxel=float(fov.get("far", {}).get("resolution", 0.50)),
        )

        # 3. SPVCNN Predictor
        self.predictor = CanonicalPredictor(self.config_path)

        # 4. Grid Engine
        self.grid_engine = HierarchicalFoveatedGridEngine(config_path=self.config_path)

    def run(self, input_data: Union[str, Path, np.ndarray]) -> PipelineResult:
        """Execute end-to-end perception and 2.5D mapping."""
        t_start = time.perf_counter()
        stage_times = {}

        # 1. Load / Validate Points
        t0 = time.perf_counter()
        if isinstance(input_data, (str, Path)):
            pts = load_lidar_points(input_data)
        elif isinstance(input_data, np.ndarray):
            pts = input_data.astype(np.float32)
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
        stage_times["loading_ms"] = (time.perf_counter() - t0) * 1000.0

        raw_count = len(pts)

        # 2. Range Filter
        t0 = time.perf_counter()
        pts_filtered, _ = self.range_filter.filter(pts)
        stage_times["range_filter_ms"] = (time.perf_counter() - t0) * 1000.0

        # 3. 3-Zone Foveation
        t0 = time.perf_counter()
        fov_pts, _, _ = self.foveated_sampler.sample(pts_filtered)
        stage_times["foveation_ms"] = (time.perf_counter() - t0) * 1000.0

        # 4. SPVCNN Inference
        t0 = time.perf_counter()
        preds, confs = self.predictor.predict(fov_pts)
        stage_times["inference_ms"] = (time.perf_counter() - t0) * 1000.0

        # 5. Grid Compilation
        t0 = time.perf_counter()
        grid = self.grid_engine.build_25d_grid(fov_pts[:, :3], preds, confs)
        stage_times["gridmap_ms"] = (time.perf_counter() - t0) * 1000.0

        total_latency = (time.perf_counter() - t_start) * 1000.0

        return PipelineResult(
            grid_map=grid,
            raw_points_count=raw_count,
            foveated_points_count=len(fov_pts),
            total_latency_ms=round(total_latency, 2),
            stage_latencies_ms={k: round(v, 2) for k, v in stage_times.items()},
            foveated_xyz=fov_pts[:, :3],
            predicted_classes=preds,
            confidences=confs,
        )
