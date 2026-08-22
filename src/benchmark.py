"""
Performance Benchmarking Module.
Profiles latency per stage, FPS, point reduction %, RAM/CPU usage,
and compares candidate foveation configurations against uniform voxel and uncompressed baselines.
"""

import time
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.types import PointCloudFrame, AggregationPolicy, FoveationBand
from src.validator import PointCloudValidator
from src.label_mapper import LabelMapper
from src.range_filter import RangeFilter
from src.foveation import FoveatedVoxelizer
from src.metrics.elevation_preservation import ElevationPreservationValidator
from src.metrics.obstacle_preservation import ObstaclePreservationValidator
from src.metrics.semantic_preservation import SemanticPreservationValidator


@dataclass
class StageTimingStats:
    stage_name: str
    mean_ms: float
    median_ms: float
    p95_ms: float
    std_ms: float
    min_ms: float
    max_ms: float


@dataclass
class ConfigurationBenchmarkResult:
    config_name: str
    config_type: str  # 'foveated', 'uniform', 'no_foveation'
    description: str
    num_frames_tested: int
    raw_points_mean: float
    filtered_points_mean: float
    foveated_points_mean: float
    point_reduction_percent: float
    compression_ratio: float
    total_latency_mean_ms: float
    total_latency_median_ms: float
    total_latency_p95_ms: float
    total_latency_std_ms: float
    fps_mean: float
    fps_p95: float
    stage_timings: Dict[str, StageTimingStats]
    elevation_rmse_m: float
    elevation_mae_m: float
    obstacle_recall_percent: float
    dynamic_object_survival_percent: float
    ram_usage_mb: float


class PerformanceBenchmark:
    """
    Runs rigorous benchmarking across multiple configurations and repetitions.
    """

    def __init__(self, max_range: float = 100.0):
        self.max_range = max_range
        self.validator = PointCloudValidator(max_allowed_range=max_range)
        self.range_filter = RangeFilter(min_range=0.0, max_range=max_range)
        self.label_mapper = LabelMapper()
        self.elevation_validator = ElevationPreservationValidator(grid_resolution=0.20, max_range=max_range)
        self.obstacle_validator = ObstaclePreservationValidator(grid_resolution=0.25, max_range=max_range)
        self.semantic_validator = SemanticPreservationValidator()

    @staticmethod
    def _get_process_memory_mb() -> float:
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            # On macOS, ru_maxrss is in bytes
            return float(usage.ru_maxrss) / (1024.0 * 1024.0)
        except Exception:
            return 0.0

    @staticmethod
    def _calc_timing_stats(timings: List[float], name: str) -> StageTimingStats:
        arr = np.array(timings)
        return StageTimingStats(
            stage_name=name,
            mean_ms=round(float(np.mean(arr)), 3),
            median_ms=round(float(np.median(arr)), 3),
            p95_ms=round(float(np.percentile(arr, 95)), 3),
            std_ms=round(float(np.std(arr)), 3),
            min_ms=round(float(np.min(arr)), 3),
            max_ms=round(float(np.max(arr)), 3)
        )

    def benchmark_pipeline_on_frames(
        self,
        frames: List[PointCloudFrame],
        config_name: str,
        config_dict: Dict[str, Any],
        repeats_per_frame: int = 5,
        policy: AggregationPolicy = AggregationPolicy.OBSTACLE_PRESERVING
    ) -> ConfigurationBenchmarkResult:
        """
        Runs end-to-end pipeline benchmark on given frames.
        """
        cfg_type = config_dict.get("type", "foveated")
        desc = config_dict.get("description", config_name)

        if cfg_type == "foveated":
            bands = [
                FoveationBand(
                    name=b.get("name", f"band_{b['min_range']}_{b['max_range']}"),
                    min_range=float(b["min_range"]),
                    max_range=float(b["max_range"]),
                    voxel_size=float(b["voxel_size"])
                ) for b in config_dict.get("bands", [])
            ]
            voxelizer = FoveatedVoxelizer(bands=bands, default_policy=policy, max_range=self.max_range)
        elif cfg_type == "uniform":
            v_size = float(config_dict.get("voxel_size", 0.15))
            voxelizer = FoveatedVoxelizer(default_policy=policy, max_range=self.max_range)
        else:
            # No foveation
            voxelizer = None

        val_times: List[float] = []
        map_times: List[float] = []
        filter_times: List[float] = []
        fov_times: List[float] = []
        total_times: List[float] = []

        raw_counts: List[int] = []
        filtered_counts: List[int] = []
        fov_counts: List[int] = []

        elevation_rmses: List[float] = []
        elevation_maes: List[float] = []
        obstacle_recalls: List[float] = []
        dynamic_survivals: List[float] = []

        for frame in frames:
            for _ in range(repeats_per_frame):
                t0 = time.perf_counter()

                # 1. Validation
                t_v0 = time.perf_counter()
                val_summary = self.validator.validate_frame(frame)
                t_v1 = time.perf_counter()

                # 2. Label Mapping
                t_m0 = time.perf_counter()
                mapped_frame = self.label_mapper.map_frame(frame)
                t_m1 = time.perf_counter()

                # 3. Range Filtering
                t_f0 = time.perf_counter()
                filtered_frame, filter_rep = self.range_filter.filter_frame(mapped_frame)
                t_f1 = time.perf_counter()

                # 4. Foveation
                t_fov0 = time.perf_counter()
                if cfg_type == "foveated":
                    fov_result = voxelizer.voxelize(filtered_frame, policy=policy, config_name=config_name)
                    fov_frame = fov_result.foveated_frame
                elif cfg_type == "uniform":
                    fov_result = voxelizer.uniform_voxelize(filtered_frame, voxel_size=v_size, policy=policy)
                    fov_frame = fov_result.foveated_frame
                else:
                    fov_frame = filtered_frame
                    fov_result = None
                t_fov1 = time.perf_counter()

                t_end = time.perf_counter()

                v_ms = (t_v1 - t_v0) * 1000.0
                m_ms = (t_m1 - t_m0) * 1000.0
                f_ms = (t_f1 - t_f0) * 1000.0
                fov_ms = (t_fov1 - t_fov0) * 1000.0
                tot_ms = (t_end - t0) * 1000.0

                val_times.append(v_ms)
                map_times.append(m_ms)
                filter_times.append(f_ms)
                fov_times.append(fov_ms)
                total_times.append(tot_ms)

            # Record points and preservation once per frame
            raw_counts.append(frame.num_points)
            filtered_counts.append(filtered_frame.num_points)
            fov_counts.append(fov_frame.num_points)

            # Metrics
            elev_rep = self.elevation_validator.evaluate(filtered_frame, fov_frame)
            obs_rep = self.obstacle_validator.evaluate(filtered_frame, fov_frame)

            elevation_rmses.append(elev_rep.overall_rmse)
            elevation_maes.append(elev_rep.overall_mae)
            obstacle_recalls.append(obs_rep.obstacle_grid_recall)
            dynamic_survivals.append(obs_rep.far_field_dynamic_survival_rate)

        raw_mean = float(np.mean(raw_counts)) if raw_counts else 0.0
        filt_mean = float(np.mean(filtered_counts)) if filtered_counts else 0.0
        fov_mean = float(np.mean(fov_counts)) if fov_counts else 0.0

        pt_red = ((raw_mean - fov_mean) / raw_mean) * 100.0 if raw_mean > 0 else 0.0
        comp_rat = (raw_mean / max(fov_mean, 1))

        tot_arr = np.array(total_times)
        mean_tot = float(np.mean(tot_arr))
        p95_tot = float(np.percentile(tot_arr, 95))
        fps_mean = 1000.0 / mean_tot if mean_tot > 0 else 0.0
        fps_p95 = 1000.0 / p95_tot if p95_tot > 0 else 0.0

        stage_timings = {
            "validation": self._calc_timing_stats(val_times, "validation"),
            "label_mapping": self._calc_timing_stats(map_times, "label_mapping"),
            "range_filter": self._calc_timing_stats(filter_times, "range_filter"),
            "foveation": self._calc_timing_stats(fov_times, "foveation"),
            "total_pipeline": self._calc_timing_stats(total_times, "total_pipeline"),
        }

        return ConfigurationBenchmarkResult(
            config_name=config_name,
            config_type=cfg_type,
            description=desc,
            num_frames_tested=len(frames),
            raw_points_mean=round(raw_mean, 1),
            filtered_points_mean=round(filt_mean, 1),
            foveated_points_mean=round(fov_mean, 1),
            point_reduction_percent=round(pt_red, 2),
            compression_ratio=round(comp_rat, 2),
            total_latency_mean_ms=round(mean_tot, 2),
            total_latency_median_ms=round(float(np.median(tot_arr)), 2),
            total_latency_p95_ms=round(p95_tot, 2),
            total_latency_std_ms=round(float(np.std(tot_arr)), 2),
            fps_mean=round(fps_mean, 1),
            fps_p95=round(fps_p95, 1),
            stage_timings=stage_timings,
            elevation_rmse_m=round(float(np.mean(elevation_rmses)), 4) if elevation_rmses else 0.0,
            elevation_mae_m=round(float(np.mean(elevation_maes)), 4) if elevation_maes else 0.0,
            obstacle_recall_percent=round(float(np.mean(obstacle_recalls)), 2) if obstacle_recalls else 100.0,
            dynamic_object_survival_percent=round(float(np.mean(dynamic_survivals)), 2) if dynamic_survivals else 100.0,
            ram_usage_mb=round(self._get_process_memory_mb(), 2)
        )
