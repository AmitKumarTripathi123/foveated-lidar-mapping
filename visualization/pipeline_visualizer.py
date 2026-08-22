"""
Core Pipeline Visualizer & Intermediate State Extractor.
Executes the real production LiDAR pipeline step-by-step and captures full intermediate state.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import torch

from src.types import PointCloudFrame, SuperClass, CellState, FoveationBand
from src.range_filter import RangeFilter
from src.foveated_grid import FoveatedGrid25D, GridMap25D, DEFAULT_FROZEN_BANDS, xy_to_cell, distance_to_band
from ml.data.dataset import load_point_cloud, load_labels
from ml.data.amit_adapter import FoveatedVoxelSampler
from phase2.dataset import remap_poss_labels
from phase2.models.spvcnn_adapter import SPVCNNInputAdapter, SPVCNNLabelAdapter
from phase2.inference.predictor import Phase2Predictor, SemanticPrediction
from phase2.adapter import MLToMappingAdapter


CLASS_NAMES = {
    0: "Drivable Terrain",
    1: "Non-Drivable Terrain",
    2: "Static Obstacle",
    3: "Dynamic Object",
    255: "Ignore / Outlier"
}

CLASS_HEX_COLORS = {
    0: "#2ca02c",   # Green
    1: "#d62728",   # Crimson Red
    2: "#1f77b4",   # Blue
    3: "#ff7f0e",   # Amber / Orange
    255: "#7f7f7f"  # Gray
}


@dataclass
class PointTrace:
    """Trace record tracking a single LiDAR point across every pipeline transformation."""
    point_index: int
    raw_xyz: Tuple[float, float, float]
    raw_intensity: float
    is_preprocessed: bool
    is_foveated_retained: bool
    foveated_index: Optional[int]
    distance_r: float
    band_name: Optional[str]
    voxel_coords: Optional[Tuple[int, int, int]]
    packed_64bit_key: Optional[int]
    recovered_voxel_coords: Optional[Tuple[int, int, int]]
    predicted_class_id: int
    predicted_class_name: str
    confidence: float
    class_probabilities: List[float]
    grid_cell_ix: Optional[int]
    grid_cell_iy: Optional[int]
    grid_cell_elevation_mean: Optional[float]
    grid_cell_semantic_class: Optional[int]
    grid_cell_traversability: Optional[float]


@dataclass
class PipelineIntermediateState:
    """Holds all intermediate state data across all 11 stages of the pipeline."""
    scan_path: str
    # 1. Raw
    raw_points: np.ndarray
    raw_labels: Optional[np.ndarray]
    raw_bounds: Dict[str, Tuple[float, float]]
    # 2. Preprocess
    preprocessed_points: np.ndarray
    preprocessed_labels: Optional[np.ndarray]
    removed_out_of_range_count: int
    # 3. Foveation
    foveated_points: np.ndarray
    foveated_labels: Optional[np.ndarray]
    foveation_reduction_pct: float
    foveated_mask: np.ndarray
    # 4. Voxelization
    voxel_size: float
    unique_voxel_count: int
    active_voxel_ratio: float
    packed_keys_sample: List[Dict[str, Any]]
    point_to_voxel_idx: np.ndarray
    # 5. SPVCNN Input
    tensor_shape: Tuple[int, ...]
    tensor_device: str
    tensor_dtype: str
    # 6. SPVCNN Prediction
    predicted_classes: np.ndarray
    class_distribution: Dict[str, Dict[str, Any]]
    # 7. Confidence
    confidences: np.ndarray
    class_probabilities: np.ndarray
    confidence_stats: Dict[str, float]
    # 8. GridMap25D
    grid_map: GridMap25D
    grid_occupied_cells: int
    # 9. Profiling
    stage_timings_ms: Dict[str, float]
    total_pipeline_ms: float
    effective_fps: float


class PipelineVisualizer:
    """Orchestrates pipeline execution, intermediate state extraction, and point tracing."""

    def __init__(
        self,
        model_path: str = "checkpoints/best_spvcnn.pt",
        device: str = "cpu",
        voxel_size: float = 0.05
    ):
        self.device = device
        self.voxel_size = voxel_size
        self.range_filter = RangeFilter(max_range=100.0, min_range=0.5)
        self.foveation_sampler = FoveatedVoxelSampler()
        self.input_adapter = SPVCNNInputAdapter(voxel_size=self.voxel_size)
        self.predictor = Phase2Predictor(model_type="spvcnn", model_path=model_path, device=device)
        self.grid_adapter = MLToMappingAdapter(bands=DEFAULT_FROZEN_BANDS)

    def process_scan(self, bin_path: Union[str, Path], profile_internal: bool = False) -> PipelineIntermediateState:
        """Executes the complete pipeline while recording every intermediate state and metric."""
        bin_p = Path(bin_path)
        lbl_p = bin_p.parent.parent / "labels" / f"{bin_p.stem}.label"

        timings = {}
        t_all_start = time.perf_counter()

        # ----------------------------------------------------------------------
        # STAGE 1: RAW LiDAR LOADING
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        raw_pts = load_point_cloud(bin_p)
        raw_lbls = load_labels(lbl_p) if lbl_p.exists() else None
        if raw_lbls is not None:
            raw_lbls = raw_lbls[:len(raw_pts)]
            raw_pts = raw_pts[:len(raw_lbls)]
        timings["1_load_ms"] = (time.perf_counter() - t0) * 1000.0

        raw_bounds = {
            "x": (float(np.min(raw_pts[:, 0])), float(np.max(raw_pts[:, 0]))),
            "y": (float(np.min(raw_pts[:, 1])), float(np.max(raw_pts[:, 1]))),
            "z": (float(np.min(raw_pts[:, 2])), float(np.max(raw_pts[:, 2]))),
        }

        # ----------------------------------------------------------------------
        # STAGE 2: PREPROCESSING (RANGE FILTERING)
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        raw_mapped_lbls = remap_poss_labels(raw_lbls) if raw_lbls is not None else np.full(len(raw_pts), 255, dtype=np.uint32)
        raw_frame = PointCloudFrame(points=raw_pts, labels=raw_mapped_lbls.astype(np.uint32), frame_id=bin_p.stem)
        filtered_frame, filter_metadata = self.range_filter.filter_frame(raw_frame)
        timings["2_preprocess_ms"] = (time.perf_counter() - t0) * 1000.0

        removed_count = len(raw_pts) - len(filtered_frame.points)

        # ----------------------------------------------------------------------
        # STAGE 3: FOVEATION
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        fov_pts, fov_lbls, fov_report = self.foveation_sampler.sample(filtered_frame.points, filtered_frame.labels)
        timings["3_foveation_ms"] = (time.perf_counter() - t0) * 1000.0

        reduction_pct = (1.0 - len(fov_pts) / max(len(filtered_frame.points), 1)) * 100.0

        # ----------------------------------------------------------------------
        # STAGE 4: VOXELIZATION (64-BIT INTEGER HASHING)
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        bundle = self.input_adapter.prepare_input(fov_pts, device=self.device)
        timings["4_voxelization_ms"] = (time.perf_counter() - t0) * 1000.0

        unique_voxel_count = bundle["num_voxels"]
        active_ratio = (unique_voxel_count / max(len(fov_pts), 1)) * 100.0

        # Inspect 64-bit coordinate packing transformations on a representative sample
        packed_samples = []
        xyz_sample = fov_pts[:min(5, len(fov_pts)), :3]
        OFFSET = 50000
        for pt_i in range(len(xyz_sample)):
            px, py, pz = xyz_sample[pt_i]
            ix = int(np.floor(px / self.voxel_size))
            iy = int(np.floor(py / self.voxel_size))
            iz = int(np.floor(pz / self.voxel_size))
            packed_key = int(((ix + OFFSET) << 42) | ((iy + OFFSET) << 21) | (iz + OFFSET))
            rec_ix = (packed_key >> 42) - OFFSET
            rec_iy = ((packed_key >> 21) & 0x1FFFFF) - OFFSET
            rec_iz = (packed_key & 0x1FFFFF) - OFFSET
            packed_samples.append({
                "point_idx": pt_i,
                "xyz": (round(float(px), 3), round(float(py), 3), round(float(pz), 3)),
                "original_voxel": (ix, iy, iz),
                "packed_64bit_key": hex(packed_key),
                "recovered_voxel": (rec_ix, rec_iy, rec_iz),
                "verified": (ix, iy, iz) == (rec_ix, rec_iy, rec_iz)
            })

        # ----------------------------------------------------------------------
        # STAGE 5 & 6 & 7: SPVCNN INFERENCE, PREDICTIONS & CONFIDENCES
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        fov_frame = PointCloudFrame(points=fov_pts, labels=fov_lbls, frame_id=bin_p.stem)
        prediction = self.predictor.predict_frame(fov_frame)
        timings["5_spvcnn_inference_ms"] = (time.perf_counter() - t0) * 1000.0

        pred_classes = prediction.predicted_class
        confidences = prediction.confidence
        probabilities = prediction.class_probabilities

        # Class distribution metrics
        class_distribution = {}
        total_p = len(pred_classes)
        for c_id, c_name in CLASS_NAMES.items():
            cnt = int(np.sum(pred_classes == c_id))
            pct = (cnt / max(total_p, 1)) * 100.0
            class_distribution[c_name] = {
                "class_id": c_id,
                "count": cnt,
                "percentage": round(pct, 2),
                "color": CLASS_HEX_COLORS.get(c_id, "#000000")
            }

        confidence_stats = {
            "min": round(float(np.min(confidences)), 4) if len(confidences) > 0 else 0.0,
            "mean": round(float(np.mean(confidences)), 4) if len(confidences) > 0 else 0.0,
            "median": round(float(np.median(confidences)), 4) if len(confidences) > 0 else 0.0,
            "max": round(float(np.max(confidences)), 4) if len(confidences) > 0 else 0.0,
        }

        # ----------------------------------------------------------------------
        # STAGE 8: 2.5D GRID GENERATION (GridMap25D)
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        grid_map = self.grid_adapter.prediction_to_grid(prediction)
        timings["6_grid_generation_ms"] = (time.perf_counter() - t0) * 1000.0

        total_pipeline_ms = sum(timings.values())
        effective_fps = 1000.0 / max(total_pipeline_ms, 1e-4)

        return PipelineIntermediateState(
            scan_path=str(bin_p),
            raw_points=raw_pts,
            raw_labels=raw_lbls,
            raw_bounds=raw_bounds,
            preprocessed_points=filtered_frame.points,
            preprocessed_labels=filtered_frame.labels,
            removed_out_of_range_count=removed_count,
            foveated_points=fov_pts,
            foveated_labels=fov_lbls,
            foveation_reduction_pct=reduction_pct,
            foveated_mask=np.ones(len(fov_pts), dtype=bool),
            voxel_size=self.voxel_size,
            unique_voxel_count=unique_voxel_count,
            active_voxel_ratio=active_ratio,
            packed_keys_sample=packed_samples,
            point_to_voxel_idx=bundle["point_to_voxel_idx"].cpu().numpy(),
            tensor_shape=tuple(bundle["features"].shape),
            tensor_device=str(bundle["features"].device),
            tensor_dtype=str(bundle["features"].dtype),
            predicted_classes=pred_classes,
            class_distribution=class_distribution,
            confidences=confidences,
            class_probabilities=probabilities,
            confidence_stats=confidence_stats,
            grid_map=grid_map,
            grid_occupied_cells=grid_map.num_occupied_cells,
            stage_timings_ms=timings,
            total_pipeline_ms=round(total_pipeline_ms, 2),
            effective_fps=round(effective_fps, 2)
        )

    def trace_point(self, state: PipelineIntermediateState, point_idx: int) -> PointTrace:
        """Traces the complete transformation lifecycle of a single point index."""
        N_raw = len(state.raw_points)
        idx = max(0, min(point_idx, N_raw - 1))

        raw_pt = state.raw_points[idx]
        rx, ry, rz = float(raw_pt[0]), float(raw_pt[1]), float(raw_pt[2])
        r_int = float(raw_pt[3]) if len(raw_pt) > 3 else 0.0
        dist_r = float(np.sqrt(rx * rx + ry * ry))

        # Range filter check
        is_preprocessed = 0.5 <= dist_r < 100.0

        # Find closest point in foveated frame
        dists = np.sum((state.foveated_points[:, :3] - np.array([rx, ry, rz])) ** 2, axis=1)
        fov_idx = int(np.argmin(dists)) if len(dists) > 0 else None
        min_d = float(np.sqrt(dists[fov_idx])) if fov_idx is not None else 999.0
        is_retained = min_d < 1e-4

        # Voxel coords
        ix = int(np.floor(rx / state.voxel_size))
        iy = int(np.floor(ry / state.voxel_size))
        iz = int(np.floor(rz / state.voxel_size))
        OFFSET = 50000
        packed_key = int(((ix + OFFSET) << 42) | ((iy + OFFSET) << 21) | (iz + OFFSET))
        rec_ix = (packed_key >> 42) - OFFSET
        rec_iy = ((packed_key >> 21) & 0x1FFFFF) - OFFSET
        rec_iz = (packed_key & 0x1FFFFF) - OFFSET

        # Band
        band = distance_to_band(dist_r, bands=DEFAULT_FROZEN_BANDS)
        band_name = band.name if band else "out_of_range"

        # Predictions
        if is_retained and fov_idx is not None:
            pred_c = int(state.predicted_classes[fov_idx])
            conf = float(state.confidences[fov_idx])
            probs = state.class_probabilities[fov_idx].tolist() if len(state.class_probabilities) > 0 else [0, 0, 0, 0]
        else:
            pred_c = SuperClass.IGNORE_LABEL
            conf = 0.0
            probs = [0.0, 0.0, 0.0, 0.0]

        # Grid Cell
        g_ix = int(np.floor(rx / (band.voxel_size if band else 0.05)))
        g_iy = int(np.floor(ry / (band.voxel_size if band else 0.05)))
        cell = state.grid_map.get_cell(band_name, g_ix, g_iy) if band else None

        return PointTrace(
            point_index=idx,
            raw_xyz=(rx, ry, rz),
            raw_intensity=r_int,
            is_preprocessed=is_preprocessed,
            is_foveated_retained=is_retained,
            foveated_index=fov_idx if is_retained else None,
            distance_r=round(dist_r, 3),
            band_name=band_name,
            voxel_coords=(ix, iy, iz),
            packed_64bit_key=packed_key,
            recovered_voxel_coords=(rec_ix, rec_iy, rec_iz),
            predicted_class_id=pred_c,
            predicted_class_name=CLASS_NAMES.get(pred_c, "Unknown"),
            confidence=round(conf, 4),
            class_probabilities=[round(p, 4) for p in probs],
            grid_cell_ix=g_ix,
            grid_cell_iy=g_iy,
            grid_cell_elevation_mean=round(cell.elevation_mean, 3) if cell and not np.isnan(cell.elevation_mean) else None,
            grid_cell_semantic_class=cell.semantic_class if cell else None,
            grid_cell_traversability=round(cell.traversability, 2) if cell else None
        )
