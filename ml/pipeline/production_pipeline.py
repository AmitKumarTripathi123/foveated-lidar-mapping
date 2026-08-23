"""
Hardened Production Perception & 2.5D Mapping Pipeline (Phase 15.7).
Provides safe configuration loading, cryptographic checkpoint verification,
robust input sanitizer, and end-to-end 10 Hz real-time inference execution.
"""

import hashlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from ml.data.amit_adapter import FoveatedVoxelSampler
from ml.data.spvcnn_adapter import SPVCNNInputAdapter
from ml.models.spvcnn import SPVCNN, build_spvcnn
from ml.models.mapping_adapter import MLToMappingAdapter, GridMap25D, PredictionBatch


class ChecksumMismatchError(ValueError):
    """Raised when checkpoint SHA256 hash does not match production manifest."""
    pass


class ConfigurationError(ValueError):
    """Raised when pipeline configuration fails schema validation."""
    pass


class InputValidationError(ValueError):
    """Raised when input point cloud data violates integrity bounds."""
    pass


def verify_file_sha256(file_path: Path, expected_hash: str) -> bool:
    """Calculate and compare SHA256 checksum."""
    if not file_path.is_file():
        return False
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    actual_hash = sha.hexdigest().lower()
    return actual_hash == expected_hash.lower()


@dataclass
class FrameProcessingResult:
    """Standardized output container for a processed LiDAR frame."""
    frame_id: str
    success: bool
    num_input_points: int
    num_foveated_points: int
    prediction_dto: Optional[PredictionBatch]
    grid_map: Optional[GridMap25D]
    latency_ms: float
    stage_latencies_ms: Dict[str, float]
    error_message: Optional[str] = None


class ProductionPipeline:
    """Hardened real-time production perception and mapping pipeline."""

    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self._validate_and_initialize()

    def _validate_and_initialize(self):
        """Validate configuration schema and initialize hardened components."""
        # 1. Validate Config Schema
        if "checkpoint" not in self.config or "path" not in self.config["checkpoint"]:
            raise ConfigurationError("Missing 'checkpoint.path' in configuration!")
        if "expected_sha256" not in self.config["checkpoint"]:
            raise ConfigurationError("Missing 'checkpoint.expected_sha256' in configuration!")

        ckpt_cfg = self.config["checkpoint"]
        ckpt_path = Path(ckpt_cfg["path"])
        if not ckpt_path.is_absolute():
            ckpt_path = repo_root / ckpt_path

        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Production checkpoint missing: {ckpt_path}")

        expected_hash = ckpt_cfg["expected_sha256"]
        if not verify_file_sha256(ckpt_path, expected_hash):
            raise ChecksumMismatchError(f"CRITICAL: Checkpoint SHA256 checksum mismatch for {ckpt_path}!")

        # 2. Setup Hardware Device
        req_dev = ckpt_cfg.get("device", "cuda")
        if req_dev == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA requested but unavailable. Falling back to CPU.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(req_dev)

        if self.device.type == "cuda" and ckpt_cfg.get("enable_tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # 3. Instantiate SPVCNN Model
        model_cfg = self.config.get("model", {})
        in_channels = model_cfg.get("in_channels", 4)
        num_classes = model_cfg.get("num_classes", 4)

        self.model = build_spvcnn(
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained_path=str(ckpt_path),
            device=self.device,
        )
        self.model.eval()

        # 4. Instantiate Adapters
        fov_cfg = self.config.get("foveation", {})
        near_v = fov_cfg.get("near_zone", {}).get("voxel_size_m", 0.05)
        near_d = fov_cfg.get("near_zone", {}).get("max_range_m", 10.0)
        mid_v = fov_cfg.get("mid_zone", {}).get("voxel_size_m", 0.15)
        mid_d = fov_cfg.get("mid_zone", {}).get("max_range_m", 40.0)
        far_v = fov_cfg.get("far_zone", {}).get("voxel_size_m", 0.50)
        far_d = fov_cfg.get("far_zone", {}).get("max_range_m", 100.0)

        self.sampler = FoveatedVoxelSampler(
            near_dist=near_d, near_voxel=near_v,
            mid_dist=mid_d, mid_voxel=mid_v,
            far_dist=far_d, far_voxel=far_v,
        )
        self.input_adapter = SPVCNNInputAdapter(voxel_size=model_cfg.get("voxel_size", 0.05))

        map_cfg = self.config.get("mapping", {})
        self.map_adapter = MLToMappingAdapter(
            bounds_x=tuple(map_cfg.get("bounds_x", (-50.0, 50.0))),
            bounds_y=tuple(map_cfg.get("bounds_y", (-50.0, 50.0))),
            resolution=map_cfg.get("resolution_m", 0.20),
            num_classes=num_classes,
            ignore_index=map_cfg.get("ignore_index", 255),
        )

    def validate_raw_points(self, raw_points: Any) -> np.ndarray:
        """Strictly validate and sanitize incoming LiDAR point cloud buffer."""
        if raw_points is None:
            raise InputValidationError("Input points cannot be None!")

        if not isinstance(raw_points, np.ndarray):
            raw_points = np.asarray(raw_points, dtype=np.float32)

        if raw_points.size == 0 or raw_points.ndim != 2:
            raise InputValidationError(f"Invalid point cloud shape: {raw_points.shape}")

        if raw_points.shape[1] < 3:
            raise InputValidationError(f"Point cloud must have at least 3 channels (XYZ), got {raw_points.shape[1]}")

        # Add intensity channel if missing
        if raw_points.shape[1] == 3:
            intensity = np.zeros((raw_points.shape[0], 1), dtype=np.float32)
            raw_points = np.hstack([raw_points, intensity])

        # Filter NaNs and Infs defensively
        finite_mask = np.isfinite(raw_points[:, 0]) & np.isfinite(raw_points[:, 1]) & np.isfinite(raw_points[:, 2])
        clean_points = raw_points[finite_mask]

        if clean_points.shape[0] == 0:
            raise InputValidationError("No finite points remaining after filtering!")

        return clean_points

    def process_frame(self, raw_points: Any, frame_id: str = "frame_000000") -> FrameProcessingResult:
        """Execute end-to-end perception and 2.5D mapping with timing telemetry."""
        t_start = time.perf_counter()
        stage_times = {}

        try:
            # 1. Input Sanitization
            t0 = time.perf_counter()
            pts = self.validate_raw_points(raw_points)
            stage_times["input_validation_ms"] = (time.perf_counter() - t0) * 1000.0

            # 2. 3-Zone Distance Foveation
            t0 = time.perf_counter()
            fov_pts, _, _ = self.sampler.sample(pts)
            stage_times["3zone_foveation_ms"] = (time.perf_counter() - t0) * 1000.0

            if len(fov_pts) == 0:
                raise InputValidationError("No points remained after 3-zone distance foveation!")

            # 3. SPVCNN Voxelization Preprocessing
            t0 = time.perf_counter()
            pts_tensor = torch.from_numpy(fov_pts).to(self.device).float()
            bundle = self.input_adapter.prepare_input(pts_tensor, device=self.device)
            stage_times["voxelization_ms"] = (time.perf_counter() - t0) * 1000.0

            if bundle["num_voxels"] == 0:
                raise InputValidationError("Zero voxels generated during spatial quantization!")

            # 4. SPVCNN CUDA Inference
            t0 = time.perf_counter()
            with torch.inference_mode():
                logits = self.model(
                    features=bundle["features"],
                    point_to_voxel_idx=bundle["point_to_voxel_idx"],
                    num_voxels=bundle["num_voxels"],
                )
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
            stage_times["cuda_inference_ms"] = (time.perf_counter() - t0) * 1000.0

            # 5. Output Prediction Contract
            t0 = time.perf_counter()
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1).cpu().numpy().astype(np.int64)
            confs = torch.max(probs, dim=-1).values.cpu().numpy().astype(np.float32)
            dto = PredictionBatch(
                xyz=fov_pts[:, :3],
                predicted_class=preds,
                confidence=confs,
            )
            validated_dto = self.map_adapter.validate_prediction(dto)
            stage_times["output_contract_ms"] = (time.perf_counter() - t0) * 1000.0

            # 6. Vectorized GridMap25D Generation
            t0 = time.perf_counter()
            grid = self.map_adapter.build_25d_grid(validated_dto)
            stage_times["gridmap25d_ms"] = (time.perf_counter() - t0) * 1000.0

            tot_lat = (time.perf_counter() - t_start) * 1000.0
            return FrameProcessingResult(
                frame_id=frame_id,
                success=True,
                num_input_points=len(raw_points),
                num_foveated_points=len(fov_pts),
                prediction_dto=validated_dto,
                grid_map=grid,
                latency_ms=round(tot_lat, 2),
                stage_latencies_ms=stage_times,
                error_message=None,
            )

        except Exception as e:
            tot_lat = (time.perf_counter() - t_start) * 1000.0
            return FrameProcessingResult(
                frame_id=frame_id,
                success=False,
                num_input_points=len(raw_points) if hasattr(raw_points, "__len__") else 0,
                num_foveated_points=0,
                prediction_dto=None,
                grid_map=None,
                latency_ms=round(tot_lat, 2),
                stage_latencies_ms=stage_times,
                error_message=str(e),
            )
