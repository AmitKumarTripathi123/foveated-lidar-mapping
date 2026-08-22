"""
Phase 2 Predictor & Output Interface Contract.
SemanticPrediction represents the frozen interface between Phase 2 AI and Phase 3 2.5D Mapping.
Supports polymorphic model architectures:
  1. SPVCNN (Primary Default: Sparse Point-Voxel CNN)
  2. FoveatedPointSegNet (Fallback: Distance-Aware Point MLP)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from phase2.models.point_seg_net import FoveatedPointSegNet
from phase2.models.spvcnn import SPVCNN, build_spvcnn, load_spvcnn_checkpoint
from phase2.models.spvcnn_adapter import SPVCNNInputAdapter, SPVCNNLabelAdapter
from src.types import PointCloudFrame, SuperClass

logger = logging.getLogger(__name__)


@dataclass
class SemanticPrediction:
    """
    Frozen output interface contract for Phase 2 semantic perception.
    Guarantees strict 1:1 point alignment and calibrated probability distributions.
    """
    points: np.ndarray              # float32 [N, 4] -> (x, y, z, intensity)
    predicted_class: np.ndarray     # int64 [N] -> superclass in {0, 1, 2, 3, 255}
    class_probabilities: np.ndarray # float32 [N, 4] -> softmax probabilities over 4 super-classes
    confidence: np.ndarray          # float32 [N] -> max scalar confidence in [0.0, 1.0]
    frame_id: str = "000000"
    timestamp: float = 0.0
    sequence_id: str = "00"
    raw_point_count: int = 0
    foveated_point_count: int = 0
    model_type: str = "spvcnn"

    @property
    def num_points(self) -> int:
        return len(self.points)

    def validate_interface(self) -> bool:
        N = len(self.points)
        assert len(self.predicted_class) == N, f"Class len {len(self.predicted_class)} != points {N}"
        assert self.class_probabilities.shape == (N, 4), f"Prob shape {self.class_probabilities.shape} != ({N}, 4)"
        assert len(self.confidence) == N, f"Confidence len {len(self.confidence)} != points {N}"
        assert (self.confidence >= 0.0).all() and (self.confidence <= 1.0001).all(), "Confidence out of bounds"
        assert np.isin(self.predicted_class, [0, 1, 2, 3, 255]).all(), f"Invalid class IDs {np.unique(self.predicted_class)}"
        return True


class Phase2Predictor:
    """
    Universal Phase 2 Predictor supporting SPVCNN (primary) and FoveatedPointSegNet (fallback).
    """

    def __init__(
        self,
        model_type: str = "spvcnn",
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = "cpu",
        voxel_size: float = 0.05,
        native_source: str = "semantickitti",
        num_classes: int = 19
    ):
        self.model_type = model_type.lower()
        self.voxel_size = float(voxel_size)
        self.native_source = native_source

        # Device selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        if self.model_type in ("spvcnn", "spv_cnn"):
            # Check for trained fine-tuned model first, then pretrained
            if model_path is not None:
                resolved_path = Path(model_path)
            elif Path("checkpoints/best_spvcnn.pt").exists():
                resolved_path = Path("checkpoints/best_spvcnn.pt")
            else:
                resolved_path = Path("checkpoints/spvcnn_pretrained.pt")

            # Auto-detect checkpoint class count
            actual_classes = num_classes
            actual_source = self.native_source
            if resolved_path.exists():
                ckpt = torch.load(str(resolved_path), map_location="cpu")
                state = ckpt.get("model_state_dict", ckpt)
                classifier_weight = state.get("classifier.4.weight", None)
                if classifier_weight is not None:
                    actual_classes = classifier_weight.shape[0]
                    if actual_classes == 4:
                        actual_source = "sih_direct"
                    elif actual_classes == 19:
                        actual_source = "semantickitti"

            self.model = build_spvcnn(
                num_classes=actual_classes,
                in_channels=4,
                pretrained_path=str(resolved_path) if resolved_path.exists() else None,
                device=self.device
            )
            self.input_adapter = SPVCNNInputAdapter(voxel_size=self.voxel_size)
            self.label_adapter = SPVCNNLabelAdapter(native_source=actual_source)
            param_count = sum(p.numel() for p in self.model.parameters())
            self.model_info = {
                "model_type": "SPVCNN",
                "device": str(self.device),
                "parameters": param_count,
                "native_classes": actual_classes,
                "target_classes": 4,
                "checkpoint": str(resolved_path) if resolved_path.exists() else "None (Uninitialized)"
            }

        else:
            # Fallback: FoveatedPointSegNet
            resolved_path = model_path if model_path is not None else "checkpoints/best_model.pth"
            self.model = FoveatedPointSegNet()
            if resolved_path and Path(resolved_path).exists():
                ckpt = torch.load(str(resolved_path), map_location=self.device)
                self.model.load_state_dict(ckpt.get("model_state_dict", ckpt))
            self.model.to(self.device)
            param_count = sum(p.numel() for p in self.model.parameters())
            self.model_info = {
                "model_type": "FoveatedPointSegNet",
                "device": str(self.device),
                "parameters": param_count,
                "native_classes": 4,
                "target_classes": 4,
                "checkpoint": str(resolved_path) if Path(resolved_path).exists() else "None (Uninitialized)"
            }

        self.model.eval()

    def predict_frame(self, frame: PointCloudFrame) -> SemanticPrediction:
        """
        Executes semantic prediction on a PointCloudFrame and returns a validated SemanticPrediction.
        """
        pts = frame.points
        if pts is None or len(pts) == 0:
            return SemanticPrediction(
                points=np.empty((0, 4), dtype=np.float32),
                predicted_class=np.empty(0, dtype=np.int64),
                class_probabilities=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty(0, dtype=np.float32),
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                sequence_id=frame.sequence_id,
                model_type=self.model_type
            )

        with torch.inference_mode():
            if self.model_type in ("spvcnn", "spv_cnn"):
                bundle = self.input_adapter.prepare_input(pts, device=self.device)
                logits = self.model(
                    features=bundle["features"],
                    point_to_voxel_idx=bundle["point_to_voxel_idx"],
                    num_voxels=bundle["num_voxels"]
                )
                sih_preds, super_probs, conf = self.label_adapter.process_logits(logits)

                return SemanticPrediction(
                    points=pts,
                    predicted_class=sih_preds,
                    class_probabilities=super_probs,
                    confidence=conf,
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    sequence_id=frame.sequence_id,
                    raw_point_count=frame.metadata.get("raw_point_count", len(pts)),
                    foveated_point_count=len(pts),
                    model_type="SPVCNN"
                )

            else:
                # Fallback: FoveatedPointSegNet
                tensor_pts = torch.from_numpy(pts.astype(np.float32)).to(self.device)
                logits = self.model(tensor_pts)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=-1).astype(np.int64)
                conf = np.max(probs, axis=-1).astype(np.float32)

                return SemanticPrediction(
                    points=pts,
                    predicted_class=preds,
                    class_probabilities=probs,
                    confidence=conf,
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    sequence_id=frame.sequence_id,
                    raw_point_count=frame.metadata.get("raw_point_count", len(pts)),
                    foveated_point_count=len(pts),
                    model_type="FoveatedPointSegNet"
                )
