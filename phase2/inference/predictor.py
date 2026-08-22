"""
Phase 2 Predictor & Output Interface Contract.
SemanticPrediction represents the frozen interface between Phase 2 AI and Phase 3 2.5D Mapping.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F

from phase2.models.point_seg_net import FoveatedPointSegNet
from src.types import PointCloudFrame


@dataclass
class SemanticPrediction:
    points: np.ndarray
    predicted_class: np.ndarray
    class_probabilities: np.ndarray
    confidence: np.ndarray
    frame_id: str = "000000"
    timestamp: float = 0.0
    sequence_id: str = "00"
    raw_point_count: int = 0
    foveated_point_count: int = 0

    @property
    def num_points(self) -> int:
        return len(self.points)

    def validate_interface(self) -> bool:
        N = len(self.points)
        assert len(self.predicted_class) == N
        assert self.class_probabilities.shape == (N, 4)
        assert len(self.confidence) == N
        assert (self.confidence >= 0.0).all() and (self.confidence <= 1.0001).all()
        assert np.isin(self.predicted_class, [0, 1, 2, 3]).all()
        return True


class Phase2Predictor:
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = "checkpoints/best_model.pth",
        device: Optional[str] = "cpu"
    ):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = FoveatedPointSegNet()
        if model_path and Path(model_path).exists():
            ckpt = torch.load(str(model_path), map_location=self.device)
            self.model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_frame(self, frame: PointCloudFrame) -> SemanticPrediction:
        pts = frame.points
        if len(pts) == 0:
            return SemanticPrediction(
                points=np.empty((0, 4), dtype=np.float32),
                predicted_class=np.empty(0, dtype=np.int64),
                class_probabilities=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty(0, dtype=np.float32),
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                sequence_id=frame.sequence_id
            )

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
            foveated_point_count=len(pts)
        )
