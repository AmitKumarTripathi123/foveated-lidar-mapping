"""PointNet++ Predictor and Frozen ML -> Mapping Interface (Phase 4).

Enforces Amit's Frozen Output Contract:
    [x, y, z, predicted_class, confidence]

Contract Requirements:
    1. coordinates:     (N, 3) float32 (identical in value & order to input XYZ)
    2. predicted_class: (N,)   int64 / uint8, values in {0, 1, 2, 3}
    3. confidence:      (N,)   float32, values in [0.0, 1.0] via max(softmax(logits))
"""

from typing import Any, Dict, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F

from ml.models.pointnet2 import PointNet2SemSeg, build_model


class PointNet2Predictor:
    """Inference predictor implementing Amit's frozen ML -> Mapping contract."""

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        device: Optional[Union[str, torch.device]] = None,
        num_classes: int = 4,
    ):
        """Initialize predictor.

        Args:
            model: Optional PyTorch PointNet++ model. If None, builds default model.
            device: Computing device ('cpu', 'cuda', or torch.device).
            num_classes: Number of target classes (default: 4).
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        if model is None:
            self.model = build_model(name="pointnet2_semseg", num_classes=num_classes)
        else:
            self.model = model

        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        points: Union[np.ndarray, torch.Tensor],
    ) -> Dict[str, np.ndarray]:
        """Run inference on a single point cloud frame and return frozen mapping contract.

        Args:
            points: Input point cloud array/tensor of shape (N, 4) with [x, y, z, intensity].

        Returns:
            Dict containing:
                - 'xyz': (N, 3) float32 coordinates (exact input order preserved)
                - 'predicted_class': (N,) int64 class indices in {0, 1, 2, 3}
                - 'confidence': (N,) float32 confidence scores in [0.0, 1.0]
        """
        # 1. Preserve original XYZ and input ordering
        if isinstance(points, torch.Tensor):
            original_xyz = points[:, :3].cpu().numpy().astype(np.float32)
            pts_tensor = points.to(self.device).float()
        else:
            original_xyz = points[:, :3].astype(np.float32)
            pts_tensor = torch.from_numpy(points.copy()).to(self.device).float()

        if pts_tensor.ndim == 2:
            pts_tensor = pts_tensor.unsqueeze(0)  # [1, N, 4]

        # 2. Model forward pass (per-point logits)
        with torch.no_grad():
            logits = self.model(pts_tensor)  # [1, N, 4]
            probabilities = F.softmax(logits, dim=-1)  # [1, N, 4]

            confidences, predicted_classes = torch.max(probabilities, dim=-1)  # [1, N], [1, N]

        # 3. Extract output numpy arrays
        pred_class_np = predicted_classes.squeeze(0).cpu().numpy().astype(np.int64)
        conf_np = confidences.squeeze(0).cpu().numpy().astype(np.float32)

        # 4. Strict contract verification
        assert original_xyz.shape[0] == pred_class_np.shape[0] == conf_np.shape[0], (
            f"Point count mismatch in output contract: "
            f"XYZ={original_xyz.shape[0]}, Class={pred_class_np.shape[0]}, Conf={conf_np.shape[0]}"
        )
        assert np.all(conf_np >= 0.0) and np.all(conf_np <= 1.0), "Confidence scores must be in [0.0, 1.0]"
        assert set(np.unique(pred_class_np)).issubset({0, 1, 2, 3}), (
            f"Predicted classes {np.unique(pred_class_np)} outside allowed {0, 1, 2, 3}"
        )

        return {
            "xyz": original_xyz,
            "predicted_class": pred_class_np,
            "confidence": conf_np,
        }
