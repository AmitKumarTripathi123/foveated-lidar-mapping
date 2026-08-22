"""SPVCNN Predictor implementing the Frozen ML -> Mapping Interface (Phase 12).

Enforces Amit''s Frozen Output Contract:
    [x, y, z, predicted_class, confidence]

Contract Requirements:
    1. coordinates:     (N, 3) float32 (identical in value & order to input XYZ)
    2. predicted_class: (N,)   int64 / uint8, values in {0, 1, 2, 3, 255}
    3. confidence:      (N,)   float32, values in [0.0, 1.0] via max(softmax(logits))
"""

from typing import Any, Dict, Optional, Union
import numpy as np
import torch

from ml.models.spvcnn import SPVCNN, build_spvcnn
from ml.data.spvcnn_adapter import SPVCNNInputAdapter
from ml.models.spvcnn_label_adapter import SPVCNNLabelAdapter


class SPVCNNPredictor:
    """Inference predictor wrapping SPVCNN with frozen ML -> Mapping contract enforcement."""

    def __init__(
        self,
        model: Optional[SPVCNN] = None,
        device: Optional[Union[str, torch.device]] = None,
        voxel_size: float = 0.05,
        native_source: str = "semantickitti",
        num_classes: int = 19,
        pretrained_path: Optional[str] = None,
    ):
        """Initialize SPVCNN predictor.

        Args:
            model: Optional SPVCNN model instance. If None, builds a new model.
            device: Computing device ('cpu', 'cuda', or torch.device).
            voxel_size: Voxel quantization size in meters (default: 0.05m).
            native_source: Native class scheme for label adapter ('semantickitti', 'semanticposs', or 'sih_direct').
            num_classes: Native model class count.
            pretrained_path: Optional path to pretrained weights.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        if model is None:
            self.model = build_spvcnn(
                num_classes=num_classes,
                in_channels=4,
                pretrained_path=pretrained_path,
                device=self.device,
            )
        else:
            self.model = model.to(self.device)

        self.model.eval()
        self.input_adapter = SPVCNNInputAdapter(voxel_size=voxel_size)
        self.label_adapter = SPVCNNLabelAdapter(native_source=native_source)

    def predict(
        self,
        points: Union[np.ndarray, torch.Tensor],
    ) -> Dict[str, np.ndarray]:
        """Run inference on a point cloud frame and return frozen mapping contract.

        Args:
            points: Input point cloud array/tensor of shape (N, 4) with [x, y, z, intensity].

        Returns:
            Dict containing:
                - 'xyz': (N, 3) float32 coordinates (exact input order preserved)
                - 'predicted_class': (N,) int64 class indices in {0, 1, 2, 3, 255}
                - 'confidence': (N,) float32 confidence scores in [0.0, 1.0]
        """
        # 1. Prepare sparse point-voxel input bundle
        bundle = self.input_adapter.prepare_input(points, device=self.device)
        original_xyz = bundle["raw_xyz"].astype(np.float32)

        # 2. Forward pass through SPVCNN
        with torch.no_grad():
            logits = self.model(
                features=bundle["features"],
                point_to_voxel_idx=bundle["point_to_voxel_idx"],
                num_voxels=bundle["num_voxels"],
            )  # (N, num_classes)

        # 3. Process logits into SIH classes and confidence scores
        sih_classes, confidences = self.label_adapter.process_logits(logits)

        # 4. Strict contract verification
        n = original_xyz.shape[0]
        assert sih_classes.shape[0] == n, f"Class count {sih_classes.shape[0]} != point count {n}"
        assert confidences.shape[0] == n, f"Confidence count {confidences.shape[0]} != point count {n}"
        assert np.all(confidences >= 0.0) and np.all(confidences <= 1.0), "Confidences must be in [0.0, 1.0]"
        assert set(np.unique(sih_classes)).issubset({0, 1, 2, 3, 255}), (
            f"Predicted SIH classes {np.unique(sih_classes)} outside allowed {0, 1, 2, 3, 255}"
        )

        return {
            "xyz": original_xyz,
            "predicted_class": sih_classes.astype(np.int64),
            "confidence": confidences.astype(np.float32),
        }
