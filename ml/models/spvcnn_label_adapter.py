"""SPVCNN Label Adapter for SIH 4-Class Semantic Ontology (Phase 12).

Maps SPVCNN native class predictions (e.g. 19-class SemanticKITTI or 14-class SemanticPOSS)
into the frozen Smart India Hackathon (SIH) 4-Class super-class ontology:
    0: drivable_terrain
    1: non_drivable_terrain
    2: static_obstacle
    3: dynamic_object
    255: ignore
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

# Authoritative SemanticKITTI 19-class to SIH 4-class ontology mapping
SEMANTICKITTI_TO_SIH = {
    0: 3,    # car -> dynamic_object
    1: 3,    # bicycle -> dynamic_object
    2: 3,    # motorcycle -> dynamic_object
    3: 3,    # truck -> dynamic_object
    4: 3,    # other-vehicle -> dynamic_object
    5: 3,    # person -> dynamic_object
    6: 3,    # bicyclist -> dynamic_object
    7: 3,    # motorcyclist -> dynamic_object
    8: 0,    # road -> drivable_terrain
    9: 0,    # parking -> drivable_terrain
    10: 1,   # sidewalk -> non_drivable_terrain
    11: 1,   # other-ground -> non_drivable_terrain
    12: 2,   # building -> static_obstacle
    13: 2,   # fence -> static_obstacle
    14: 2,   # vegetation -> static_obstacle
    15: 2,   # trunk -> static_obstacle
    16: 1,   # terrain -> non_drivable_terrain
    17: 2,   # pole -> static_obstacle
    18: 2,   # traffic-sign -> static_obstacle
}

# Authoritative SemanticPOSS raw to SIH 4-class ontology mapping
SEMANTICPOSS_TO_SIH = {
    0: 255,  # unlabeled -> ignore
    1: 255,  # outlier -> ignore
    4: 3,    # people -> dynamic_object
    5: 3,    # people -> dynamic_object
    6: 3,    # rider -> dynamic_object
    7: 3,    # car -> dynamic_object
    8: 2,    # trunk -> static_obstacle
    9: 2,    # plants -> static_obstacle
    10: 2,   # traffic sign -> static_obstacle
    11: 2,   # traffic sign -> static_obstacle
    12: 2,   # traffic sign -> static_obstacle
    13: 2,   # pole -> static_obstacle
    14: 2,   # trashcan -> static_obstacle
    15: 2,   # building -> static_obstacle
    16: 2,   # cone/stone -> static_obstacle
    17: 2,   # fence -> static_obstacle
    18: 2,   # bike-fence -> static_obstacle
    19: 2,   # other-structure -> static_obstacle
    20: 0,   # road -> drivable_terrain
    21: 3,   # bike -> dynamic_object
    22: 1,   # ground -> non_drivable_terrain
}


class SPVCNNLabelAdapter:
    """Adapts SPVCNN native class logits and predictions to the frozen SIH 4-class ontology."""

    def __init__(
        self,
        native_source: str = "semantickitti",
        custom_mapping: Optional[Dict[int, int]] = None,
    ):
        """Initialize adapter.

        Args:
            native_source: Native class scheme ('semantickitti', 'semanticposs', or 'sih_direct').
            custom_mapping: Optional override mapping dictionary.
        """
        self.native_source = native_source.lower()
        if custom_mapping is not None:
            self.mapping = custom_mapping
        elif self.native_source == "semantickitti":
            self.mapping = SEMANTICKITTI_TO_SIH
        elif self.native_source == "semanticposs":
            self.mapping = SEMANTICPOSS_TO_SIH
        elif self.native_source == "sih_direct":
            self.mapping = {0: 0, 1: 1, 2: 2, 3: 3}
        else:
            self.mapping = SEMANTICKITTI_TO_SIH

        # Build lookup table for fast O(N) numpy vectorization
        max_class = max(self.mapping.keys()) if self.mapping else 255
        self.lut = np.full(max(max_class + 1, 256), 255, dtype=np.int64)
        for k, v in self.mapping.items():
            if 0 <= k < len(self.lut):
                self.lut[k] = v

    def remap_predictions(
        self,
        native_classes: np.ndarray,
    ) -> np.ndarray:
        """Remap native class predictions to SIH classes.

        Args:
            native_classes: Array of native class IDs of shape (N,).

        Returns:
            Array of SIH class IDs of shape (N,) in {0, 1, 2, 3, 255}.
        """
        raw = np.asarray(native_classes, dtype=np.int64)
        valid_mask = (raw >= 0) & (raw < len(self.lut))
        sih_classes = np.full_like(raw, 255)
        sih_classes[valid_mask] = self.lut[raw[valid_mask]]
        return sih_classes

    def process_logits(
        self,
        logits: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert raw network logits to calibrated confidence and SIH classes.

        Args:
            logits: Array or Tensor of shape (N, C) containing raw output logits.

        Returns:
            Tuple of (sih_predicted_classes, confidences).
        """
        if isinstance(logits, torch.Tensor):
            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)
                confs, native_preds = torch.max(probs, dim=-1)
                confs_np = confs.detach().cpu().numpy().astype(np.float32)
                native_preds_np = native_preds.detach().cpu().numpy().astype(np.int64)
        else:
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            native_preds_np = np.argmax(probs, axis=-1).astype(np.int64)
            confs_np = np.max(probs, axis=-1).astype(np.float32)

        sih_preds_np = self.remap_predictions(native_preds_np)
        return sih_preds_np, confs_np

    def audit_predictions(
        self,
        sih_predictions: np.ndarray,
    ) -> Dict[str, Any]:
        """Audit the distribution of SIH class predictions."""
        total = len(sih_predictions)
        unique_classes, counts = np.unique(sih_predictions, return_counts=True)
        dist = {int(cls): int(cnt) for cls, cnt in zip(unique_classes, counts)}

        mapped_count = sum(cnt for cls, cnt in dist.items() if cls in {0, 1, 2, 3})
        ignored_count = dist.get(255, 0)

        return {
            "total_points": total,
            "mapped_points": mapped_count,
            "ignored_points": ignored_count,
            "mapped_percentage": round(100 * mapped_count / total, 2) if total else 0.0,
            "class_distribution": dist,
        }
