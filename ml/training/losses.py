"""
Phase 13: Advanced Loss Functions for 3D LiDAR Semantic Segmentation.
Implements:
- Standard Cross Entropy Loss
- Class-Weighted Cross Entropy Loss
- Multi-Class Focal Loss (Lin et al., ICCV 2017)
- Class-Balanced Focal Loss (Cui et al., CVPR 2019)
All loss functions strictly support ignore_index=255 and numeric stability.
"""

from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class Focal Loss supporting ignore_index and class weights."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[List[float], torch.Tensor]] = None,
        ignore_index: int = 255,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.ignore_index = int(ignore_index)
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, list):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = alpha.float()
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss between logits (N, C) and targets (N,).

        Args:
            logits: Predicted class logits (N, C).
            targets: Ground-truth class labels (N,).

        Returns:
            torch.Tensor: Computed scalar focal loss.
        """
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask]

        log_probs = F.log_softmax(valid_logits, dim=-1)
        probs = torch.exp(log_probs)

        # Gather target log_probs and probs: (N_valid,)
        target_log_probs = log_probs.gather(dim=-1, index=valid_targets.unsqueeze(-1)).squeeze(-1)
        target_probs = probs.gather(dim=-1, index=valid_targets.unsqueeze(-1)).squeeze(-1)

        # Focal modulating factor: (1 - p_t)^gamma
        focal_weight = torch.pow(1.0 - target_probs + 1e-8, self.gamma)

        # Alpha class balancing
        if self.alpha is not None:
            alpha_device = self.alpha.to(valid_logits.device)
            target_alpha = alpha_device.gather(dim=0, index=valid_targets)
            loss = -target_alpha * focal_weight * target_log_probs
        else:
            loss = -focal_weight * target_log_probs

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def build_loss_function(
    config: Optional[Dict[str, Any]] = None,
    class_weights: Optional[Union[List[float], torch.Tensor]] = None,
    device: Optional[Union[str, torch.device]] = None,
    loss_type: Optional[str] = None,
    ignore_index: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    """Build loss function based on experiment configuration or direct arguments."""
    loss_cfg = config.get("loss", {}) if config is not None else {}
    l_type = loss_type or loss_cfg.get("type", "cross_entropy")
    l_type = str(l_type).lower().strip()

    ign_idx = ignore_index if ignore_index is not None else int(loss_cfg.get("ignore_index", 255))
    weights = class_weights or loss_cfg.get("class_weights")

    weight_tensor = None
    if weights is not None:
        if isinstance(weights, torch.Tensor):
            weight_tensor = weights.float()
        else:
            weight_tensor = torch.tensor(weights, dtype=torch.float32)
        if device is not None:
            weight_tensor = weight_tensor.to(device)

    if l_type in ("cross_entropy", "ce"):
        return nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=ign_idx)
    elif l_type in ("weighted_cross_entropy", "weighted_ce", "w_ce"):
        assert weight_tensor is not None, "Weighted Cross-Entropy requires non-null class weights"
        return nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=ign_idx)
    elif l_type in ("focal", "focal_loss"):
        gamma = float(kwargs.get("gamma", loss_cfg.get("gamma", 2.0)))
        return FocalLoss(gamma=gamma, alpha=weight_tensor, ignore_index=ign_idx)
    elif l_type in ("class_balanced_focal", "balanced_focal", "cb_focal"):
        gamma = float(kwargs.get("gamma", loss_cfg.get("gamma", 2.0)))
        assert weight_tensor is not None, "Class Balanced Focal Loss requires class weights (alpha)"
        return FocalLoss(gamma=gamma, alpha=weight_tensor, ignore_index=ign_idx)
    else:
        raise ValueError(f"Unsupported loss type: {l_type}")


def get_loss_function(
    loss_type: str = "cross_entropy",
    class_weights: Optional[Union[List[float], torch.Tensor]] = None,
    ignore_index: int = 255,
    device: Optional[Union[str, torch.device]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> nn.Module:
    """Legacy alias for build_loss_function supporting direct arguments."""
    return build_loss_function(
        config=config,
        class_weights=class_weights,
        device=device,
        loss_type=loss_type,
        ignore_index=ignore_index,
        **kwargs,
    )


def compute_class_weights(
    dataset_or_counts: Any,
    num_classes: int = 4,
    ignore_index: int = 255,
    strategy: str = "inverse_frequency",
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Compute class weights returning torch.Tensor for backward compatibility."""
    from ml.training.class_weights import get_class_weights
    strat_clean = "inverse" if "inv" in strategy else strategy
    w_list = get_class_weights(dataset_or_counts, strategy=strat_clean, num_classes=num_classes, ignore_index=ignore_index)
    w_tensor = torch.tensor(w_list, dtype=torch.float32)
    if device is not None:
        w_tensor = w_tensor.to(device)
    return w_tensor

