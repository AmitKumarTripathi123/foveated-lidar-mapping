"""Loss Functions and Training-Set Class Weighting Engine (Phase 5).

Supports:
  1. Plain Cross-Entropy Loss with ignore_index=255
  2. Class-Weighted Cross-Entropy Loss (weights computed strictly from training split)
"""

from typing import Dict, Optional, Union
import numpy as np
import torch
import torch.nn as nn


def compute_class_weights(
    class_counts: Dict[int, int],
    num_classes: int = 4,
    strategy: str = "inverse_frequency",
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Compute per-class weights strictly from training split class counts.

    Args:
        class_counts: Dictionary mapping class ID (0..num_classes-1) to point count.
        num_classes: Number of target classes (default: 4).
        strategy: Weighting strategy ("inverse_frequency", "sqrt_inverse", "median_frequency").
        epsilon: Small numerical stability constant.

    Returns:
        torch.Tensor: Normalized weight tensor of shape (num_classes,).
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    for cid in range(num_classes):
        counts[cid] = class_counts.get(cid, 0)

    # Handle zero-count classes gracefully
    counts = np.maximum(counts, 1.0)

    if strategy == "inverse_frequency":
        raw_weights = 1.0 / (counts + epsilon)
    elif strategy == "sqrt_inverse":
        raw_weights = 1.0 / np.sqrt(counts + epsilon)
    elif strategy == "median_frequency":
        median_val = np.median(counts)
        raw_weights = median_val / (counts + epsilon)
    else:
        raise ValueError(
            f"Unknown class weighting strategy '{strategy}'. "
            f"Supported: 'inverse_frequency', 'sqrt_inverse', 'median_frequency'."
        )

    # Normalize weights so mean weight == 1.0
    normalized_weights = raw_weights / np.mean(raw_weights)
    return torch.from_numpy(normalized_weights.astype(np.float32)).float()


def get_loss_function(
    loss_type: str = "cross_entropy",
    class_weights: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ignore_index: int = 255,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Instantiate loss criterion with ignore_index and optional class weighting.

    Args:
        loss_type: Type of loss ("cross_entropy", "weighted_cross_entropy").
        class_weights: Optional class weight tensor of shape (num_classes,).
        ignore_index: Semantic index to ignore in loss calculation (default: 255).
        device: Device to place weight tensor on.

    Returns:
        nn.Module: PyTorch loss function.
    """
    weights_tensor: Optional[torch.Tensor] = None

    if loss_type in ("weighted_cross_entropy", "weighted_ce") or class_weights is not None:
        if class_weights is not None:
            if isinstance(class_weights, np.ndarray):
                weights_tensor = torch.from_numpy(class_weights).float()
            elif isinstance(class_weights, torch.Tensor):
                weights_tensor = class_weights.float()

            if device is not None:
                weights_tensor = weights_tensor.to(device)

    return nn.CrossEntropyLoss(weight=weights_tensor, ignore_index=ignore_index)
