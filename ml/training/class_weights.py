"""
Phase 13: Advanced Class Weighting Calculators for 3D LiDAR Semantic Segmentation.
Calculates class balance weights strictly on training partition without data leakage.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset


def compute_training_class_frequencies(
    dataset: Dataset,
    num_classes: int = 4,
    ignore_index: int = 255,
    sample_fraction: float = 1.0,
) -> np.ndarray:
    """Compute per-class point counts across the training dataset.

    Args:
        dataset: Training dataset instance.
        num_classes: Number of target supervised classes (default: 4).
        ignore_index: Label index to ignore (default: 255).
        sample_fraction: Fraction of dataset frames to sample for speed (default: 1.0).

    Returns:
        np.ndarray: Vector of shape (num_classes,) with total point frequencies.
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    n_samples = len(dataset)
    step = max(1, int(1.0 / max(sample_fraction, 0.01))) if sample_fraction < 1.0 else 1

    for idx in range(0, n_samples, step):
        item = dataset[idx]
        if isinstance(item, (list, tuple)):
            lbls = item[1]
        elif isinstance(item, dict):
            lbls = None
            for key in ["labels", "label", "semantic_labels"]:
                if key in item:
                    lbls = item[key]
                    break
        else:
            continue

        if lbls is None:
            continue

        if isinstance(lbls, torch.Tensor):
            lbls_np = lbls.detach().cpu().numpy().astype(np.int64)
        else:
            lbls_np = np.asarray(lbls, dtype=np.int64)

        valid = (lbls_np != ignore_index) & (lbls_np >= 0) & (lbls_np < num_classes)
        v_lbls = lbls_np[valid]
        if len(v_lbls) > 0:
            b_counts = np.bincount(v_lbls, minlength=num_classes)
            counts += b_counts[:num_classes]

    return counts


def _parse_counts(
    dataset_or_counts: Union[Dataset, np.ndarray, List[int], Dict[Any, int], torch.Tensor],
    num_classes: int = 4,
    ignore_index: int = 255,
) -> np.ndarray:
    """Helper to convert various input types (dict, list, tensor, dataset) to count array."""
    if isinstance(dataset_or_counts, dict):
        counts = [float(dataset_or_counts.get(i, dataset_or_counts.get(str(i), 0))) for i in range(num_classes)]
        return np.asarray(counts, dtype=np.float64)
    elif isinstance(dataset_or_counts, torch.Tensor):
        return dataset_or_counts.detach().cpu().numpy().astype(np.float64)
    elif isinstance(dataset_or_counts, (np.ndarray, list)):
        return np.asarray(dataset_or_counts, dtype=np.float64)
    else:
        return compute_training_class_frequencies(dataset_or_counts, num_classes, ignore_index).astype(np.float64)


def compute_inverse_frequency_weights(
    dataset_or_counts: Union[Dataset, np.ndarray, List[int], Dict[Any, int], torch.Tensor],
    num_classes: int = 4,
    ignore_index: int = 255,
    smooth: float = 1.0,
) -> List[float]:
    """Inverse frequency weighting: w_c = 1.0 / (N_c + smooth)."""
    counts = _parse_counts(dataset_or_counts, num_classes, ignore_index)

    total_valid = np.sum(counts)
    if total_valid == 0:
        return [1.0] * num_classes

    freqs = (counts + smooth) / (total_valid + smooth * num_classes)
    raw_weights = 1.0 / freqs
    norm_weights = raw_weights / np.mean(raw_weights)
    return [round(float(w), 4) for w in norm_weights]


def compute_sqrt_inverse_frequency_weights(
    dataset_or_counts: Union[Dataset, np.ndarray, List[int], Dict[Any, int], torch.Tensor],
    num_classes: int = 4,
    ignore_index: int = 255,
    smooth: float = 1.0,
) -> List[float]:
    """Square-root inverse frequency weighting: w_c = 1.0 / sqrt(N_c + smooth)."""
    counts = _parse_counts(dataset_or_counts, num_classes, ignore_index)

    total_valid = np.sum(counts)
    if total_valid == 0:
        return [1.0] * num_classes

    freqs = (counts + smooth) / (total_valid + smooth * num_classes)
    raw_weights = 1.0 / np.sqrt(freqs)
    norm_weights = raw_weights / np.mean(raw_weights)
    return [round(float(w), 4) for w in norm_weights]


def compute_effective_number_weights(
    dataset_or_counts: Union[Dataset, np.ndarray, List[int], Dict[Any, int], torch.Tensor],
    beta: float = 0.9999,
    num_classes: int = 4,
    ignore_index: int = 255,
) -> List[float]:
    """Effective Number of Samples weighting (Cui et al., CVPR 2019):
    w_c = (1 - beta) / (1 - beta^N_c)
    """
    counts = _parse_counts(dataset_or_counts, num_classes, ignore_index)

    effective_num = 1.0 - np.power(beta, counts + 1.0)
    raw_weights = (1.0 - beta) / (effective_num + 1e-8)
    norm_weights = raw_weights / np.mean(raw_weights)
    return [round(float(w), 4) for w in norm_weights]


def get_class_weights(
    dataset_or_counts: Union[Dataset, np.ndarray, List[int]],
    strategy: str = "inverse",
    beta: float = 0.9999,
    num_classes: int = 4,
    ignore_index: int = 255,
) -> List[float]:
    """Factory retrieving class weights according to strategy."""
    strat = str(strategy).lower().strip()
    if strat in ("inverse", "inv", "inverse_freq", "inverse_frequency"):
        return compute_inverse_frequency_weights(dataset_or_counts, num_classes, ignore_index)
    elif strat in ("sqrt", "sqrt_inv", "sqrt_inverse", "sqrt_inverse_frequency"):
        return compute_sqrt_inverse_frequency_weights(dataset_or_counts, num_classes, ignore_index)
    elif strat in ("effective", "effective_num", "effective_number", "cui"):
        return compute_effective_number_weights(dataset_or_counts, beta=beta, num_classes=num_classes, ignore_index=ignore_index)
    elif strat in ("none", "uniform", "equal"):
        return [1.0] * num_classes
    else:
        raise ValueError(f"Unknown class weighting strategy: {strategy}")
