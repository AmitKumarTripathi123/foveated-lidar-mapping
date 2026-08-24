"""
Phase 19.1 Global Semantic Accuracy Auditor.
Computes overall mIoU, point accuracy, mean class accuracy, and per-class
IoU, Precision, Recall, F1, and Support against ground truth labels (excluding 255).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ml.data.dataset import load_labels
from ml.data.semanticposs_label_mapping import SemanticPOSSLabelRemapper

_REMAPPER = SemanticPOSSLabelRemapper()

def remap_semanticposs_labels(raw_labels: np.ndarray) -> np.ndarray:
    """Vectorized authoritative remapping of SemanticPOSS 16-bit labels to 4-class SIH ontology."""
    return _REMAPPER.remap(raw_labels)


CLASS_KEYS = ["drivable", "non_drivable", "static", "dynamic"]
CLASS_IDS = [0, 1, 2, 3]


def compute_multiclass_metrics(confusion_matrix: np.ndarray) -> Dict[str, Any]:
    """Compute mIoU, per-class IoU, precision, recall, F1, and support from a 4x4 confusion matrix."""
    assert confusion_matrix.shape == (4, 4), f"Expected (4, 4) confusion matrix, got {confusion_matrix.shape}"
    
    tp = np.diag(confusion_matrix).astype(np.float64)
    fp = (np.sum(confusion_matrix, axis=0) - tp).astype(np.float64)
    fn = (np.sum(confusion_matrix, axis=1) - tp).astype(np.float64)
    support = np.sum(confusion_matrix, axis=1).astype(np.int64)

    ious = []
    precisions = []
    recalls = []
    f1s = []
    class_stats = {}

    for i, c_name in enumerate(CLASS_KEYS):
        denom_iou = tp[i] + fp[i] + fn[i]
        iou = float(tp[i] / denom_iou) if denom_iou > 0 else 0.0
        ious.append(iou)

        denom_prec = tp[i] + fp[i]
        prec = float(tp[i] / denom_prec) if denom_prec > 0 else 0.0
        precisions.append(prec)

        denom_rec = tp[i] + fn[i]
        rec = float(tp[i] / denom_rec) if denom_rec > 0 else 0.0
        recalls.append(rec)

        denom_f1 = prec + rec
        f1 = float(2 * prec * rec / denom_f1) if denom_f1 > 0 else 0.0
        f1s.append(f1)

        class_stats[c_name] = {
            "iou": round(iou, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int(support[i]),
        }

    total_pts = float(np.sum(confusion_matrix))
    total_tp = float(np.sum(tp))
    point_acc = float(total_tp / total_pts) if total_pts > 0 else 0.0
    mean_class_acc = float(np.mean(recalls))
    miou = float(np.mean(ious))

    return {
        "overall": {
            "miou": round(miou, 4),
            "point_accuracy": round(point_acc, 4),
            "mean_class_accuracy": round(mean_class_acc, 4),
            "total_valid_points": int(total_pts),
        },
        "classes": class_stats,
    }


def update_confusion_matrix(
    cm: np.ndarray,
    preds: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Accumulate predictions and ground-truth targets into 4x4 confusion matrix, ignoring label 255."""
    assert preds.shape[0] == targets.shape[0]
    valid_mask = (targets >= 0) & (targets <= 3) & (preds >= 0) & (preds <= 3)
    
    if np.any(valid_mask):
        valid_targets = targets[valid_mask].astype(np.int64)
        valid_preds = preds[valid_mask].astype(np.int64)
        
        flat_idx = valid_targets * 4 + valid_preds
        counts = np.bincount(flat_idx, minlength=16).reshape(4, 4)
        cm += counts
        
    return cm
