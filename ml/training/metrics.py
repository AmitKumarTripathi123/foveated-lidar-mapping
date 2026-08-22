"""Evaluation Metrics and 4x4 Confusion Matrix Engine (Phase 5).

Computes:
  - Per-class Intersection over Union (IoU)
  - Mean Intersection over Union (mIoU)
  - Per-class Precision & Recall
  - Overall Accuracy
  - 4x4 Confusion Matrix (ground truth rows, prediction columns)
  - Explicit exclusion of ignore target points (255)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from ml.data.label_mapping import SIH_CLASS_NAMES


@dataclass
class ClassMetric:
    """Detailed evaluation metrics for a single semantic class."""
    class_id: int
    class_name: str
    iou: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int
    support: int


@dataclass
class MetricReport:
    """Comprehensive segmentation evaluation summary report."""
    num_classes: int
    per_class: Dict[int, ClassMetric]
    miou: float
    overall_accuracy: float
    confusion_matrix: np.ndarray
    total_evaluated_points: int
    total_ignored_points: int


class SemanticSegmentationMetrics:
    """Accumulates predictions across batches and computes segmentation metrics."""

    def __init__(
        self,
        num_classes: int = 4,
        ignore_index: int = 255,
        class_names: Optional[Dict[int, str]] = None,
    ):
        """Initialize metric accumulator.

        Args:
            num_classes: Number of target semantic classes (default: 4).
            ignore_index: Semantic index to exclude from metrics (default: 255).
            class_names: Optional mapping from class ID to human-readable name.
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or dict(SIH_CLASS_NAMES)
        self.reset()

    def reset(self) -> None:
        """Reset internal confusion matrix and point counters."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_evaluated_points = 0
        self.total_ignored_points = 0

    def update(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
    ) -> None:
        """Update metrics with a new batch of predictions and ground-truth targets.

        Args:
            predictions: Array/Tensor of predicted class IDs (shape: (N,) or (B, N)).
            targets: Array/Tensor of ground-truth class IDs (shape: (N,) or (B, N)).
        """
        if isinstance(predictions, torch.Tensor):
            preds = predictions.detach().cpu().numpy().flatten()
        else:
            preds = np.asarray(predictions).flatten()

        if isinstance(targets, torch.Tensor):
            tgts = targets.detach().cpu().numpy().flatten()
        else:
            tgts = np.asarray(targets).flatten()

        if len(preds) != len(tgts):
            raise ValueError(f"Shape mismatch: preds={preds.shape}, tgts={tgts.shape}")

        # 1. Filter out ignore points (255)
        valid_mask = tgts != self.ignore_index
        ignored_count = int((~valid_mask).sum())
        self.total_ignored_points += ignored_count

        valid_preds = preds[valid_mask]
        valid_tgts = tgts[valid_mask]

        if len(valid_preds) == 0:
            return

        self.total_evaluated_points += len(valid_preds)

        # 2. Accumulate into confusion matrix: CM[target, prediction]
        for t, p in zip(valid_tgts, valid_preds):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.confusion_matrix[t, p] += 1

    def compute(self) -> MetricReport:
        """Compute final metrics from accumulated confusion matrix.

        Returns:
            MetricReport: Comprehensive metrics summary.
        """
        per_class: Dict[int, ClassMetric] = {}
        ious: List[float] = []

        total_tp = 0
        total_valid = 0

        for c in range(self.num_classes):
            c_name = self.class_names.get(c, f"class_{c}")
            tp = int(self.confusion_matrix[c, c])
            fp = int(self.confusion_matrix[:, c].sum() - tp)
            fn = int(self.confusion_matrix[c, :].sum() - tp)
            support = tp + fn

            total_tp += tp
            total_valid += support

            denom_iou = tp + fp + fn
            iou = float(tp) / float(denom_iou) if denom_iou > 0 else 0.0

            denom_prec = tp + fp
            precision = float(tp) / float(denom_prec) if denom_prec > 0 else 0.0

            denom_rec = tp + fn
            recall = float(tp) / float(denom_rec) if denom_rec > 0 else 0.0

            if support > 0 or denom_iou > 0:
                ious.append(iou)

            per_class[c] = ClassMetric(
                class_id=c,
                class_name=c_name,
                iou=iou,
                precision=precision,
                recall=recall,
                tp=tp,
                fp=fp,
                fn=fn,
                support=support,
            )

        miou = float(np.mean(ious)) if len(ious) > 0 else 0.0
        overall_acc = float(total_tp) / float(total_valid) if total_valid > 0 else 0.0

        return MetricReport(
            num_classes=self.num_classes,
            per_class=per_class,
            miou=miou,
            overall_accuracy=overall_acc,
            confusion_matrix=self.confusion_matrix.copy(),
            total_evaluated_points=self.total_evaluated_points,
            total_ignored_points=self.total_ignored_points,
        )


def format_metric_report(report: MetricReport) -> str:
    """Format MetricReport into a human-readable string summary."""
    lines = []
    lines.append("=" * 68)
    lines.append("              3D SEMANTIC SEGMENTATION EVALUATION REPORT")
    lines.append("=" * 68)
    lines.append(f"\nEvaluated Points: {report.total_evaluated_points:,}")
    lines.append(f"Ignored Points  : {report.total_ignored_points:,}")
    lines.append(f"Mean IoU (mIoU) : {report.miou * 100.0:6.2f}%")
    lines.append(f"Overall Accuracy: {report.overall_accuracy * 100.0:6.2f}%\n")

    # 1. Per-Class Metrics Table
    lines.append("1. Per-Class Performance:")
    lines.append("  " + "-" * 62)
    lines.append(f"  {'ID':>2} | {'Class Name':<20} | {'IoU':>8} | {'Precision':>10} | {'Recall':>8} | {'Support':>8}")
    lines.append("  " + "-" * 62)
    for cid in range(report.num_classes):
        cm = report.per_class[cid]
        lines.append(
            f"  {cm.class_id:2d} | {cm.class_name:<20} | {cm.iou * 100.0:7.2f}% | "
            f"{cm.precision * 100.0:9.2f}% | {cm.recall * 100.0:7.2f}% | {cm.support:8,d}"
        )
    lines.append("  " + "-" * 62)

    # 2. 4x4 Confusion Matrix
    lines.append("\n2. 4x4 Confusion Matrix (Rows = Ground Truth, Cols = Predicted):")
    lines.append("  " + "-" * 62)
    gt_pred_label = "GT \\ Pred"
    header = f"  {gt_pred_label:<12} | " + " | ".join(f"C{c:>7}" for c in range(report.num_classes))

    lines.append(header)
    lines.append("  " + "-" * 62)
    for r in range(report.num_classes):
        row_str = f"  {f'Class {r}':<12} | " + " | ".join(f"{report.confusion_matrix[r, c]:8,d}" for c in range(report.num_classes))
        lines.append(row_str)
    lines.append("  " + "-" * 62)
    lines.append("=" * 68)

    return "\n".join(lines)
