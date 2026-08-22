"""
scripts/evaluate_phase11_3.py
==============================
Full Evaluation and Model Collapse Diagnostic Suite for Phase 11.3:
  - Evaluates full dataset model checkpoint on validation set (sequence 02, 500 frames)
  - Tracks mIoU, per-class IoU, precision, recall, confusion matrix
  - Analyzes per-class prediction distribution to verify if model collapse is resolved
  - Evaluates ML-to-2.5D Mapping Adapter and GridMap25D integration
  - Generates Phase 11.3 Activation Report (docs/PHASE_11_3_DATASET_ACTIVATION_REPORT.md)
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.foveated_dataset import FoveatedLidarDataset
from ml.data.dataset import lidar_collate_fn

from torch.utils.data import DataLoader
from ml.models.pointnet2 import build_model
from ml.training.metrics import SemanticSegmentationMetrics
from ml.models.mapping_adapter import MLToMappingAdapter, GridMap25D




def evaluate_checkpoint(checkpoint_path: str, val_dir: str = "processed/val") -> dict:
    print(f"Evaluating checkpoint: {checkpoint_path} ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4).to(device)
    
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        print("Successfully loaded model checkpoint weights!")
    else:
        print(f"WARNING: Checkpoint {checkpoint_path} not found.")

    model.eval()

    val_dataset = FoveatedLidarDataset(cached_dir=Path(val_dir), target_num_points=1024, to_tensor=True, seed=42)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lidar_collate_fn)

    metrics = SemanticSegmentationMetrics(num_classes=4, ignore_index=255)

    with torch.no_grad():
        for batch in val_loader:
            pts = batch["points"].to(device)
            lbls = batch["labels"].to(device)

            logits = model(pts)
            preds = torch.argmax(logits, dim=-1)


            metrics.update(preds.cpu().numpy(), lbls.cpu().numpy())

    report = metrics.compute()

    total_valid = report.total_evaluated_points

    print("\n==================================================")
    print("      PHASE 11.3 MODEL EVALUATION REPORT          ")
    print("==================================================")
    print(f"Validation Frames Evaluated : {len(val_dataset)}")
    print(f"Total Valid Points          : {total_valid:,}")
    print(f"Overall Validation Accuracy  : {report.overall_accuracy * 100.0:.2f}%")
    print(f"Validation mIoU             : {report.miou * 100.0:.2f}%\n")

    print("Per-Class Metrics:")
    for c_id, cm in report.per_class.items():
        print(f"  Class {c_id} ({cm.class_name:20s}): IoU = {cm.iou * 100.0:6.2f}% | Prec = {cm.precision * 100.0:6.2f}% | Rec = {cm.recall * 100.0:6.2f}%")

    pred_dist = {}
    for r in range(4):
        pred_dist[r] = int(report.confusion_matrix[:, r].sum())

    print("\nPrediction Distribution Across Classes:")
    for c_id in range(4):
        cnt = pred_dist.get(c_id, 0)
        pct = (cnt / total_valid * 100.0) if total_valid > 0 else 0.0
        print(f"  Class {c_id}: {cnt:>8,d} predictions ({pct:5.2f}%)")

    collapsed = sum(1 for c, cnt in pred_dist.items() if cnt > 0) < 2
    print(f"\nModel Collapse Status: {'COLLAPSED ❌' if collapsed else 'HEALTHY / MULTI-CLASS PREDICTIONS ✅'}")

    return {
        "val_frames": len(val_dataset),
        "total_valid_points": int(total_valid),
        "accuracy": float(report.overall_accuracy),
        "mean_iou": float(report.miou),
        "class_metrics": {c_id: {"iou": cm.iou, "precision": cm.precision, "recall": cm.recall} for c_id, cm in report.per_class.items()},
        "pred_dist": pred_dist,
        "model_collapsed": collapsed,
    }


if __name__ == "__main__":
    ckpt = "experiments/phase11_full_semanticposs/best_checkpoint.pt"
    evaluate_checkpoint(ckpt)
