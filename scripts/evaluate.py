#!/usr/bin/env python3
"""PointNet++ Checkpoint Evaluation, Contract Verification & Consistency CLI (Master Task).

Evaluates the exact same validation dataset used during training to verify:
  1. Exact metric consistency between training-time and post-reload evaluation
  2. 4x4 Confusion matrix & per-class IoU/precision/recall
  3. Prediction distribution and model collapse detection
  4. Amit's frozen ML -> Mapping contract [x, y, z, predicted_class, confidence]

Usage:
    python scripts/evaluate.py --checkpoint experiments/baseline_ce/best_checkpoint.pt
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.dataset import load_point_cloud, load_labels, lidar_collate_fn
from ml.data.preprocessing import filter_invalid_points
from ml.data.amit_adapter import FoveatedVoxelSampler
from ml.data.label_mapping import SemanticLabelRemapper, SIH_CLASS_NAMES
from ml.data.foveated_dataset import FoveatedLidarDataset
from ml.models.pointnet2 import build_model
from ml.models.predictor import PointNet2Predictor
from ml.training.metrics import SemanticSegmentationMetrics, format_metric_report


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Evaluate a trained PointNet++ checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--val-dir", type=str, default="processed/val", help="Path to cached validation dataset")
    parser.add_argument("--num-points", type=int, default=1024, help="Evaluation point resolution")

    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        print(f"Error: Checkpoint file not found at {ckpt_path.resolve()}")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device)

    # 1. Rebuild and load model
    model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    ckpt_epoch = checkpoint.get("epoch", -1)
    ckpt_val_miou = checkpoint.get("val_miou", 0.0) * 100.0
    print("=" * 68)
    print(f"LOADED CHECKPOINT: {ckpt_path.name}")
    print(f"Checkpoint Epoch: {ckpt_epoch} | Checkpoint Best Val mIoU: {ckpt_val_miou:5.2f}%")
    print("=" * 68 + "\n")

    # 2. Evaluate on the EXACT SAME validation split as training
    val_dir = Path(args.val_dir)
    num_points = args.num_points

    if val_dir.is_dir() and len(list(val_dir.glob("*_pts.npy"))) > 0:
        val_dataset = FoveatedLidarDataset(
            cached_dir=val_dir, target_num_points=num_points, to_tensor=True, seed=1042
        )
    else:
        from ml.data.manifest import discover_dataset
        manifest = discover_dataset("dataset")
        val_dataset = FoveatedLidarDataset(
            raw_manifest=manifest["val"], target_num_points=num_points, to_tensor=True, seed=1042
        )

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lidar_collate_fn)

    metrics = SemanticSegmentationMetrics(num_classes=4, ignore_index=255)

    with torch.no_grad():
        for batch in val_loader:
            pts = batch["points"].to(device).float()
            lbls = batch["labels"].to(device).long()
            logits = model(pts)
            preds = logits.argmax(dim=-1)
            metrics.update(preds, lbls)

    report = metrics.compute()
    print(format_metric_report(report))

    # 3. Checkpoint Reload Metric Consistency Check
    reloaded_val_miou = report.miou * 100.0
    miou_diff = abs(reloaded_val_miou - ckpt_val_miou)
    consistency_pass = miou_diff < 0.01

    print("\n3. Metric Consistency Verification:")
    print("  " + "-" * 62)
    print(f"  Training-Time Validation mIoU : {ckpt_val_miou:6.2f}%")
    print(f"  Post-Reload Validation mIoU  : {reloaded_val_miou:6.2f}%")
    print(f"  Absolute Delta               : {miou_diff:6.4f}%")
    print(f"  Status                       : [{'PASS' if consistency_pass else 'FAIL'}]")

    # 4. Model Collapse Detection
    cm = report.confusion_matrix
    total_preds = int(cm.sum())
    pred_by_class = [int(cm[:, c].sum()) for c in range(4)]
    max_pred_pct = max(pred_by_class) / total_preds if total_preds > 0 else 0.0
    is_collapsed = max_pred_pct > 0.90

    print("\n4. Prediction Distribution & Collapse Diagnostic:")
    print("  " + "-" * 62)
    for c in range(4):
        pct = (pred_by_class[c] / total_preds) * 100.0 if total_preds > 0 else 0.0
        print(f"  Class {c} ({SIH_CLASS_NAMES[c]:<20}) : {pred_by_class[c]:6,d} pts ({pct:5.2f}%)")
    print("  " + "-" * 62)
    if is_collapsed:
        print("  [WARNING] MODEL_COLLAPSE_WARNING: Single class accounts for >90% of predictions.")
    else:
        print("  [OK] No single-class collapse detected (>90% threshold).")

    # 5. Amit's Frozen ML -> Mapping Contract Verification
    predictor = PointNet2Predictor(model=model, device=device)
    raw_sample = val_dataset[0]
    raw_sample_pts = raw_sample["points"].numpy()
    pred_res = predictor.predict(raw_sample_pts)

    out_xyz = pred_res["xyz"]
    out_class = pred_res["predicted_class"]
    out_conf = pred_res["confidence"]

    xyz_ok = np.array_equal(raw_sample_pts[:, :3].astype(np.float32), out_xyz)
    class_ok = set(np.unique(out_class)).issubset({0, 1, 2, 3})
    conf_ok = bool((out_conf >= 0.0).all() and (out_conf <= 1.0).all())

    print("\n5. Amit's Frozen ML -> Mapping Contract Verification:")
    print("  " + "-" * 62)
    print(f"  [{'PASS' if xyz_ok else 'FAIL'}] Exact 1-to-1 XYZ coordinates preserved")
    print(f"  [{'PASS' if class_ok else 'FAIL'}] Predicted classes strictly in {{0, 1, 2, 3}}")
    print(f"  [{'PASS' if conf_ok else 'FAIL'}] Confidence strictly in [0.0, 1.0] (mean={out_conf.mean():.4f})")
    print("=" * 68 + "\n")

    return 0 if (consistency_pass and xyz_ok and class_ok and conf_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
