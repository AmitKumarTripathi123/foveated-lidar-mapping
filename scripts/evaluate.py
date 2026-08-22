#!/usr/bin/env python3
"""PointNet++ Checkpoint Evaluation and Contract Verification CLI (Phase 5).

Usage:
    python scripts/evaluate.py --checkpoint experiments/baseline_ce/best_checkpoint.pt
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.dataset import load_point_cloud, load_labels
from ml.data.preprocessing import LidarPreprocessor, PreprocessingConfig, SamplingConfig
from ml.data.label_mapping import SemanticLabelRemapper, SIH_CLASS_NAMES
from ml.models.pointnet2 import build_model
from ml.models.predictor import PointNet2Predictor
from ml.training.metrics import SemanticSegmentationMetrics, format_metric_report


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Evaluate a trained PointNet++ checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--scan", type=str, default="dataset/sequences/00/velodyne/000000.bin", help="Path to .bin scan")
    parser.add_argument("--label", type=str, default="dataset/sequences/00/labels/000000.label", help="Path to .label file")
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

    print(f"Loaded checkpoint from: {ckpt_path.name}")
    print(f"Checkpoint Epoch: {checkpoint.get('epoch')}, Checkpoint Best Val mIoU: {checkpoint.get('val_miou', 0.0) * 100.0:.2f}%\n")

    # 2. Preprocess scan
    raw_points = load_point_cloud(args.scan)
    raw_labels = load_labels(args.label)

    prep = LidarPreprocessor(
        PreprocessingConfig(sampling=SamplingConfig(strategy="random", num_points=args.num_points, seed=1042))
    )
    processed = prep(raw_points, raw_labels)

    remapper = SemanticLabelRemapper()
    sih_labels = remapper.remap(processed.labels)

    # 3. Compute Metrics
    pts_tensor = torch.from_numpy(processed.points).unsqueeze(0).to(device).float()
    with torch.no_grad():
        logits = model(pts_tensor)  # [1, N, 4]
        preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

    metrics = SemanticSegmentationMetrics(num_classes=4, ignore_index=255)
    metrics.update(preds, sih_labels)
    report = metrics.compute()

    print(format_metric_report(report))

    # 4. Predictor Interface & Amit's Frozen Contract Verification
    predictor = PointNet2Predictor(model=model, device=device)
    result = predictor.predict(processed.points)

    out_xyz = result["xyz"]
    out_class = result["predicted_class"]
    out_conf = result["confidence"]

    xyz_ok = np.array_equal(processed.points[:, :3].astype(np.float32), out_xyz)
    class_ok = set(np.unique(out_class)).issubset({0, 1, 2, 3})
    conf_ok = bool((out_conf >= 0.0).all() and (out_conf <= 1.0).all())

    # Prediction distribution
    unique_preds, pred_counts = np.unique(out_class, return_counts=True)
    total_preds = len(out_class)

    print("\n3. Prediction Distribution & Confidence Statistics:")
    print("  " + "-" * 62)
    for c in range(4):
        cnt = int(pred_counts[unique_preds == c][0]) if c in unique_preds else 0
        pct = (cnt / total_preds) * 100.0
        print(f"  Class {c:d} ({SIH_CLASS_NAMES[c]:<20}) : {cnt:6,d} points ({pct:5.2f}%)")
    print("  " + "-" * 62)
    print(f"  Confidence Mean   : {out_conf.mean():.4f}")
    print(f"  Confidence Median : {np.median(out_conf):.4f}")
    print(f"  Confidence Min    : {out_conf.min():.4f}")
    print(f"  Confidence Max    : {out_conf.max():.4f}")

    print("\n4. Frozen ML -> Mapping Contract Verification:")
    print(f"  [{'PASS' if xyz_ok else 'FAIL'}] Exact 1-to-1 XYZ coordinates preserved")
    print(f"  [{'PASS' if class_ok else 'FAIL'}] Classes strictly in {{0, 1, 2, 3}}")
    print(f"  [{'PASS' if conf_ok else 'FAIL'}] Confidence strictly in [0.0, 1.0]")
    print("=" * 68)

    return 0 if (xyz_ok and class_ok and conf_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
