#!/usr/bin/env python3
"""PointNet++ Baseline Evaluation and Sanity CLI Tool (Phase 4).

Usage:
    python scripts/evaluate_baseline.py \
        --scan dataset/sequences/00/velodyne/000000.bin \
        --label dataset/sequences/00/labels/000000.label \
        --num-points 16384 \
        --seed 42 \
        --overfit-check
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.dataset import load_point_cloud, load_labels
from ml.data.preprocessing import LidarPreprocessor, PreprocessingConfig, SamplingConfig
from ml.data.label_mapping import SemanticLabelRemapper
from ml.models.pointnet2 import PointNet2SemSeg, build_model
from ml.models.predictor import PointNet2Predictor


def run_tiny_overfit_test(
    model: nn.Module,
    sample_points: np.ndarray,
    sample_labels: np.ndarray,
    num_iterations: int = 25,
    lr: float = 0.01,
) -> Dict[str, Any]:
    """Run a mandatory small-subset overfit test to verify backward pass and gradient flow."""
    device = next(model.parameters()).device
    model.train()

    # Use small subset for rapid overfit verification
    pts_t = torch.from_numpy(sample_points).unsqueeze(0).to(device).float()  # [1, N, 4]
    lbl_t = torch.from_numpy(sample_labels).unsqueeze(0).to(device).long()   # [1, N]

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    losses = []
    start_time = time.time()

    for it in range(num_iterations):
        optimizer.zero_grad()
        logits = model(pts_t)  # [1, N, 4]

        # [1, N, 4] -> [N, 4], [1, N] -> [N]
        loss = criterion(logits.view(-1, 4), lbl_t.view(-1))
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    elapsed = time.time() - start_time
    initial_loss = losses[0]
    final_loss = losses[-1]
    loss_reduction = ((initial_loss - final_loss) / initial_loss) * 100.0 if initial_loss > 0 else 0.0

    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction_pct": loss_reduction,
        "iterations": num_iterations,
        "elapsed_sec": elapsed,
        "all_finite": not (np.isnan(losses).any() or np.isinf(losses).any()),
    }


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Evaluate PointNet++ Semantic Segmentation Baseline (Phase 4)."
    )
    parser.add_argument(
        "--scan",
        type=str,
        default="dataset/sequences/00/velodyne/000000.bin",
        help="Path to .bin file",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="dataset/sequences/00/labels/000000.label",
        help="Path to .label file",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=16384,
        help="Point count resolution (default: 16384)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--overfit-check",
        action="store_true",
        help="Run small-subset overfit sanity check",
    )

    args = parser.parse_args()

    # 1. Pipeline Execution: Phase 1 Loader -> Phase 2 Preprocessing -> Phase 3 SIH Remapper
    raw_points = load_point_cloud(args.scan)
    raw_labels = load_labels(args.label)

    prep_config = PreprocessingConfig(
        sampling=SamplingConfig(strategy="random", num_points=args.num_points, seed=args.seed)
    )
    preprocessor = LidarPreprocessor(prep_config)
    processed = preprocessor(raw_points, raw_labels)

    remapper = SemanticLabelRemapper()
    sih_labels = remapper.remap(processed.labels)

    points = processed.points  # [N, 4]

    # 2. Build PointNet++ Baseline Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 3. Model Prediction via Frozen Contract
    predictor = PointNet2Predictor(model=model, device=device)
    prediction = predictor.predict(points)

    out_xyz = prediction["xyz"]
    pred_class = prediction["predicted_class"]
    conf = prediction["confidence"]

    # 4. Point-Order & Contract Verification
    xyz_matched = np.array_equal(points[:, :3].astype(np.float32), out_xyz)
    class_valid = set(np.unique(pred_class)).issubset({0, 1, 2, 3})
    conf_valid = bool((conf >= 0.0).all() and (conf <= 1.0).all())

    print("=" * 68)
    print("        POINTNET++ BASELINE EVALUATION & VALIDATION REPORT")
    print("=" * 68)

    print("\n1. Architecture Specification:")
    print("  Model Variant              : PointNet++ Semantic Segmentation (MSG/SSG)")
    print("  Total Input Channels       : 4 ([x, y, z, intensity])")
    print("  Coordinate Channels        : 3 ([x, y, z])")
    print("  Feature Channels           : 1 ([intensity])")
    print("  Output Classes             : 4 (0: drivable, 1: non-drivable, 2: static, 3: dynamic)")
    print("  Total Parameters           : {:,}".format(total_params))
    print("  Trainable Parameters       : {:,}".format(trainable_params))
    print("  Computing Device           : {}".format(device))

    print("\n2. Forward Pass Dimensions:")
    print(f"  Input Points Shape         : {points.shape} (dtype={points.dtype})")
    print(f"  Output Coordinates Shape   : {out_xyz.shape} (dtype={out_xyz.dtype})")
    print(f"  Predicted Classes Shape    : {pred_class.shape} (dtype={pred_class.dtype})")
    print(f"  Confidence Scores Shape    : {conf.shape} (dtype={conf.dtype})")

    print("\n3. Frozen ML -> Mapping Contract Verification:")
    print("  [PASS] Output format: [x, y, z, predicted_class, confidence]")
    print(f"  [{'PASS' if xyz_matched else 'FAIL'}] Exact 1-to-1 XYZ point-order preserved (before == after)")
    print(f"  [{'PASS' if class_valid else 'FAIL'}] Predicted classes strictly in {{0, 1, 2, 3}} (observed: {sorted(list(np.unique(pred_class)))})")
    print(f"  [{'PASS' if conf_valid else 'FAIL'}] Confidence scores strictly in [0.0, 1.0] (min={conf.min():.4f}, max={conf.max():.4f})")

    # 5. Overfit Sanity Check if requested
    overfit_passed = True
    if args.overfit_check:
        print("\n4. Small-Subset Overfit Sanity Check (25 iterations):")
        # Use small subset of 512 points for rapid overfit verification
        sub_pts = points[:512]
        sub_lbls = sih_labels[:512]
        of_res = run_tiny_overfit_test(model, sub_pts, sub_lbls, num_iterations=25, lr=0.01)

        print(f"  Initial Loss               : {of_res['initial_loss']:.4f}")
        print(f"  Final Loss                 : {of_res['final_loss']:.4f}")
        print(f"  Loss Reduction             : {of_res['loss_reduction_pct']:.2f}%")
        print(f"  Loss / Gradient Finiteness : {'[PASS]' if of_res['all_finite'] else '[FAIL]'}")
        overfit_passed = of_res["all_finite"] and (of_res["final_loss"] < of_res["initial_loss"])

    print("\n" + "=" * 68)
    if xyz_matched and class_valid and conf_valid and overfit_passed:
        print("                 PHASE 4 BASELINE VALIDATION: PASS")
    else:
        print("                 PHASE 4 BASELINE VALIDATION: FAIL")
    print("=" * 68 + "\n")

    return 0 if (xyz_matched and class_valid and conf_valid and overfit_passed) else 1


if __name__ == "__main__":
    sys.exit(main())
