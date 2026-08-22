#!/usr/bin/env python3
"""Phase 9 Experiment Comparison and Model Selection Tool.

Compares validation metrics across training strategies and selects the best
checkpoint based strictly on Validation mIoU while checking collapse diagnostics.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def load_experiment_metrics(exp_dir: Path) -> Dict[str, Any]:
    """Load metrics.json from an experiment directory."""
    metrics_path = exp_dir / "metrics.json"
    if not metrics_path.is_file():
        return {}
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_experiments(experiments_root: Path) -> None:
    """Compare all experiment runs and print a summary table."""
    exp_dirs = [d for d in experiments_root.iterdir() if d.is_dir()]
    if not exp_dirs:
        print(f"No experiment directories found in {experiments_root}")
        return

    results: List[Dict[str, Any]] = []

    for d in sorted(exp_dirs):
        m = load_experiment_metrics(d)
        if not m:
            continue
        results.append(m)

    if not results:
        print("No valid metrics.json files found.")
        return

    # Print Table
    print("\n" + "=" * 95)
    print("                    PHASE 9 EXPERIMENT COMPARISON SUMMARY")
    print("=" * 95)
    print(
        f"{'Experiment':<24} | {'Best Ep':<7} | {'Val Loss':<8} | {'Accuracy':<8} | "
        f"{'mIoU':<7} | {'IoU0':<6} | {'IoU1':<6} | {'IoU2':<6} | {'IoU3':<6} | {'Collapse':<8}"
    )
    print("-" * 95)

    best_exp = None
    best_miou = -1.0

    for r in results:
        name = r.get("experiment_name", "unknown")
        ep = r.get("best_epoch", 0)
        v_loss = r.get("best_val_loss", 0.0)
        acc = r.get("best_val_accuracy", 0.0) * 100.0
        miou = r.get("best_val_miou", 0.0) * 100.0
        ious = r.get("best_class_ious", {})
        iou0 = ious.get("0", 0.0) * 100.0
        iou1 = ious.get("1", 0.0) * 100.0
        iou2 = ious.get("2", 0.0) * 100.0
        iou3 = ious.get("3", 0.0) * 100.0
        collapse = "YES" if r.get("model_collapse_warning", False) else "NO"

        if miou > best_miou:
            best_miou = miou
            best_exp = name

        print(
            f"{name:<24} | {ep:<7} | {v_loss:<8.4f} | {acc:<7.2f}% | "
            f"{miou:<6.2f}% | {iou0:<5.1f}% | {iou1:<5.1f}% | {iou2:<5.1f}% | {iou3:<5.1f}% | {collapse:<8}"
        )

    print("=" * 95)
    print(f"BEST EXPERIMENT (by Val mIoU): {best_exp} ({best_miou:.2f}% mIoU)\n")


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Compare Phase 9 Training Experiments.")
    parser.add_argument(
        "--experiments-dir",
        type=str,
        default="experiments",
        help="Directory containing experiment subfolders.",
    )
    args = parser.parse_args()
    exp_path = Path(args.experiments_dir)
    if not exp_path.is_dir():
        print(f"Experiments directory not found: {exp_path}")
        return 1

    compare_experiments(exp_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
