#!/usr/bin/env python3
"""Compare Multiple PointNet++ Training Experiments (Phase 5).

Usage:
    python scripts/compare_experiments.py --experiments baseline_ce weighted_ce weighted_ce_aug
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Compare PointNet++ Training Experiments.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["baseline_ce", "weighted_ce", "weighted_ce_aug"],
        help="Experiment directory names inside experiments/",
    )
    parser.add_argument(
        "--exp-root",
        type=str,
        default="experiments",
        help="Base experiments directory",
    )

    args = parser.parse_args()
    exp_root = Path(args.exp_root)

    results: List[Dict[str, Any]] = []

    for exp_name in args.experiments:
        metrics_file = exp_root / exp_name / "metrics.json"
        if not metrics_file.is_file():
            print(f"[!] metrics.json not found for experiment '{exp_name}' in {metrics_file.parent}")
            continue

        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        final_rec = data.get("final_epoch_record", {})
        results.append({
            "name": exp_name,
            "best_epoch": data.get("best_epoch", "-"),
            "best_val_miou": data.get("best_val_miou", 0.0) * 100.0,
            "accuracy": final_rec.get("val_accuracy", 0.0) * 100.0,
            "iou_drivable": final_rec.get("iou_drivable", 0.0) * 100.0,
            "iou_non_drivable": final_rec.get("iou_non_drivable", 0.0) * 100.0,
            "iou_static": final_rec.get("iou_static", 0.0) * 100.0,
            "iou_dynamic": final_rec.get("iou_dynamic", 0.0) * 100.0,
        })

    if not results:
        print("No completed experiment records found.")
        return 1

    print("\n" + "=" * 94)
    print("                    POINTNET++ EXPERIMENT COMPARISON SUMMARY")
    print("=" * 94)
    header = (
        f"{'Experiment':<18} | {'Best Ep':>7} | {'Val mIoU':>9} | {'Accuracy':>9} | "
        f"{'Drivable':>8} | {'Non-Drive':>9} | {'Static':>8} | {'Dynamic':>8}"
    )
    print(header)
    print("-" * 94)

    for r in results:
        line = (
            f"{r['name']:<18} | {r['best_epoch']:7} | {r['best_val_miou']:8.2f}% | {r['accuracy']:8.2f}% | "
            f"{r['iou_drivable']:7.2f}% | {r['iou_non_drivable']:8.2f}% | {r['iou_static']:7.2f}% | {r['iou_dynamic']:7.2f}%"
        )
        print(line)

    print("=" * 94 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
