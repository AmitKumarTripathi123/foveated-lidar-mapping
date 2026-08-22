#!/usr/bin/env python3
"""Phase 7 Multi-Frame Experiment Comparison CLI Tool.

Usage:
    python scripts/compare_phase7.py --experiments phase7_baseline_ce phase7_weighted_ce phase7_weighted_ce_aug
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Compare Phase 7 Experiments.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["phase7_baseline_ce", "phase7_weighted_ce", "phase7_weighted_ce_aug"],
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
        best_miou = data.get("best_val_miou", 0.0) * 100.0

        # Check collapse heuristic: if 3 classes have 0% IoU and 1 class has >90% IoU/support
        ious = [
            final_rec.get("iou_drivable", 0.0),
            final_rec.get("iou_non_drivable", 0.0),
            final_rec.get("iou_static", 0.0),
            final_rec.get("iou_dynamic", 0.0),
        ]
        zero_count = sum(1 for v in ious if v == 0.0)
        is_collapsed = zero_count >= 3

        results.append({
            "name": exp_name,
            "best_epoch": data.get("best_epoch", "-"),
            "best_val_miou": best_miou,
            "val_loss": final_rec.get("val_loss", 0.0),
            "accuracy": final_rec.get("val_accuracy", 0.0) * 100.0,
            "iou_0": final_rec.get("iou_drivable", 0.0) * 100.0,
            "iou_1": final_rec.get("iou_non_drivable", 0.0) * 100.0,
            "iou_2": final_rec.get("iou_static", 0.0) * 100.0,
            "iou_3": final_rec.get("iou_dynamic", 0.0) * 100.0,
            "collapse": "YES (WARNING)" if is_collapsed else "NO",
        })

    if not results:
        print("No completed experiment records found.")
        return 1

    print("\n" + "=" * 106)
    print("                          PHASE 7 EXPERIMENT COMPARISON SUMMARY")
    print("=" * 106)
    header = (
        f"{'Experiment':<24} | {'Best Ep':>7} | {'Val Loss':>8} | {'Val mIoU':>8} | "
        f"{'Accuracy':>8} | {'IoU C0':>7} | {'IoU C1':>7} | {'IoU C2':>7} | {'IoU C3':>7} | {'Collapse':>13}"
    )
    print(header)
    print("-" * 106)

    for r in results:
        line = (
            f"{r['name']:<24} | {r['best_epoch']:7} | {r['val_loss']:8.4f} | {r['best_val_miou']:7.2f}% | "
            f"{r['accuracy']:7.2f}% | {r['iou_0']:6.2f}% | {r['iou_1']:6.2f}% | {r['iou_2']:6.2f}% | {r['iou_3']:6.2f}% | {r['collapse']:>13}"
        )
        print(line)

    print("=" * 106 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
