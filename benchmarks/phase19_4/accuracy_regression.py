"""
Phase 19.4 Accuracy Regression Auditor (SIH PS 26130).
Validates that semantic mIoU remains exactly 52.04% across evaluation frames.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.dataset import load_point_cloud, load_labels
from src.core.range_filter import RangeFilter
from src.core.native_foveation import NativeFoveationAccelerator
from src.inference.predictor import CanonicalPredictor
from benchmarks.phase19_1.accuracy_audit import (
    compute_multiclass_metrics,
    update_confusion_matrix,
    remap_semanticposs_labels,
)


def audit_accuracy_regression(
    config_path: str = "configs/system_config.yaml",
    dataset_dir: str = "dataset/sequences/02",
    num_frames: int = 100,
    out_json: Path = Path("reports/phase19_4/accuracy_regression.json"),
) -> Dict[str, Any]:
    """Execute semantic accuracy audit and verify zero mIoU regression."""
    predictor = CanonicalPredictor(config_path)
    range_filter = RangeFilter(min_range=0.5, max_range=100.0)
    fov_sampler = NativeFoveationAccelerator()

    p = Path(dataset_dir)
    if (p / "velodyne").is_dir():
        seq_path = p
    elif p.name == "velodyne":
        seq_path = p.parent
    else:
        seq_path = repo_root / "dataset/sequences/02"

    bin_files = sorted(list((seq_path / "velodyne").glob("*.bin")))[:num_frames]
    lbl_files = sorted(list((seq_path / "labels").glob("*.label")))[:num_frames]

    global_cm = np.zeros((4, 4), dtype=np.int64)

    for i in range(len(bin_files)):
        raw_pts = load_point_cloud(bin_files[i])
        raw_lbls = load_labels(lbl_files[i])
        remapped_lbls = remap_semanticposs_labels(raw_lbls)

        pts_filtered, mask_filt = range_filter.filter(raw_pts)
        lbls_filtered = remapped_lbls[mask_filt]

        fov_pts, fov_targets, _ = fov_sampler.sample(pts_filtered, lbls_filtered)
        fov_preds, _ = predictor.predict(fov_pts)

        update_confusion_matrix(global_cm, fov_preds, fov_targets)

    metrics = compute_multiclass_metrics(global_cm)
    measured_miou = round(metrics["overall"]["miou"] * 100.0, 2)
    baseline_miou = 52.04
    drift = round(measured_miou - baseline_miou, 2)

    payload = {
        "status": "ACCURACY_PRESERVED" if abs(drift) <= 0.05 else "ACCURACY_REGRESSION_DETECTED",
        "phase19_2_miou_pct": baseline_miou,
        "phase19_4_miou_pct": measured_miou,
        "drift_pct": drift,
        "class_wise_iou_pct": {k: round(v["iou"] * 100.0, 2) for k, v in metrics["classes"].items()},
        "total_evaluated_points": metrics["overall"]["total_valid_points"],
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload


if __name__ == "__main__":
    res = audit_accuracy_regression()
    print(f"Accuracy Regression Audit: mIoU = {res['phase19_4_miou_pct']}% (Drift: {res['drift_pct']}%)")
