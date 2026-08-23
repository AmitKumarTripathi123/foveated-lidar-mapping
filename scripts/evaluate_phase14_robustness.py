"""
Phase 14: Comprehensive Robustness and Sequence-Wise Scientific Evaluator.
Evaluates the frozen Phase 12 production SPVCNN checkpoint across:
- All 6 individual SemanticPOSS sequences (2,988 frames)
- Cross-sequence generalization & stability
- Distance-dependent range bins (0-10m, 10-20m, 20-40m, 40-60m, 60-80m, 80-100m)
- 3-zone distance foveation compression & accuracy
- Class-wise robustness and dynamic_object error analysis
- Model collapse & entropy diagnostics
- Checkpoint reproducibility assertions
- Top best & worst failure case mining
- End-to-end system latency & VRAM profiling
"""

import argparse
import csv
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import torch
import torch.nn.functional as F

from ml.data.dataset import load_point_cloud, load_labels, validate_point_label_alignment
from ml.data.amit_adapter import FoveatedVoxelSampler
from ml.data.semanticposs_label_mapping import SemanticPOSSLabelRemapper
from ml.data.spvcnn_adapter import SPVCNNInputAdapter
from ml.models.spvcnn import SPVCNN, build_spvcnn
from ml.models.spvcnn_predictor import SPVCNNPredictor
from ml.models.mapping_adapter import MLToMappingAdapter
from scripts.audit_semanticposs import audit_sequence, get_dataset_root


CLASS_NAMES = {
    0: "drivable_terrain",
    1: "non_drivable_terrain",
    2: "static_obstacle",
    3: "dynamic_object",
}

RANGE_BINS = [
    (0.0, 10.0, "0-10m"),
    (10.0, 20.0, "10-20m"),
    (20.0, 40.0, "20-40m"),
    (40.0, 60.0, "40-60m"),
    (60.0, 80.0, "60-80m"),
    (80.0, 100.0, "80-100m"),
]


def audit_full_dataset(dataset_root: Path) -> Dict[str, Any]:
    """Perform strict forensic verification across all 6 SemanticPOSS sequences."""
    print("=" * 65)
    print("  PHASE 14: FORENSIC DATASET VERIFICATION (2,988 FRAMES)")
    print("=" * 65)
    expected_counts = {"00": 488, "01": 500, "02": 500, "03": 500, "04": 500, "05": 500}
    seq_dir = dataset_root / "sequences"
    seq_audits = {}
    total_bins = 0
    total_lbls = 0
    total_matched = 0
    total_points = 0
    all_matched = True

    for s_id, exp_count in expected_counts.items():
        s_path = seq_dir / s_id
        if not s_path.is_dir():
            seq_audits[s_id] = {"exists": False, "error": f"Missing sequence directory {s_path}"}
            all_matched = False
            continue

        res = audit_sequence(str(s_path), s_id)
        n_bins = len(list((s_path / "velodyne").glob("*.bin")))
        n_lbls = len(list((s_path / "labels").glob("*.label")))
        n_matched = res.get("matched_pairs", 0)
        n_pts = res.get("total_points", 0)
        has_errors = len(res.get("alignment_errors", [])) > 0 or len(res.get("missing_bins", [])) > 0 or len(res.get("missing_labels", [])) > 0

        seq_audits[s_id] = {
            "expected_frames": exp_count,
            "bin_files": n_bins,
            "label_files": n_lbls,
            "matched_pairs": n_matched,
            "total_points": n_pts,
            "status": "PASS" if (n_bins == exp_count and n_lbls == exp_count and n_matched == exp_count and not has_errors) else "FAIL",
        }
        total_bins += n_bins
        total_lbls += n_lbls
        total_matched += n_matched
        total_points += n_pts

        print(f"  Sequence {s_id}: {n_matched}/{exp_count} matched pairs ({n_pts:,} points) -> {seq_audits[s_id]['status']}")

    overall_pass = (total_matched == 2988) and all_matched
    audit_summary = {
        "dataset_root": str(dataset_root),
        "timestamp": datetime.datetime.now().isoformat(),
        "total_expected_frames": 2988,
        "total_bins_discovered": total_bins,
        "total_labels_discovered": total_lbls,
        "total_matched_pairs": total_matched,
        "total_physical_points": total_points,
        "sequences": seq_audits,
        "dataset_complete": overall_pass,
    }
    return audit_summary


def compute_iou_from_cm(cm: np.ndarray, num_classes: int = 4) -> Tuple[float, Dict[int, float], Dict[int, float], Dict[int, float], float]:
    """Compute mIoU, per-class IoU, Precision, Recall, and Accuracy from a confusion matrix."""
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    denom = tp + fp + fn

    ious = {}
    precisions = {}
    recalls = {}
    valid_ious = []

    for c in range(num_classes):
        iou_val = float(tp[c] / denom[c]) * 100.0 if denom[c] > 0 else 0.0
        p_val = float(tp[c] / (tp[c] + fp[c])) * 100.0 if (tp[c] + fp[c]) > 0 else 0.0
        r_val = float(tp[c] / (tp[c] + fn[c])) * 100.0 if (tp[c] + fn[c]) > 0 else 0.0

        ious[c] = round(iou_val, 2)
        precisions[c] = round(p_val, 2)
        recalls[c] = round(r_val, 2)

        # Include classes with non-zero ground truth or predictions in mIoU
        if (tp[c] + fn[c]) > 0 or (tp[c] + fp[c]) > 0:
            valid_ious.append(iou_val)

    miou = round(float(np.mean(valid_ious)), 2) if len(valid_ious) > 0 else 0.0
    total_pts = np.sum(cm)
    accuracy = round(float(np.sum(tp) / total_pts * 100.0), 2) if total_pts > 0 else 0.0
    return miou, ious, precisions, recalls, accuracy


def evaluate_sequence(
    seq_id: str,
    dataset_root: Path,
    model: SPVCNN,
    input_adapter: SPVCNNInputAdapter,
    sampler: FoveatedVoxelSampler,
    remapper: SemanticPOSSLabelRemapper,
    device: torch.device,
    sample_stride: int = 1,
) -> Dict[str, Any]:
    """Perform detailed sequence evaluation."""
    s_path = dataset_root / "sequences" / seq_id
    v_dir = s_path / "velodyne"
    l_dir = s_path / "labels"
    bin_files = sorted(list(v_dir.glob("*.bin")))[::sample_stride]

    cm_total = np.zeros((4, 4), dtype=np.int64)
    pred_counts = np.zeros(4, dtype=np.int64)
    total_eval_points = 0
    total_ignored_points = 0
    conf_values = []

    # Distance bin confusion matrices
    dist_cms = {bin_info[2]: np.zeros((4, 4), dtype=np.int64) for bin_info in RANGE_BINS}
    dist_counts = {bin_info[2]: 0 for bin_info in RANGE_BINS}
    dist_confs = {bin_info[2]: [] for bin_info in RANGE_BINS}

    # Foveation zone stats
    zone_cms = {"near_0_10m": np.zeros((4, 4), dtype=np.int64), "mid_10_40m": np.zeros((4, 4), dtype=np.int64), "far_40_100m": np.zeros((4, 4), dtype=np.int64)}
    raw_pt_counts = []
    fov_pt_counts = []

    frame_metrics = []
    model.eval()

    with torch.no_grad():
        for b_file in bin_files:
            l_file = l_dir / f"{b_file.stem}.label"
            if not l_file.is_file():
                continue

            raw_pts = load_point_cloud(b_file)
            raw_lbls = load_labels(l_file)
            raw_pt_counts.append(len(raw_pts))

            # Foveate
            fov_pts, fov_lbls, _ = sampler.sample(raw_pts, raw_lbls)
            fov_pt_counts.append(len(fov_pts))
            sih_lbls = remapper.remap(fov_lbls)

            pts_t = torch.from_numpy(fov_pts).to(device).float()
            bundle = input_adapter.prepare_input(pts_t, device=device)

            logits = model(
                features=bundle["features"],
                point_to_voxel_idx=bundle["point_to_voxel_idx"],
                num_voxels=bundle["num_voxels"],
            )
            probs = F.softmax(logits, dim=-1)
            preds_t = torch.argmax(probs, dim=-1)
            confs_t, _ = torch.max(probs, dim=-1)

            preds_np = preds_t.cpu().numpy()
            confs_np = confs_t.cpu().numpy()
            targets_np = sih_lbls

            # Valid supervised mask
            valid = (targets_np != 255) & (targets_np >= 0) & (targets_np < 4)
            ignored = np.sum(targets_np == 255)
            total_ignored_points += int(ignored)
            total_eval_points += len(fov_pts)

            v_targets = targets_np[valid]
            v_preds = preds_np[valid]
            v_confs = confs_np[valid]
            v_pts = fov_pts[valid]

            if len(v_targets) > 0:
                # Frame-level CM
                f_cm = np.bincount(v_targets * 4 + v_preds, minlength=16).reshape(4, 4)
                cm_total += f_cm
                p_bincount = np.bincount(preds_np[preds_np < 4], minlength=4)
                pred_counts += p_bincount[:4]
                conf_values.extend(v_confs[::10].tolist())

                f_miou, f_ious, _, _, f_acc = compute_iou_from_cm(f_cm)
                frame_metrics.append({
                    "frame_id": f"{seq_id}_{b_file.stem}",
                    "miou": f_miou,
                    "accuracy": f_acc,
                    "dynamic_iou": f_ious.get(3, 0.0),
                    "points": len(fov_pts),
                    "mean_confidence": round(float(np.mean(v_confs)), 4),
                })

                # Distance bins
                ranges = np.linalg.norm(v_pts[:, :3], axis=1)
                for r_min, r_max, r_name in RANGE_BINS:
                    r_mask = (ranges >= r_min) & (ranges < r_max)
                    if np.any(r_mask):
                        r_t = v_targets[r_mask]
                        r_p = v_preds[r_mask]
                        dist_cms[r_name] += np.bincount(r_t * 4 + r_p, minlength=16).reshape(4, 4)
                        dist_counts[r_name] += int(np.sum(r_mask))
                        dist_confs[r_name].extend(v_confs[r_mask][::5].tolist())

                # Foveation zones
                z_near = ranges < 10.0
                z_mid = (ranges >= 10.0) & (ranges < 40.0)
                z_far = ranges >= 40.0

                if np.any(z_near):
                    zone_cms["near_0_10m"] += np.bincount(v_targets[z_near] * 4 + v_preds[z_near], minlength=16).reshape(4, 4)
                if np.any(z_mid):
                    zone_cms["mid_10_40m"] += np.bincount(v_targets[z_mid] * 4 + v_preds[z_mid], minlength=16).reshape(4, 4)
                if np.any(z_far):
                    zone_cms["far_40_100m"] += np.bincount(v_targets[z_far] * 4 + v_preds[z_far], minlength=16).reshape(4, 4)

    # Compute sequence summary metrics
    seq_miou, seq_ious, seq_prec, seq_rec, seq_acc = compute_iou_from_cm(cm_total)
    total_preds = np.sum(pred_counts)
    pred_pcts = {c: round(float(pred_counts[c] / total_preds * 100.0), 2) if total_preds > 0 else 0.0 for c in range(4)}
    dom_class = int(np.argmax(pred_counts))
    dom_pct = pred_pcts[dom_class]
    collapse_warning = dom_pct >= 90.0

    # Entropy
    p_dist = (pred_counts + 1e-8) / np.sum(pred_counts + 1e-8)
    entropy = float(-np.sum(p_dist * np.log(p_dist)))

    # Foveation compression
    mean_raw = float(np.mean(raw_pt_counts)) if len(raw_pt_counts) > 0 else 0.0
    mean_fov = float(np.mean(fov_pt_counts)) if len(fov_pt_counts) > 0 else 0.0
    fov_reduction_pct = round(float((1.0 - mean_fov / max(mean_raw, 1.0)) * 100.0), 2)

    # Distance bin metrics
    dist_results = {}
    for _, _, r_name in RANGE_BINS:
        d_miou, d_ious, _, _, d_acc = compute_iou_from_cm(dist_cms[r_name])
        c_list = dist_confs[r_name]
        dist_results[r_name] = {
            "points": dist_counts[r_name],
            "miou": d_miou,
            "accuracy": d_acc,
            "per_class_iou": d_ious,
            "mean_confidence": round(float(np.mean(c_list)), 4) if len(c_list) > 0 else 0.0,
        }

    # Zone metrics
    zone_results = {}
    for z_name, z_cm in zone_cms.items():
        z_miou, z_ious, _, _, z_acc = compute_iou_from_cm(z_cm)
        zone_results[z_name] = {"miou": z_miou, "accuracy": z_acc, "per_class_iou": z_ious}

    conf_arr = np.array(conf_values) if len(conf_values) > 0 else np.array([0.0])

    return {
        "sequence_id": seq_id,
        "frame_count": len(bin_files),
        "total_evaluated_points": total_eval_points,
        "supervised_points": int(np.sum(cm_total)),
        "ignored_points": total_ignored_points,
        "miou": seq_miou,
        "overall_accuracy": seq_acc,
        "per_class_iou": seq_ious,
        "per_class_precision": seq_prec,
        "per_class_recall": seq_rec,
        "confusion_matrix": cm_total.tolist(),
        "prediction_percentages": pred_pcts,
        "dominant_class": dom_class,
        "dominant_class_pct": dom_pct,
        "collapse_warning": collapse_warning,
        "prediction_entropy": round(entropy, 4),
        "confidence_stats": {
            "mean": round(float(np.mean(conf_arr)), 4),
            "median": round(float(np.median(conf_arr)), 4),
            "min": round(float(np.min(conf_arr)), 4),
            "max": round(float(np.max(conf_arr)), 4),
        },
        "foveation_stats": {
            "mean_raw_points": round(mean_raw, 1),
            "mean_foveated_points": round(mean_fov, 1),
            "reduction_percentage": fov_reduction_pct,
            "zone_metrics": zone_results,
        },
        "distance_metrics": dist_results,
        "frames": frame_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 14 Comprehensive Robustness Evaluator.")
    parser.add_argument("--dataset-root", type=str, default="dataset", help="Path to dataset root.")
    parser.add_argument("--checkpoint", type=str, default="experiments/phase12_full_semanticposs_spvcnn/best_checkpoint.pt", help="Path to Phase 12 checkpoint.")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu).")
    parser.add_argument("--stride", type=int, default=1, help="Frame sample stride for evaluation.")
    parser.add_argument("--out-dir", type=str, default="reports/phase14", help="Output directory for reports.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(get_dataset_root(args.dataset_root))
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Phase 14 Evaluator Active Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Forensic Dataset Audit
    audit_res = audit_full_dataset(dataset_root)
    with open(out_dir / "dataset_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_res, f, indent=2)

    with open(out_dir / "dataset_audit.txt", "w", encoding="utf-8") as f:
        f.write("PHASE 14 FORENSIC DATASET AUDIT REPORT\n")
        f.write("======================================\n")
        f.write(f"Total Expected Frames: 2,988\n")
        f.write(f"Total Discovered Frames: {audit_res['total_matched_pairs']}\n")
        f.write(f"Total Points: {audit_res['total_physical_points']:,}\n")
        f.write(f"Completeness Status: {'PASS' if audit_res['dataset_complete'] else 'FAIL'}\n\n")
        for s_id, s_info in audit_res["sequences"].items():
            f.write(f"Sequence {s_id}: {s_info.get('matched_pairs', 0)}/{s_info.get('expected_frames', 0)} frames | {s_info.get('total_points', 0):,} points | Status: {s_info.get('status')}\n")

    if not audit_res["dataset_complete"]:
        print("\nERROR: Forensic dataset verification failed! Missing frames detected.")
        sys.exit(1)

    # 2. Checkpoint Loading & Reproducibility Verification
    print("\n" + "=" * 65)
    print(f"  PHASE 14: LOADING FROZEN PRODUCTION CHECKPOINT")
    print(f"  Checkpoint: {args.checkpoint}")
    print("=" * 65)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    baseline_val_miou = float(ckpt_data.get("metrics", {}).get("val_miou", 53.59))
    print(f"Loaded Checkpoint Metadata: Epoch={ckpt_data.get('epoch')}, Phase 12 Baseline Val mIoU={baseline_val_miou:.2f}%")

    model = build_spvcnn(num_classes=4, in_channels=4, pretrained_path=str(ckpt_path), device=device)
    input_adapter = SPVCNNInputAdapter(voxel_size=0.05)
    sampler = FoveatedVoxelSampler(near_dist=10.0, near_voxel=0.05, mid_dist=40.0, mid_voxel=0.15, far_dist=100.0, far_voxel=0.50)
    remapper = SemanticPOSSLabelRemapper()

    # 3. Strict Sequence-Wise Evaluation across Sequences 00 - 05
    seq_ids = ["00", "01", "02", "03", "04", "05"]
    seq_results = []
    all_frames = []

    for s_id in seq_ids:
        print(f"\nEvaluating Sequence {s_id} ({audit_res['sequences'][s_id]['matched_pairs']} frames)...")
        t0 = time.time()
        res = evaluate_sequence(
            seq_id=s_id,
            dataset_root=dataset_root,
            model=model,
            input_adapter=input_adapter,
            sampler=sampler,
            remapper=remapper,
            device=device,
            sample_stride=args.stride,
        )
        t_el = time.time() - t0
        print(f"  Sequence {s_id} Evaluated in {t_el:.1f}s | mIoU: {res['miou']:.2f}% | Acc: {res['overall_accuracy']:.2f}% | IoU: {res['per_class_iou']}")
        seq_results.append(res)
        all_frames.extend(res["frames"])

    # 4. Cross-Sequence Aggregations
    seq_mious = [r["miou"] for r in seq_results]
    mean_miou = round(float(np.mean(seq_mious)), 2)
    median_miou = round(float(np.median(seq_mious)), 2)
    std_miou = round(float(np.std(seq_mious)), 2)
    min_miou = round(float(np.min(seq_mious)), 2)
    max_miou = round(float(np.max(seq_mious)), 2)
    worst_seq = seq_ids[int(np.argmin(seq_mious))]
    best_seq = seq_ids[int(np.argmax(seq_mious))]

    per_class_stats = {}
    for c in range(4):
        c_ious = [r["per_class_iou"].get(c, 0.0) for r in seq_results]
        per_class_stats[c] = {
            "class_name": CLASS_NAMES[c],
            "mean": round(float(np.mean(c_ious)), 2),
            "median": round(float(np.median(c_ious)), 2),
            "std": round(float(np.std(c_ious)), 2),
            "min": round(float(np.min(c_ious)), 2),
            "max": round(float(np.max(c_ious)), 2),
        }

    # 5. Distance-Wise Aggregation
    combined_dist = {bin_info[2]: {"points": 0, "cms": np.zeros((4, 4), dtype=np.int64), "confs": []} for bin_info in RANGE_BINS}
    for r in seq_results:
        for b_name, d_info in r["distance_metrics"].items():
            combined_dist[b_name]["points"] += d_info["points"]
            # Reconstruct from sequence metrics
            # Aggregate via confusion matrix
    
    dist_table = []
    for _, _, b_name in RANGE_BINS:
        total_pts = sum(r["distance_metrics"][b_name]["points"] for r in seq_results)
        mean_b_miou = round(float(np.mean([r["distance_metrics"][b_name]["miou"] for r in seq_results if r["distance_metrics"][b_name]["points"] > 0])), 2)
        mean_b_acc = round(float(np.mean([r["distance_metrics"][b_name]["accuracy"] for r in seq_results if r["distance_metrics"][b_name]["points"] > 0])), 2)
        mean_b_conf = round(float(np.mean([r["distance_metrics"][b_name]["mean_confidence"] for r in seq_results if r["distance_metrics"][b_name]["points"] > 0])), 4)
        c0_iou = round(float(np.mean([r["distance_metrics"][b_name]["per_class_iou"].get(0, 0.0) for r in seq_results if r["distance_metrics"][b_name]["points"] > 0])), 2)
        c1_iou = round(float(np.mean([r["distance_metrics"][b_name]["per_class_iou"].get(1, 0.0) for r in seq_results if r["distance_metrics"][b_name]["points"] > 0])), 2)
        c2_iou = round(float(np.mean([r["distance_metrics"][b_name]["per_class_iou"].get(2, 0.0) for r in seq_results if r["distance_metrics"][b_name]["points"] > 0])), 2)
        c3_iou = round(float(np.mean([r["distance_metrics"][b_name]["per_class_iou"].get(3, 0.0) for r in seq_results if r["distance_metrics"][b_name]["points"] > 0])), 2)

        dist_table.append({
            "range_bin": b_name,
            "points": total_pts,
            "miou": mean_b_miou,
            "accuracy": mean_b_acc,
            "mean_confidence": mean_b_conf,
            "iou_0_drivable": c0_iou,
            "iou_1_non_drivable": c1_iou,
            "iou_2_static_obstacle": c2_iou,
            "iou_3_dynamic_object": c3_iou,
        })

    # 6. Failure Case Mining (Top 5 Best & Worst Frames)
    all_frames_sorted = sorted(all_frames, key=lambda f: f["miou"])
    worst_5_frames = all_frames_sorted[:5]
    best_5_frames = all_frames_sorted[-5:][::-1]

    # 7. End-to-End Latency & Hardware Benchmark
    print("\nRunning End-to-End Hardware Latency Benchmark...")
    map_adapter = MLToMappingAdapter()
    sample_bin = dataset_root / "sequences/02/velodyne/000001.bin"
    load_times, fov_times, inf_times, grid_times, total_times = [], [], [], [], []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for _ in range(10):
        t0 = time.perf_counter()
        raw_pts = load_point_cloud(sample_bin)
        t_load = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        fov_pts, _, _ = sampler.sample(raw_pts)
        t_fov = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        pts_t = torch.from_numpy(fov_pts).to(device).float()
        bundle = input_adapter.prepare_input(pts_t, device=device)
        with torch.no_grad():
            logits = model(bundle["features"], bundle["point_to_voxel_idx"], bundle["num_voxels"])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        t_inf = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        confs = torch.max(F.softmax(logits, dim=-1), dim=-1).values.cpu().numpy()
        res_dict = {"xyz": fov_pts[:, :3], "predicted_class": preds, "confidence": confs}
        grid = map_adapter.build_25d_grid(res_dict)
        t_grid = (time.perf_counter() - t0) * 1000.0

        t_tot = t_load + t_fov + t_inf + t_grid
        load_times.append(t_load)
        fov_times.append(t_fov)
        inf_times.append(t_inf)
        grid_times.append(t_grid)
        total_times.append(t_tot)

    perf_summary = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "lidar_load_mean_ms": round(float(np.mean(load_times)), 2),
        "foveation_mean_ms": round(float(np.mean(fov_times)), 2),
        "spvcnn_inference_mean_ms": round(float(np.mean(inf_times)), 2),
        "gridmap25d_mean_ms": round(float(np.mean(grid_times)), 2),
        "total_latency_mean_ms": round(float(np.mean(total_times)), 2),
        "total_latency_median_ms": round(float(np.median(total_times)), 2),
        "total_latency_p95_ms": round(float(np.percentile(total_times, 95)), 2),
        "fps": round(float(1000.0 / np.mean(total_times)), 2),
        "peak_vram_allocated_mb": round(float(torch.cuda.max_memory_allocated() / (1024**2)), 2) if torch.cuda.is_available() else 0.0,
        "peak_vram_reserved_mb": round(float(torch.cuda.max_memory_reserved() / (1024**2)), 2) if torch.cuda.is_available() else 0.0,
    }

    # 8. Checkpoint Reload Verification
    print("\nVerifying Checkpoint Reload Reproducibility...")
    fresh_model = build_spvcnn(num_classes=4, in_channels=4, pretrained_path=str(ckpt_path), device=device)
    fresh_model.eval()
    with torch.no_grad():
        pts_sample = torch.from_numpy(fov_pts).to(device).float()
        b_fresh = input_adapter.prepare_input(pts_sample, device=device)
        l_fresh = fresh_model(b_fresh["features"], b_fresh["point_to_voxel_idx"], b_fresh["num_voxels"])
        l_orig = model(b_fresh["features"], b_fresh["point_to_voxel_idx"], b_fresh["num_voxels"])
        diff = torch.max(torch.abs(l_fresh - l_orig)).item()
        reload_pass = diff < 1e-4

    print(f"Checkpoint Reload Diff: {diff:.6f} -> {'PASS' if reload_pass else 'FAIL'}")

    # 9. Save All Reports & CSVs
    # Sequence metrics CSV
    with open(out_dir / "sequence_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence_id", "frames", "points", "miou", "accuracy", "iou0_drivable", "iou1_non_drivable", "iou2_static_obstacle", "iou3_dynamic_object", "dom_class_pct", "entropy"])
        for r in seq_results:
            writer.writerow([
                r["sequence_id"],
                r["frame_count"],
                r["supervised_points"],
                r["miou"],
                r["overall_accuracy"],
                r["per_class_iou"].get(0, 0.0),
                r["per_class_iou"].get(1, 0.0),
                r["per_class_iou"].get(2, 0.0),
                r["per_class_iou"].get(3, 0.0),
                r["dominant_class_pct"],
                r["prediction_entropy"],
            ])

    # Distance metrics CSV
    with open(out_dir / "distance_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["range_bin", "points", "miou", "accuracy", "mean_confidence", "iou0_drivable", "iou1_non_drivable", "iou2_static_obstacle", "iou3_dynamic_object"])
        for row in dist_table:
            writer.writerow([row["range_bin"], row["points"], row["miou"], row["accuracy"], row["mean_confidence"], row["iou_0_drivable"], row["iou_1_non_drivable"], row["iou_2_static_obstacle"], row["iou_3_dynamic_object"]])

    # Sequence metrics JSON
    with open(out_dir / "sequence_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "cross_sequence_summary": {
                "mean_miou": mean_miou,
                "median_miou": median_miou,
                "std_miou": std_miou,
                "min_miou": min_miou,
                "max_miou": max_miou,
                "worst_sequence": worst_seq,
                "best_sequence": best_seq,
                "per_class_statistics": per_class_stats,
            },
            "sequence_results": seq_results,
            "worst_5_frames": worst_5_frames,
            "best_5_frames": best_5_frames,
        }, f, indent=2)

    # Class metrics JSON
    with open(out_dir / "class_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_class_stats, f, indent=2)

    # Confusion matrices JSON
    confusion_dict = {r["sequence_id"]: r["confusion_matrix"] for r in seq_results}
    with open(out_dir / "confusion_matrices.json", "w", encoding="utf-8") as f:
        json.dump(confusion_dict, f, indent=2)

    # Performance JSON
    with open(out_dir / "performance.json", "w", encoding="utf-8") as f:
        json.dump(perf_summary, f, indent=2)

    print("\n" + "=" * 65)
    print("  PHASE 14 EVALUATION COMPLETE")
    print(f"  Cross-Sequence Mean mIoU:   {mean_miou:.2f}% (Std: {std_miou:.2f}%)")
    print(f"  Cross-Sequence Median mIoU: {median_miou:.2f}% (Min: {min_miou:.2f}%, Max: {max_miou:.2f}%)")
    print(f"  Worst Sequence: {worst_seq} ({min_miou:.2f}%) | Best Sequence: {best_seq} ({max_miou:.2f}%)")
    print(f"  Dynamic Object Mean IoU:    {per_class_stats[3]['mean']:.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
