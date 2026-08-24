import os
import sys
import time
import math
import hashlib
from pathlib import Path
import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.types import SuperClass, PointCloudFrame
from src.data_loader import LiDARDataLoader
from src.foveated_grid import FoveatedGrid25D, HAS_CPP_GRID
if HAS_CPP_GRID:
    import foveated_grid_cpp
from ml.models.spvcnn import SPVCNN, build_spvcnn
from ml.data.spvcnn_adapter import SPVCNNInputAdapter
from ml.models.spvcnn_label_adapter import SPVCNNLabelAdapter
from phase2.metrics.semantic_evaluator import Phase2SemanticEvaluator


def run_phase14d_audit():
    print("=" * 80)
    print("  PHASE 14D — CPU STUDENT LATENCY / QUALITY TRADE-OFF & MODEL SELECTION")
    print("=" * 80)

    # 1. Freeze Evaluation Protocol
    torch.set_num_threads(6)
    device = torch.device("cpu")
    evaluator = Phase2SemanticEvaluator(num_classes=4, ignore_label=SuperClass.IGNORE_LABEL)
    label_adapter = SPVCNNLabelAdapter(native_source="semanticposs")
    input_adapter = SPVCNNInputAdapter(voxel_size=0.05)
    cpp_engine = foveated_grid_cpp.FoveatedGridEngine() if HAS_CPP_GRID else None

    dataset_velodyne = repo_root / "dataset/sequences/00/velodyne"
    dataset_labels = repo_root / "dataset/sequences/00/labels"
    bin_files = sorted(dataset_velodyne.glob("*.bin"))
    lbl_files = sorted(dataset_labels.glob("*.label")) if dataset_labels.exists() else []

    num_eval_frames = min(50, len(bin_files))
    print(f"Frozen Protocol: Dataset=SemanticPOSS Seq00 | Frames={num_eval_frames} | Threads=6 | Device=CPU")

    loader = LiDARDataLoader()
    eval_data = []
    for idx in range(num_eval_frames):
        bf = bin_files[idx]
        lf = dataset_labels / f"{bf.stem}.label" if dataset_labels.exists() else None
        frm = loader.load_frame(bf, lf)
        m_lbls = label_adapter.remap_predictions(frm.labels)
        
        # Pre-filter range [0, 100)m
        r = np.sqrt(frm.points[:, 0]**2 + frm.points[:, 1]**2)
        valid_mask = (r >= 0.0) & (r < 100.0) & np.isfinite(frm.points[:, 0]) & np.isfinite(frm.points[:, 1]) & np.isfinite(frm.points[:, 2])
        pts_filt = frm.points[valid_mask]
        lbls_filt = m_lbls[valid_mask]
        bundle = input_adapter.prepare_input(pts_filt, device=device)
        eval_data.append((pts_filt, lbls_filt, bundle))

    print(f"Loaded and preprocessed {len(eval_data)} validation frames.")

    # 2. Model Channel Sweep
    channel_configs = [32, 16, 14, 12, 10, 8, 6]
    results = []

    print("\n" + "-" * 80)
    print(f"{'Channels':>8} | {'Params':>8} | {'Latency':>8} | {'P50':>7} | {'P95':>7} | {'P99':>7} | {'FPS':>5} | {'mIoU':>7} | {'Acc':>7} | {'Dyn IoU':>7} | {'Obs IoU':>7}")
    print("-" * 80)

    for ch in channel_configs:
        model = SPVCNN(num_classes=4, in_channels=4, base_channels=ch).to(device).eval()
        params = sum(p.numel() for p in model.parameters())

        # Measure latency on 100 iterations of sample frame
        sample_bundle = eval_data[0][2]
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(sample_bundle["features"], sample_bundle["point_to_voxel_idx"], sample_bundle["num_voxels"])

        latencies = []
        with torch.no_grad():
            for _ in range(30):
                t0 = time.perf_counter()
                logits = model(sample_bundle["features"], sample_bundle["point_to_voxel_idx"], sample_bundle["num_voxels"])
                # Grid call
                preds = torch.argmax(logits, dim=-1).numpy()
                confs = np.full(len(preds), 0.9, dtype=np.float32)
                _ = cpp_engine.build_grid_numpy(eval_data[0][0], preds, confs)
                dt = (time.perf_counter() - t0) * 1000.0
                latencies.append(dt)

        lat_arr = np.array(latencies)
        mean_lat = np.mean(lat_arr)
        p50 = np.percentile(lat_arr, 50)
        p95 = np.percentile(lat_arr, 95)
        p99 = np.percentile(lat_arr, 99)
        fps = 1000.0 / mean_lat

        # Evaluate Semantic Quality over validation frames
        all_preds = []
        all_targs = []
        with torch.no_grad():
            for pts, targs, bnd in eval_data:
                lg = model(bnd["features"], bnd["point_to_voxel_idx"], bnd["num_voxels"])
                pr = torch.argmax(lg, dim=-1).numpy()
                all_preds.append(pr)
                all_targs.append(targs)

        concat_preds = np.concatenate(all_preds)
        concat_targs = np.concatenate(all_targs)
        metrics = evaluator.evaluate(concat_preds, concat_targs)

        miou = metrics["mIoU"] * 100.0
        acc = metrics["overall_accuracy"] * 100.0
        dyn_iou = metrics["dynamic_object_IoU"] * 100.0
        obs_iou = metrics["static_obstacle_IoU"] * 100.0

        res_dict = {
            "channels": ch,
            "params": params,
            "mean_lat": mean_lat,
            "p50": p50,
            "p95": p95,
            "p99": p99,
            "fps": fps,
            "miou": miou,
            "acc": acc,
            "dyn_iou": dyn_iou,
            "obs_iou": obs_iou,
            "metrics": metrics,
            "meets_50ms": (p95 < 50.0)
        }
        results.append(res_dict)

        print(f"{ch:8d} | {params:8,d} | {mean_lat:7.2f}m | {p50:6.2f}m | {p95:6.2f}m | {p99:6.2f}m | {fps:5.1f} | {miou:6.1f}% | {acc:6.1f}% | {dyn_iou:6.1f}% | {obs_iou:6.1f}%")

    print("-" * 80)

    # 3. Pareto Frontier Analysis
    print("\n--- 3. PARETO FRONTIER & SELECTION ---")
    for r in results:
        status = "REJECT — TOO SLOW (>50ms)" if not r["meets_50ms"] else ("CANDIDATE (Meets <50ms)" if r["channels"] >= 8 else "REJECT — QUALITY TOO LOW")
        print(f"  {r['channels']:2d}-Channel: Latency={r['p95']:.2f}ms (P95), mIoU={r['miou']:.1f}%, Dyn IoU={r['dyn_iou']:.1f}% -> {status}")

    print("\n" + "=" * 80)
    print("  PHASE 14D AUDIT COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    run_phase14d_audit()
