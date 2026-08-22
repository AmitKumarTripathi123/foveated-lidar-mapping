#!/usr/bin/env python3
"""scripts/infer_spvcnn.py

End-to-End SPVCNN Inference & 2.5D Mapping CLI Script (Phase 12).
Executes:
    Raw LiDAR -> Amit Foveation -> SPVCNN Adapter -> Pretrained SPVCNN
    -> SIH 4-Class Mapping -> [x,y,z,class,conf] -> MLToMappingAdapter -> GridMap25D
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch

# Add repository root to sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.dataset import load_point_cloud, load_labels
from ml.data.amit_adapter import FoveatedVoxelSampler
from ml.models.spvcnn_predictor import SPVCNNPredictor
from ml.models.mapping_adapter import MLToMappingAdapter


def run_inference(
    dataset_root: str,
    sequence: str,
    frame: str,
    checkpoint: str = None,
    device: str = None,
    voxel_size: float = 0.05,
    native_source: str = "semantickitti",
    output_dir: str = "reports/spvcnn_inference",
) -> dict:
    """Run full end-to-end SPVCNN perception and 2.5D mapping."""
    dev = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    seq_str = str(sequence).zfill(2)
    frame_str = str(frame).zfill(6)

    root = Path(dataset_root)
    velo_path = root / "sequences" / seq_str / "velodyne" / f"{frame_str}.bin"
    if not velo_path.exists():
        velo_path = root / seq_str / "velodyne" / f"{frame_str}.bin"

    if not velo_path.exists():
        raise FileNotFoundError(f"LiDAR scan not found at: {velo_path}")

    lbl_path = root / "sequences" / seq_str / "labels" / f"{frame_str}.label"
    if not lbl_path.exists():
        lbl_path = root / seq_str / "labels" / f"{frame_str}.label"

    print("==================================================")
    print("      SPVCNN Perception & 2.5D Mapping CLI        ")
    print("==================================================")
    print(f"Target Scan : {velo_path}")
    print(f"Device      : {dev}")
    print(f"Voxel Size  : {voxel_size} m")
    print(f"Checkpoint  : {checkpoint}\n")

    t0 = time.perf_counter()

    # 1. Load raw LiDAR
    t_load_start = time.perf_counter()
    raw_pts = load_point_cloud(velo_path)
    t_load = (time.perf_counter() - t_load_start) * 1000.0
    print(f"[1/6] Loaded {raw_pts.shape[0]:,} raw points ({t_load:.2f} ms)")

    # 2. Amit Foveated Preprocessing
    t_fov_start = time.perf_counter()
    sampler = FoveatedVoxelSampler(
        near_dist=10.0,
        near_voxel=0.05,
        mid_dist=40.0,
        mid_voxel=0.15,
        far_dist=100.0,
        far_voxel=0.50,
    )
    foveated_pts, _, _ = sampler.sample(raw_pts)
    t_fov = (time.perf_counter() - t_fov_start) * 1000.0
    print(f"[2/6] Foveated downsampling -> {foveated_pts.shape[0]:,} points ({t_fov:.2f} ms, -{100*(1-len(foveated_pts)/len(raw_pts)):.2f}%)")

    # 3. SPVCNN Predictor
    t_pred_start = time.perf_counter()
    predictor = SPVCNNPredictor(
        device=dev,
        voxel_size=voxel_size,
        native_source=native_source,
        pretrained_path=checkpoint if (checkpoint and os.path.isfile(checkpoint)) else None,
    )
    pred_res = predictor.predict(foveated_pts)
    t_pred = (time.perf_counter() - t_pred_start) * 1000.0
    print(f"[3/6] SPVCNN Inference & SIH Mapping completed ({t_pred:.2f} ms)")

    # 4. Validate ML Contract
    assert pred_res["xyz"].shape[0] == foveated_pts.shape[0]
    assert np.all(pred_res["confidence"] >= 0.0) and np.all(pred_res["confidence"] <= 1.0)
    print(f"[4/6] ML Output Contract Verified ([x, y, z, class, conf])")

    # 5. Mapping Adapter -> GridMap25D
    t_map_start = time.perf_counter()
    adapter = MLToMappingAdapter(
        bounds_x=(-50.0, 50.0),
        bounds_y=(-50.0, 50.0),
        resolution=0.20,
    )
    grid_map = adapter.build_25d_grid(pred_res)
    t_map = (time.perf_counter() - t_map_start) * 1000.0
    print(f"[5/6] 2.5D GridMap Generated ({t_map:.2f} ms)")

    t_total = (time.perf_counter() - t0) * 1000.0
    fps = 1000.0 / t_total if t_total > 0 else 0.0

    print(f"\n[6/6] Pipeline Execution Finished in {t_total:.2f} ms ({fps:.2f} FPS)")

    # Distribution audit
    u_cls, c_cls = np.unique(pred_res["predicted_class"], return_counts=True)
    class_dist = {int(k): int(v) for k, v in zip(u_cls, c_cls)}
    print(f"SIH Class Distribution: {class_dist}\n")

    summary = {
        "scan": str(velo_path),
        "device": dev,
        "input_points": int(raw_pts.shape[0]),
        "foveated_points": int(foveated_pts.shape[0]),
        "reduction_pct": round(100.0 * (1.0 - len(foveated_pts) / len(raw_pts)), 2),
        "latency_ms": {
            "load": round(t_load, 2),
            "foveation": round(t_fov, 2),
            "prediction": round(t_pred, 2),
            "mapping": round(t_map, 2),
            "total": round(t_total, 2),
        },
        "fps": round(fps, 2),
        "class_distribution": class_dist,
        "contract_verified": True,
    }

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    report_file = out_p / f"spvcnn_inference_{seq_str}_{frame_str}.json"
    with open(report_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved inference report to: {report_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="SPVCNN Inference CLI.")
    parser.add_argument("--dataset-root", type=str, default="dataset", help="Dataset root directory.")
    parser.add_argument("--sequence", type=str, default="00", help="Sequence ID (or 'all' for all sequences).")
    parser.add_argument("--frame", type=str, default="000000", help="Frame ID (or 'all' for all frames).")
    parser.add_argument("--all-frames", action="store_true", help="Process all 2,988 frames across all sequences.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/spvcnn_pretrained.pt", help="Path to SPVCNN checkpoint.")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu or cuda).")
    parser.add_argument("--voxel-size", type=float, default=0.05, help="Voxel size in meters.")
    parser.add_argument("--output-dir", type=str, default="reports/spvcnn_inference", help="Output directory.")

    args = parser.parse_args()

    root = Path(args.dataset_root)
    if args.all_frames or args.frame.lower() == "all" or args.sequence.lower() == "all":

        seqs = ["00", "01", "02", "03", "04", "05"] if (args.sequence.lower() == "all" or args.all_frames) else [args.sequence.zfill(2)]
        total_processed = 0
        for seq_id in seqs:
            velo_dir = root / "sequences" / seq_id / "velodyne"
            if not velo_dir.exists():
                velo_dir = root / seq_id / "velodyne"
            if not velo_dir.exists():
                continue
            bin_files = sorted(list(velo_dir.glob("*.bin")))
            print(f"\n=== Processing Sequence {seq_id} ({len(bin_files)} frames) ===")
            for b_file in bin_files:
                frame_stem = b_file.stem
                run_inference(
                    dataset_root=args.dataset_root,
                    sequence=seq_id,
                    frame=frame_stem,
                    checkpoint=args.checkpoint,
                    device=args.device,
                    voxel_size=args.voxel_size,
                    output_dir=args.output_dir,
                )
                total_processed += 1
        print(f"\n✅ Total Frames Processed: {total_processed:,}")
    else:
        run_inference(
            dataset_root=args.dataset_root,
            sequence=args.sequence,
            frame=args.frame,
            checkpoint=args.checkpoint,
            device=args.device,
            voxel_size=args.voxel_size,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()

