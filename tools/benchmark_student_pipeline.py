import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.types import SuperClass, PointCloudFrame
from src.data_loader import LiDARDataLoader
from src.range_filter import RangeFilter
from src.foveated_grid import FoveatedGrid25D
from phase2.dataset import remap_poss_labels
from phase2.models.spvcnn import SPVCNN, build_spvcnn
from phase2.models.spvcnn_adapter import SPVCNNInputAdapter, SPVCNNLabelAdapter
from phase2.inference.predictor import Phase2Predictor, SemanticPrediction
from phase2.metrics.semantic_evaluator import Phase2SemanticEvaluator

def benchmark_student_pipeline(
    student_ckpt="checkpoints/spvcnn_student_16ch.pt",
    device="cpu",
    voxel_size=0.08,
    num_runs=30
):
    print("=" * 80)
    print("  PHASE 10: END-TO-END 16-CHANNEL STUDENT SPVCNN PIPELINE BENCHMARK")
    print("=" * 80)

    scan_file = repo_root / "dataset/sequences/00/velodyne/000000.bin"
    loader = LiDARDataLoader(dataset_path=repo_root / "dataset", sequence_id="00")
    range_filter = RangeFilter(min_range=0.5, max_range=100.0)
    grid_engine = FoveatedGrid25D(use_cpp=True)

    # 1. Load Student Model
    dev = torch.device(device)
    student = SPVCNN(num_classes=4, in_channels=4, base_channels=16).to(dev)
    if Path(student_ckpt).exists():
        ckpt = torch.load(student_ckpt, map_location=dev)
        student.load_state_dict(ckpt.get("model_state_dict", ckpt))
        print(f"Loaded student weights from {student_ckpt}")
    student.eval()
    
    input_adapter = SPVCNNInputAdapter(voxel_size=voxel_size)
    label_adapter = SPVCNNLabelAdapter(native_source="sih_direct")

    # Warmup
    f_raw = loader.load_frame(scan_file)
    f_filt, _ = range_filter.filter_frame(f_raw)
    bundle = input_adapter.prepare_input(f_filt.points, device=device)
    for _ in range(10):
        with torch.inference_mode():
            _ = student(bundle["features"], bundle["point_to_voxel_idx"], bundle["num_voxels"])

    # 2. Benchmark Full End-to-End Execution
    times_load = []
    times_filt = []
    times_vox = []
    times_ml = []
    times_post = []
    times_grid = []
    times_total = []

    for _ in range(num_runs):
        t0 = time.perf_counter()
        f_raw = loader.load_frame(scan_file)
        t_load = time.perf_counter()

        f_filt, _ = range_filter.filter_frame(f_raw)
        t_filt = time.perf_counter()

        bundle = input_adapter.prepare_input(f_filt.points, device=device)
        t_vox = time.perf_counter()

        with torch.inference_mode():
            logits = student(bundle["features"], bundle["point_to_voxel_idx"], bundle["num_voxels"])
        t_ml = time.perf_counter()

        sih_p, s_prob, s_c = label_adapter.process_logits(logits)
        pred = SemanticPrediction(
            points=f_filt.points,
            predicted_class=sih_p,
            class_probabilities=s_prob,
            confidence=s_c,
            frame_id=f_filt.frame_id,
            sequence_id=f_filt.sequence_id
        )
        t_post = time.perf_counter()

        grid = grid_engine.build_grid(pred.points, pred.predicted_class, pred.confidence)
        t_grid = time.perf_counter()

        times_load.append((t_load - t0) * 1000)
        times_filt.append((t_filt - t_load) * 1000)
        times_vox.append((t_vox - t_filt) * 1000)
        times_ml.append((t_ml - t_vox) * 1000)
        times_post.append((t_post - t_ml) * 1000)
        times_grid.append((t_grid - t_post) * 1000)
        times_total.append((t_grid - t0) * 1000)

    mean_load = float(np.mean(times_load))
    mean_filt = float(np.mean(times_filt))
    mean_vox = float(np.mean(times_vox))
    mean_ml = float(np.mean(times_ml))
    mean_post = float(np.mean(times_post))
    mean_grid = float(np.mean(times_grid))
    mean_total = float(np.mean(times_total))
    p95_total = float(np.percentile(times_total, 95))
    fps = float(1000.0 / mean_total)

    # 3. Evaluate Semantic Accuracy on Sequence 00 Validation Scans
    seq_path = repo_root / "dataset/sequences/00"
    velo_dir = seq_path / "velodyne"
    label_dir = seq_path / "labels"
    scan_files = sorted(velo_dir.glob("*.bin"))[:10]
    evaluator = Phase2SemanticEvaluator(num_classes=4)

    all_preds, all_targs = [], []
    for sf in scan_files:
        lf = label_dir / f"{sf.stem}.label"
        if not lf.exists(): continue
        raw_pts = np.fromfile(str(sf), dtype=np.float32).reshape(-1, 4)
        raw_lbls = np.fromfile(str(lf), dtype=np.uint32) & 0xFFFF
        raw_f = PointCloudFrame(points=raw_pts, labels=raw_lbls)
        filt_f, _ = range_filter.filter_frame(raw_f)

        gt_sih = remap_poss_labels(filt_f.labels)
        b = input_adapter.prepare_input(filt_f.points, device=device)
        with torch.inference_mode():
            l = student(b["features"], b["point_to_voxel_idx"], b["num_voxels"])
        probs = torch.softmax(l, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)

        all_preds.append(preds)
        all_targs.append(gt_sih)

    preds_cat = np.concatenate(all_preds)
    targs_cat = np.concatenate(all_targs)
    metrics = evaluator.evaluate(preds_cat, targs_cat)

    print("\n" + "=" * 80)
    print("  STUDENT PIPELINE LATENCY PROFILE (CPU, voxel=0.08m, 16-channel SPVCNN)")
    print("=" * 80)
    print(f"1. Input Loading:              {mean_load:6.2f} ms")
    print(f"2. Range Filtering:            {mean_filt:6.2f} ms")
    print(f"3. 3D Voxelization (0.08m):    {mean_vox:6.2f} ms")
    print(f"4. 16-Ch SPVCNN Forward:       {mean_ml:6.2f} ms  (vs 127.17 ms Teacher -> {127.17/mean_ml:.2f}x Speedup!)")
    print(f"5. Semantic Postprocessing:    {mean_post:6.2f} ms")
    print(f"6. C++ Foveated Grid Engine:   {mean_grid:6.2f} ms")
    print("--------------------------------------------------------------------------------")
    print(f"TOTAL END-TO-END LATENCY:      {mean_total:6.2f} ms (P95: {p95_total:.2f} ms)")
    print(f"PIPELINE THROUGHPUT:           {fps:6.2f} FPS")
    print("--------------------------------------------------------------------------------")
    print(f"VALIDATION ACCURACY (OA):      {metrics['overall_accuracy']*100:6.2f}%")
    print(f"VALIDATION mIoU:               {metrics['mIoU']*100:6.2f}%")
    print(f"  - Static Obstacle IoU:       {metrics['static_obstacle_IoU']*100:6.2f}%")
    print(f"  - Dynamic Object IoU:        {metrics['dynamic_object_IoU']*100:6.2f}%")
    print(f"  - Non-Drivable Terrain IoU:  {metrics['non_drivable_terrain_IoU']*100:6.2f}%")
    print(f"  - Drivable Terrain IoU:      {metrics['drivable_terrain_IoU']*100:6.2f}%")
    print("--------------------------------------------------------------------------------")
    target_met = "ACHIEVED (<50 ms)" if mean_total < 50.0 else "NOT YET (<50 ms target requires further subsampling or GPU)"
    print(f"<50 ms REAL-TIME TARGET:       {target_met}")
    print("=" * 80)

if __name__ == "__main__":
    benchmark_student_pipeline()
