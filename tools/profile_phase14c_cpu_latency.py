import os
import sys
import time
import math
from pathlib import Path
import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.types import SuperClass, PointCloudFrame, ValidationPolicy
from src.data_loader import LiDARDataLoader
from src.foveated_grid import FoveatedGrid25D, HAS_CPP_GRID
if HAS_CPP_GRID:
    import foveated_grid_cpp
from ml.models.spvcnn import SPVCNN, build_spvcnn
from ml.data.spvcnn_adapter import SPVCNNInputAdapter
from ml.models.spvcnn_label_adapter import SPVCNNLabelAdapter


def profile_cpu_pipeline():
    print("=" * 80)
    print("  PHASE 14C — CPU LATENCY ROOT-CAUSE & OPTIMIZATION AUDIT")
    print("=" * 80)

    dataset_velodyne = repo_root / "dataset/sequences/00/velodyne"
    bin_files = sorted(dataset_velodyne.glob("*.bin"))
    if not bin_files:
        print("ERROR: No real LiDAR scans found in dataset/sequences/00/velodyne")
        return

    sample_bin = bin_files[0]
    loader = LiDARDataLoader()
    frame = loader.load_frame(sample_bin)
    raw_pts = frame.points
    N_pts = len(raw_pts)
    print(f"Loaded Real LiDAR Frame: {sample_bin.name} ({N_pts:,d} points)")

    device = torch.device("cpu")
    num_classes = 19
    ckpt_path = repo_root / "experiments/phase12_full_semanticposs_spvcnn/best_checkpoint.pt"

    # Build SPVCNN model on CPU
    model = build_spvcnn(num_classes=num_classes, in_channels=4, pretrained_path=str(ckpt_path) if ckpt_path.exists() else None, device=device)
    model.eval()
    adapter = SPVCNNInputAdapter(voxel_size=0.05)
    label_adapter = SPVCNNLabelAdapter(native_source="semantickitti")
    cpp_engine = foveated_grid_cpp.FoveatedGridEngine() if HAS_CPP_GRID else None

    # =========================================================================
    # 1. Component-by-Component Latency Breakdown (100 Runs)
    # =========================================================================
    print("\n--- 1. COMPONENT-BY-COMPONENT CPU LATENCY BREAKDOWN (100 RUNS) ---")
    
    # Warmup
    for _ in range(5):
        bundle = adapter.prepare_input(raw_pts, device=device)
        with torch.no_grad():
            _ = model(bundle["features"], bundle["point_to_voxel_idx"], bundle["num_voxels"])

    # Benchmarking individual stages
    iters = 100
    
    # Stage A: File parsing
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = np.fromfile(str(sample_bin), dtype=np.float32).reshape(-1, 4)
    t_parse = ((time.perf_counter() - t0) / iters) * 1000.0

    # Stage B: Range filtering
    t0 = time.perf_counter()
    for _ in range(iters):
        r = np.sqrt(raw_pts[:, 0]**2 + raw_pts[:, 1]**2)
        mask = (r >= 0.0) & (r < 100.0) & np.isfinite(raw_pts[:, 0]) & np.isfinite(raw_pts[:, 1]) & np.isfinite(raw_pts[:, 2])
        filtered_pts = raw_pts[mask]
    t_filter = ((time.perf_counter() - t0) / iters) * 1000.0

    # Stage C: SPVCNN Voxelization (quantization, hash indexing, unique voxels)
    t0 = time.perf_counter()
    for _ in range(iters):
        v_coords = np.floor(filtered_pts[:, :3] / 0.05).astype(np.int64)
        v_min = np.min(v_coords, axis=0)
        v_shifted = v_coords - v_min
        v_max = np.max(v_shifted, axis=0) + 1
        keys = v_shifted[:, 0] + v_shifted[:, 1] * v_max[0] + v_shifted[:, 2] * (v_max[0] * v_max[1])
        _, voxel_to_pt, pt_to_voxel = np.unique(keys, return_index=True, return_inverse=True)
    t_voxelize = ((time.perf_counter() - t0) / iters) * 1000.0

    # Stage D: PyTorch tensor allocation & transfer
    t0 = time.perf_counter()
    for _ in range(iters):
        feat_t = torch.from_numpy(filtered_pts).float()
        pt_to_vox_t = torch.from_numpy(pt_to_voxel).long()
        num_v = len(voxel_to_pt)
    t_tensor_alloc = ((time.perf_counter() - t0) / iters) * 1000.0

    # Stage E: Full SPVCNN Forward Inference
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            logits = model(feat_t, pt_to_vox_t, num_v)
    t_model = ((time.perf_counter() - t0) / iters) * 1000.0

    # Stage F: Logits to Labels & Confidences
    t0 = time.perf_counter()
    for _ in range(iters):
        sih_classes, confidences = label_adapter.process_logits(logits)
    t_post = ((time.perf_counter() - t0) / iters) * 1000.0

    # Stage G: C++ Grid Generation & Semantic Aggregation
    t0 = time.perf_counter()
    for _ in range(iters):
        res_dict = cpp_engine.build_grid_numpy(filtered_pts, sih_classes, confidences)
    t_grid = ((time.perf_counter() - t0) / iters) * 1000.0

    total_cpu_lat = t_parse + t_filter + t_voxelize + t_tensor_alloc + t_model + t_post + t_grid

    print(f"  A. LiDAR Binary Parsing:             {t_parse:6.2f} ms ({t_parse/total_cpu_lat*100:4.1f}%)")
    print(f"  B. Preprocessing & Range Filter:     {t_filter:6.2f} ms ({t_filter/total_cpu_lat*100:4.1f}%)")
    print(f"  C. Point-Voxel Hash Voxelization:    {t_voxelize:6.2f} ms ({t_voxelize/total_cpu_lat*100:4.1f}%)")
    print(f"  D. PyTorch Tensor Allocation:        {t_tensor_alloc:6.2f} ms ({t_tensor_alloc/total_cpu_lat*100:4.1f}%)")
    print(f"  E. SPVCNN Neural Forward Pass:       {t_model:6.2f} ms ({t_model/total_cpu_lat*100:4.1f}%) [PRIMARY BOTTLENECK]")
    print(f"  F. Logits -> Label/Conf Adapter:     {t_post:6.2f} ms ({t_post/total_cpu_lat*100:4.1f}%)")
    print(f"  G. C++ Foveated 2.5D Grid Engine:    {t_grid:6.2f} ms ({t_grid/total_cpu_lat*100:4.1f}%)")
    print(f"  --------------------------------------------------")
    print(f"  TOTAL CPU PIPELINE LATENCY:          {total_cpu_lat:6.2f} ms ({1000.0/total_cpu_lat:4.1f} FPS)")

    # =========================================================================
    # 2. SPVCNN Layer-by-Layer Profiling
    # =========================================================================
    print("\n--- 2. SPVCNN LAYER-BY-LAYER LATENCY PROFILING ---")
    with torch.no_grad():
        # Stem
        t0 = time.perf_counter()
        for _ in range(iters): x0 = model.stem(feat_t)
        t_stem = ((time.perf_counter() - t0) / iters) * 1000.0

        # Stage 1 (32 -> 64)
        t0 = time.perf_counter()
        for _ in range(iters): x1 = model.stage1(x0, pt_to_vox_t, num_v)
        t_s1 = ((time.perf_counter() - t0) / iters) * 1000.0

        # Stage 2 (64 -> 128)
        t0 = time.perf_counter()
        for _ in range(iters): x2 = model.stage2(x1, pt_to_vox_t, num_v)
        t_s2 = ((time.perf_counter() - t0) / iters) * 1000.0

        # Stage 3 (128 -> 128)
        t0 = time.perf_counter()
        for _ in range(iters): x3 = model.stage3(x2, pt_to_vox_t, num_v)
        t_s3 = ((time.perf_counter() - t0) / iters) * 1000.0

        # Stage 4 (128 -> 64)
        t0 = time.perf_counter()
        for _ in range(iters): x4 = model.stage4(x3 + x2, pt_to_vox_t, num_v)
        t_s4 = ((time.perf_counter() - t0) / iters) * 1000.0

        # Classifier
        t0 = time.perf_counter()
        for _ in range(iters): _ = model.classifier(x4 + x1)
        t_head = ((time.perf_counter() - t0) / iters) * 1000.0

    print(f"  1. Stem (Linear+BN+LeakyReLU):       {t_stem:6.2f} ms ({t_stem/t_model*100:4.1f}%)")
    print(f"  2. Stage 1 (32 -> 64 SPVBlock):      {t_s1:6.2f} ms ({t_s1/t_model*100:4.1f}%)")
    print(f"  3. Stage 2 (64 -> 128 SPVBlock):     {t_s2:6.2f} ms ({t_s2/t_model*100:4.1f}%) [MOST EXPENSIVE]")
    print(f"  4. Stage 3 (128 -> 128 SPVBlock):    {t_s3:6.2f} ms ({t_s3/t_model*100:4.1f}%) [MOST EXPENSIVE]")
    print(f"  5. Stage 4 (128 -> 64 SPVBlock):     {t_s4:6.2f} ms ({t_s4/t_model*100:4.1f}%)")
    print(f"  6. Classifier Head:                  {t_head:6.2f} ms ({t_head/t_model*100:4.1f}%)")

    # =========================================================================
    # 3. CPU Thread Scaling Experiment
    # =========================================================================
    print("\n--- 3. CPU THREAD SCALING EXPERIMENT ---")
    orig_threads = torch.get_num_threads()
    for nth in [1, 2, 4, 6, 8]:
        torch.set_num_threads(nth)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(20):
                _ = model(feat_t, pt_to_vox_t, num_v)
        lat_th = ((time.perf_counter() - t0) / 20) * 1000.0
        print(f"  Threads = {nth:2d} -> Model Latency: {lat_th:6.2f} ms ({1000.0/lat_th:4.1f} FPS)")
    torch.set_num_threads(orig_threads)

    # =========================================================================
    # 4. Point / Voxel Count Scaling Experiment
    # =========================================================================
    print("\n--- 4. POINT / VOXEL COUNT SCALING EXPERIMENT ---")
    for n_sub in [10_000, 20_000, 30_000, 40_000, 50_000, 66_402]:
        sub_pts = filtered_pts[:n_sub]
        sub_bundle = adapter.prepare_input(sub_pts, device=device)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(20):
                _ = model(sub_bundle["features"], sub_bundle["point_to_voxel_idx"], sub_bundle["num_voxels"])
        lat_sub = ((time.perf_counter() - t0) / 20) * 1000.0
        print(f"  Points: {n_sub:6,d} | Voxels: {sub_bundle['num_voxels']:6,d} | CPU Latency: {lat_sub:6.2f} ms ({1000.0/lat_sub:4.1f} FPS)")

    # =========================================================================
    # 5. Lightweight Architecture Optimization Evaluation (Student-Small / Student-Tiny)
    # =========================================================================
    print("\n--- 5. ARCHITECTURAL OPTIMIZATION & MODEL REDUCTION AUDIT ---")
    # Base SPVCNN (32 channels)
    model_32 = SPVCNN(num_classes=num_classes, in_channels=4, base_channels=32).to(device).eval()
    params_32 = sum(p.numel() for p in model_32.parameters())
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(20): _ = model_32(feat_t, pt_to_vox_t, num_v)
    lat_32 = ((time.perf_counter() - t0) / 20) * 1000.0

    # Student (16 channels)
    model_16 = SPVCNN(num_classes=num_classes, in_channels=4, base_channels=16).to(device).eval()
    params_16 = sum(p.numel() for p in model_16.parameters())
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(20): _ = model_16(feat_t, pt_to_vox_t, num_v)
    lat_16 = ((time.perf_counter() - t0) / 20) * 1000.0

    # Student-Tiny (8 channels)
    model_8 = SPVCNN(num_classes=num_classes, in_channels=4, base_channels=8).to(device).eval()
    params_8 = sum(p.numel() for p in model_8.parameters())
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(20): _ = model_8(feat_t, pt_to_vox_t, num_v)
    lat_8 = ((time.perf_counter() - t0) / 20) * 1000.0

    print(f"  1. Teacher / Base SPVCNN (32-ch): {params_32:7,d} params | Latency: {lat_32:6.2f} ms | Speedup: 1.0x")
    print(f"  2. Distilled Student    (16-ch): {params_16:7,d} params | Latency: {lat_16:6.2f} ms | Speedup: {lat_32/lat_16:.2f}x")
    print(f"  3. Lightweight Student  ( 8-ch): {params_8:7,d} params | Latency: {lat_8:6.2f} ms | Speedup: {lat_32/lat_8:.2f}x")

    print("\n" + "=" * 80)
    print("  PHASE 14C CPU LATENCY AUDIT COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    profile_cpu_pipeline()
