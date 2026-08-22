import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import time
import torch
import numpy as np
from src.types import PointCloudFrame
from src.data_loader import LiDARDataLoader
from src.range_filter import RangeFilter
from src.foveated_grid import FoveatedGrid25D
from phase2.models.spvcnn import SPVCNN
from phase2.models.spvcnn_adapter import SPVCNNInputAdapter, SPVCNNLabelAdapter
from phase2.inference.predictor import SemanticPrediction

scan_file = Path("dataset/sequences/00/velodyne/000000.bin")
loader = LiDARDataLoader(dataset_path="dataset", sequence_id="00")
range_filter = RangeFilter(min_range=0.5, max_range=100.0)
grid_engine = FoveatedGrid25D(use_cpp=True)

f_raw = loader.load_frame(scan_file)
f_filt, _ = range_filter.filter_frame(f_raw)
pts = f_filt.points

has_mps = torch.backends.mps.is_available()

for dev_name in (["cpu", "mps"] if has_mps else ["cpu"]):
    for ratio in [1.0, 0.5]:
        n_s = int(len(pts) * ratio)
        idx_s = np.linspace(0, len(pts) - 1, n_s, dtype=np.int64)
        sub_pts = pts[idx_s]

        student = SPVCNN(num_classes=4, in_channels=4, base_channels=16).to(dev_name)
        ckpt = torch.load("checkpoints/spvcnn_student_16ch.pt", map_location=dev_name)
        student.load_state_dict(ckpt.get("model_state_dict", ckpt))
        student.eval()

        input_adapter = SPVCNNInputAdapter(voxel_size=0.10)
        label_adapter = SPVCNNLabelAdapter(native_source="sih_direct")

        # Warmup
        b = input_adapter.prepare_input(sub_pts, device=dev_name)
        for _ in range(5):
            with torch.inference_mode():
                _ = student(b["features"], b["point_to_voxel_idx"], b["num_voxels"])
        if dev_name == "mps": torch.mps.synchronize()

        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            f_raw = loader.load_frame(scan_file)
            f_filt, _ = range_filter.filter_frame(f_raw)
            sub_pts = f_filt.points[idx_s]

            b = input_adapter.prepare_input(sub_pts, device=dev_name)
            with torch.inference_mode():
                logits = student(b["features"], b["point_to_voxel_idx"], b["num_voxels"])
            if dev_name == "mps": torch.mps.synchronize()

            sih_p, s_prob, s_c = label_adapter.process_logits(logits)
            nn_idx = np.searchsorted(idx_s, np.arange(len(f_filt.points)))
            nn_idx = np.clip(nn_idx, 0, len(idx_s) - 1)
            full_sih = sih_p[nn_idx]
            full_conf = s_c[nn_idx]

            pred = SemanticPrediction(
                points=f_filt.points,
                predicted_class=full_sih,
                class_probabilities=np.zeros((len(f_filt.points), 4), dtype=np.float32),
                confidence=full_conf
            )
            g = grid_engine.build_grid(pred.points, pred.predicted_class, pred.confidence)
            times.append((time.perf_counter() - t0) * 1000)

        mean_ms = float(np.mean(times))
        p95_ms = float(np.percentile(times, 95))
        fps = 1000.0 / mean_ms
        print(f"Device: {dev_name.upper():4s} | Sampling: {ratio*100:3.0f}% ({n_s:5d} pts) | Total Latency: {mean_ms:6.2f} ms (P95: {p95_ms:5.2f} ms) | {fps:5.1f} FPS")

