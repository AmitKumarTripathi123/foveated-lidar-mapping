import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.types import SuperClass, PointCloudFrame
from src.range_filter import RangeFilter
from phase2.dataset import remap_poss_labels, SEMANTICPOSS_TO_PROJECT
from phase2.models.spvcnn import SPVCNN, build_spvcnn, load_spvcnn_checkpoint
from phase2.models.spvcnn_adapter import SPVCNNInputAdapter, SEMANTICKITTI_TO_SIH
from phase2.inference.predictor import Phase2Predictor

def evaluate_both_mappings(checkpoint_path="checkpoints/best_spvcnn.pt", num_frames=10):
    seq_path = repo_root / "dataset/sequences/00"
    velo_dir = seq_path / "velodyne"
    label_dir = seq_path / "labels"
    scan_files = sorted(velo_dir.glob("*.bin"))[:num_frames]

    predictor = Phase2Predictor(model_path=checkpoint_path, device="cpu", voxel_size=0.05)
    range_filter = RangeFilter(min_range=0.5, max_range=100.0)

    # 1. Evaluate with INCORRECT SemanticKITTI mapping (used mistakenly in Phase 9 script)
    # 2. Evaluate with CORRECT SemanticPOSS mapping (SEMANTICPOSS_TO_PROJECT)

    for mode in ["semantickitti_incorrect", "semanticposs_correct"]:
        total_intersection = np.zeros(4, dtype=np.int64)
        total_union = np.zeros(4, dtype=np.int64)
        total_correct = 0
        total_valid = 0

        for scan_file in scan_files:
            lbl_file = label_dir / f"{scan_file.stem}.label"
            raw_points = np.fromfile(str(scan_file), dtype=np.float32).reshape(-1, 4)
            raw_labels = np.fromfile(str(lbl_file), dtype=np.uint32) & 0xFFFF

            raw_frame = PointCloudFrame(points=raw_points, labels=raw_labels)
            filt_frame, _ = range_filter.filter_frame(raw_frame)

            pts_filt = filt_frame.points
            lbls_raw = filt_frame.labels

            if mode == "semantickitti_incorrect":
                gt_sih = np.full(len(lbls_raw), SuperClass.IGNORE_LABEL, dtype=np.int64)
                for native_c, sih_c in SEMANTICKITTI_TO_SIH.items():
                    gt_sih[lbls_raw == native_c] = sih_c
            else:
                gt_sih = remap_poss_labels(lbls_raw)

            pred = predictor.predict_frame(filt_frame)
            pred_sih = pred.predicted_class

            valid = (gt_sih != SuperClass.IGNORE_LABEL) & (gt_sih < 4)
            v_gt = gt_sih[valid]
            v_pred = pred_sih[valid]

            total_correct += np.sum(v_gt == v_pred)
            total_valid += len(v_gt)

            for c in range(4):
                gt_c = (v_gt == c)
                pred_c = (v_pred == c)
                total_intersection[c] += np.sum(gt_c & pred_c)
                total_union[c] += np.sum(gt_c | pred_c)

        ious = np.zeros(4)
        for c in range(4):
            ious[c] = total_intersection[c] / max(total_union[c], 1)
        miou = np.mean(ious) * 100.0
        acc = (total_correct / max(total_valid, 1)) * 100.0

        print("=" * 70)
        print(f"EVALUATION MODE: {mode}")
        print("=" * 70)
        print(f"Valid points evaluated: {total_valid}")
        print(f"Per-Class IoU:")
        print(f"  Class 0 (Drivable):     {ious[0]*100:6.2f}%")
        print(f"  Class 1 (Non-Drivable): {ious[1]*100:6.2f}%")
        print(f"  Class 2 (Static Obs):   {ious[2]*100:6.2f}%")
        print(f"  Class 3 (Dynamic Obj):  {ious[3]*100:6.2f}%")
        print(f"Mean IoU (mIoU):          {miou:6.2f}%")
        print(f"Overall Accuracy:         {acc:6.2f}%")

if __name__ == "__main__":
    evaluate_both_mappings()
