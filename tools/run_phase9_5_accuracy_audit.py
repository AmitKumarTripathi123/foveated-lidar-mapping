import os
import sys
import json
import hashlib
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
from phase2.dataset import remap_poss_labels, SEMANTICPOSS_TO_PROJECT
from phase2.models.spvcnn import SPVCNN, build_spvcnn, load_spvcnn_checkpoint
from phase2.models.spvcnn_adapter import SPVCNNInputAdapter, SPVCNNLabelAdapter, SEMANTICKITTI_TO_SIH
from phase2.inference.predictor import Phase2Predictor, SemanticPrediction
from phase2.metrics.semantic_evaluator import Phase2SemanticEvaluator

def main():
    print("=" * 80)
    print("  PHASE 9.5 — SEMANTICPOSS ACCURACY DISCREPANCY AUDIT ENGINE")
    print("=" * 80)

    # 1. Environment & Checkpoint Provenance
    ckpt_path = repo_root / "checkpoints/best_spvcnn.pt"
    ckpt_data = torch.load(str(ckpt_path), map_location="cpu")
    ckpt_bytes = open(ckpt_path, "rb").read()
    ckpt_sha256 = hashlib.sha256(ckpt_bytes).hexdigest()

    state = ckpt_data.get("model_state_dict", ckpt_data)
    num_params = sum(p.numel() for p in state.values() if isinstance(p, torch.Tensor))
    out_classes = state.get("classifier.4.weight", torch.empty((0,))).shape[0]

    print("\n--- STEP 1 & 3: CHECKPOINT & ENVIRONMENT AUDIT ---")
    print(f"Checkpoint Path:     {ckpt_path}")
    print(f"SHA-256 Hash:        {ckpt_sha256}")
    print(f"File Size:           {len(ckpt_bytes):,d} bytes")
    print(f"Stored Metadata:     Epoch={ckpt_data.get('epoch')}, val_mIoU={ckpt_data.get('val_miou')}, val_OA={ckpt_data.get('val_oa')}")
    print(f"Model Type:          {ckpt_data.get('model_type', 'N/A')}")
    print(f"Parameter Count:     {num_params:,d}")
    print(f"Classifier Output:   {out_classes} classes")

    # Verify loading into SPVCNN
    model = SPVCNN(num_classes=out_classes, in_channels=4, base_channels=32)
    load_report = load_spvcnn_checkpoint(model, ckpt_path, strict=True)
    print(f"Load Status:         Missing={len(load_report['missing_keys'])}, Unexpected={len(load_report['unexpected_keys'])} (Strict Match: PASS)")

    # 2. Dataset & Sequence Verification
    seq_path = repo_root / "dataset/sequences/00"
    velo_dir = seq_path / "velodyne"
    label_dir = seq_path / "labels"
    scan_files = sorted(velo_dir.glob("*.bin"))[:10]

    print("\n--- STEP 5 & 6: DATASET & RAW LABEL VALUES AUDIT ---")
    print(f"Dataset Path:        {seq_path}")
    print(f"Evaluated Scans:     {len(scan_files)} scans from Sequence 00")

    raw_labels_all = []
    for f in scan_files:
        l_f = label_dir / f"{f.stem}.label"
        if l_f.exists():
            lbl = np.fromfile(str(l_f), dtype=np.uint32) & 0xFFFF
            raw_labels_all.append(lbl)
    raw_labels_cat = np.concatenate(raw_labels_all)

    unique_raw, counts_raw = np.unique(raw_labels_cat, return_counts=True)
    print(f"Total Raw Points:    {len(raw_labels_cat):,d}")
    print(f"{'Raw ID':8s} | {'Count':10s} | {'Percentage':10s} | {'SemanticPOSS Meaning':25s} | {'Mapped SIH Superclass':22s}")
    print("-" * 85)
    for r_id, cnt in zip(unique_raw, counts_raw):
        pct = cnt / len(raw_labels_cat) * 100.0
        poss_meaning = {
            0: "unlabeled", 1: "outlier", 4: "person", 5: "two-wheelers", 6: "rider", 7: "car",
            8: "other-vehicle/truck", 9: "building", 10: "fence", 11: "other-structure",
            13: "pole", 14: "traffic-sign", 15: "cone", 16: "trashcan", 17: "vegetation",
            18: "trunk", 19: "terrain", 20: "other-ground", 21: "ground/road", 22: "outlier"
        }.get(r_id, "unknown")
        sih_cls = SEMANTICPOSS_TO_PROJECT.get(r_id, SuperClass.IGNORE_LABEL)
        sih_name = SuperClass.get_name(sih_cls)
        print(f"{r_id:8d} | {cnt:10,d} | {pct:9.2f}% | {poss_meaning:25s} | {sih_name:22s} ({sih_cls})")

    # 3. Controlled Accuracy Comparison: SemanticKITTI ontology mapping vs SemanticPOSS ontology mapping
    print("\n--- STEP 10, 11, 12: FORENSIC ONTOLOGY COMPARISON & CONFUSION MATRIX ---")
    
    predictor = Phase2Predictor(model_path=ckpt_path, device="cpu", voxel_size=0.05)
    range_filter = RangeFilter(min_range=0.5, max_range=100.0)
    evaluator = Phase2SemanticEvaluator()

    # Mode A: Phase 9 script bug (SEMANTICKITTI_TO_SIH)
    # Mode B: Correct Ground Truth (SEMANTICPOSS_TO_PROJECT)
    
    results_audit = {}

    for mode in ["Phase 9 Error (SemanticKITTI mapping on POSS data)", "Correct (SemanticPOSS authoritative mapping)"]:
        all_preds = []
        all_targets = []
        all_ranges = []

        for scan_file in scan_files:
            lbl_file = label_dir / f"{scan_file.stem}.label"
            raw_points = np.fromfile(str(scan_file), dtype=np.float32).reshape(-1, 4)
            raw_labels = np.fromfile(str(lbl_file), dtype=np.uint32) & 0xFFFF

            raw_frame = PointCloudFrame(points=raw_points, labels=raw_labels)
            filt_frame, _ = range_filter.filter_frame(raw_frame)

            pts_filt = filt_frame.points
            lbls_raw = filt_frame.labels
            r_filt = np.sqrt(pts_filt[:, 0]**2 + pts_filt[:, 1]**2)

            if "Phase 9 Error" in mode:
                gt_sih = np.full(len(lbls_raw), SuperClass.IGNORE_LABEL, dtype=np.int64)
                for native_c, sih_c in SEMANTICKITTI_TO_SIH.items():
                    gt_sih[lbls_raw == native_c] = sih_c
            else:
                gt_sih = remap_poss_labels(lbls_raw)

            pred = predictor.predict_frame(filt_frame)

            # Check point correspondence
            assert len(pred.predicted_class) == len(pts_filt), "Point length mismatch!"

            all_preds.append(pred.predicted_class)
            all_targets.append(gt_sih)
            all_ranges.append(r_filt)

        preds_cat = np.concatenate(all_preds)
        targs_cat = np.concatenate(all_targets)
        ranges_cat = np.concatenate(all_ranges)

        metrics = evaluator.evaluate(preds_cat, targs_cat, ranges=ranges_cat)
        results_audit[mode] = metrics

        print("=" * 80)
        print(f"CONFIGURATION: {mode}")
        print("=" * 80)
        print(f"Overall Accuracy:       {metrics['overall_accuracy']*100:6.2f}%")
        print(f"Mean IoU (mIoU):        {metrics['mIoU']*100:6.2f}%")
        print(f"Per-Class IoU Breakdown:")
        print(f"  Class 0 (Drivable):     {metrics['drivable_terrain_IoU']*100:6.2f}% (Precision: {metrics['drivable_terrain_Precision']*100:5.2f}%, Recall: {metrics['drivable_terrain_Recall']*100:5.2f}%)")
        print(f"  Class 1 (Non-Drivable): {metrics['non_drivable_terrain_IoU']*100:6.2f}% (Precision: {metrics['non_drivable_terrain_Precision']*100:5.2f}%, Recall: {metrics['non_drivable_terrain_Recall']*100:5.2f}%)")
        print(f"  Class 2 (Static Obs):   {metrics['static_obstacle_IoU']*100:6.2f}% (Precision: {metrics['static_obstacle_Precision']*100:5.2f}%, Recall: {metrics['static_obstacle_Recall']*100:5.2f}%)")
        print(f"  Class 3 (Dynamic Obj):  {metrics['dynamic_object_IoU']*100:6.2f}% (Precision: {metrics['dynamic_object_Precision']*100:5.2f}%, Recall: {metrics['dynamic_object_Recall']*100:5.2f}%)")
        print(f"\nConfusion Matrix (Rows=GroundTruth, Cols=Predictions):")
        cm = np.array(metrics['confusion_matrix'])
        print(f"        Drivable | Non-Driv | Static | Dynamic")
        for i, row in enumerate(cm):
            c_name = ["Drivable", "Non-Driv", "Static  ", "Dynamic "][i]
            print(f"{c_name:8s}: {row[0]:8d} | {row[1]:8d} | {row[2]:8d} | {row[3]:8d}")

    # 4. Save results to JSON
    with open("benchmarks/phase9_5_audit_results.json", "w") as f:
        json.dump(results_audit, f, indent=2)
    print("\nSaved forensic audit results to benchmarks/phase9_5_audit_results.json")

if __name__ == "__main__":
    main()
