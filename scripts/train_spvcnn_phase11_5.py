#!/usr/bin/env python3
"""scripts/train_spvcnn_phase11_5.py

Phase 11.5 SPVCNN Fine-Tuning Execution Script with Dataset Activation Gate.
Performs:
  1. Deep physical storage audit and completeness gating (2,988 expected frames)
  2. Disjoint sequence-level dataset partitioning (Train: 00,01,03,04,05 | Val: 02)
  3. Pretrained SPVCNN weight initialization
  4. Cross-entropy fine-tuning with ignore_index=255
  5. Checkpoint save & reload consistency verification
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Add repository root to sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.dataset import load_point_cloud, load_labels, validate_point_label_alignment
from ml.data.amit_adapter import FoveatedVoxelSampler
from ml.data.semanticposs_label_mapping import SemanticPOSSLabelRemapper
from ml.models.spvcnn import SPVCNN, build_spvcnn
from ml.training.spvcnn_trainer import SPVCNNTrainer
from scripts.audit_semanticposs import audit_sequence, get_dataset_root


class SPVCNNFoveatedDataset(Dataset):
    """PyTorch Dataset loading real SemanticPOSS LiDAR scans with Amit foveation and SIH mapping."""

    def __init__(
        self,
        frame_records: List[Tuple[str, str, str]],
        foveate: bool = True,
    ):
        self.records = frame_records
        self.foveate = foveate
        self.sampler = FoveatedVoxelSampler(
            near_dist=10.0, near_voxel=0.05,
            mid_dist=40.0, mid_voxel=0.15,
            far_dist=100.0, far_voxel=0.50,
        )
        self.remapper = SemanticPOSSLabelRemapper()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_id, bin_path, lbl_path = self.records[idx]
        pts = load_point_cloud(bin_path)
        lbls = load_labels(lbl_path)

        if self.foveate:
            pts_fov, lbls_fov, _ = self.sampler.sample(pts, lbls)
        else:
            pts_fov, lbls_fov = pts, lbls

        sih_lbls = self.remapper.remap(lbls_fov)

        return (
            torch.from_numpy(pts_fov).float(),
            torch.from_numpy(sih_lbls).long(),
        )


def collate_spvcnn(batch):
    pts_list = [item[0] for item in batch]
    lbls_list = [item[1] for item in batch]
    return pts_list, lbls_list


def check_dataset_completeness(root_dir: Path, expected_counts: Dict[str, int]) -> Dict[str, Any]:
    """Audit physical dataset against expected frame counts."""
    seq_dir = root_dir / "sequences"
    discovered = {}
    missing_seqs = []
    total_found = 0
    total_expected = sum(expected_counts.values())

    for seq_id, exp_count in expected_counts.items():
        s_path = seq_dir / seq_id
        if not s_path.is_dir():
            discovered[seq_id] = {"exists": False, "found": 0, "expected": exp_count, "missing": exp_count}
            missing_seqs.append(seq_id)
        else:
            res = audit_sequence(str(s_path), seq_id)
            matched = res.get("matched_pairs", 0)
            discovered[seq_id] = {"exists": True, "found": matched, "expected": exp_count, "missing": max(0, exp_count - matched)}
            total_found += matched
            if matched < exp_count:
                missing_seqs.append(seq_id)

    is_complete = (total_found == total_expected) and len(missing_seqs) == 0

    return {
        "is_complete": is_complete,
        "total_found": total_found,
        "total_expected": total_expected,
        "sequence_breakdown": discovered,
        "missing_sequences": missing_seqs,
    }


def compute_training_class_weights(dataset: SPVCNNFoveatedDataset, num_classes: int = 4, ignore_index: int = 255) -> List[float]:
    """Compute inverse-frequency class weights strictly from TRAINING DATA ONLY."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for i in range(len(dataset)):
        _, lbls = dataset[i]
        lbls_np = lbls.numpy()
        for c in range(num_classes):
            counts[c] += np.sum(lbls_np == c)

    total = np.sum(counts)
    if total == 0:
        return [1.0] * num_classes

    freqs = counts / total
    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if freqs[c] > 0:
            weights[c] = 1.0 / (np.log(1.02 + freqs[c]))
        else:
            weights[c] = 1.0

    # Normalize weights so mean is 1.0
    weights = weights / np.mean(weights)
    return weights.tolist()


def main():
    parser = argparse.ArgumentParser(description="Phase 11.5 SPVCNN Fine-Tuning Script.")
    parser.add_argument("--config", type=str, default="configs/phase11_5_spvcnn_training.yaml", help="Config file path.")
    parser.add_argument("--dataset-root", type=str, default=None, help="Override dataset root.")
    parser.add_argument("--allow-single-frame", action="store_true", help="Allow running single-scan demonstration if full dataset is absent.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    ds_root = Path(get_dataset_root(args.dataset_root if args.dataset_root else cfg.get("dataset", {}).get("root", "dataset")))
    expected_counts = cfg.get("dataset", {}).get("expected_frames", {"00": 488, "01": 500, "02": 500, "03": 500, "04": 500, "05": 500})

    print("==================================================")
    print("   PHASE 11.5 SPVCNN DATASET ACTIVATION GATE     ")
    print("==================================================")
    print(f"Dataset Root: {ds_root}")

    audit_res = check_dataset_completeness(ds_root, expected_counts)
    print(f"Expected Frames : {audit_res['total_expected']}")
    print(f"Discovered Frames: {audit_res['total_found']}")

    for seq, info in audit_res["sequence_breakdown"].items():
        status = "COMPLETE" if info["found"] == info["expected"] else f"PARTIAL ({info['found']}/{info['expected']})"
        print(f"  Sequence {seq}: {status}")

    if not audit_res["is_complete"]:
        print("\n--------------------------------------------------")
        print("GATE DECISION: DATASET ACTIVATION BLOCKED")
        print(f"Reason: Missing sequences/frames: {audit_res['missing_sequences']}")
        print("--------------------------------------------------\n")

        if not args.allow_single_frame:
            print("To proceed with a single-scan demonstration, rerun with --allow-single-frame.")
            print("Exiting at dataset gate (Zero-fabrication policy enforced).")
            return 1

    # Discover available files
    train_seqs = cfg.get("dataset", {}).get("train_sequences", ["00", "01", "03", "04", "05"])
    val_seqs = cfg.get("dataset", {}).get("val_sequences", ["02"])

    # Build file records
    train_records = []
    val_records = []
    seq_dir = ds_root / "sequences"

    for s in train_seqs:
        s_path = seq_dir / s
        if s_path.is_dir():
            res = audit_sequence(str(s_path), s)
            for stem, b_p, l_p, _ in res.get("frame_details", []):
                train_records.append((f"{s}_{stem}", b_p, l_p))

    for s in val_seqs:
        s_path = seq_dir / s
        if s_path.is_dir():
            res = audit_sequence(str(s_path), s)
            for stem, b_p, l_p, _ in res.get("frame_details", []):
                val_records.append((f"{s}_{stem}", b_p, l_p))

    if len(train_records) == 0:
        print("ERROR: No training records discovered. Exiting.")
        return 1

    # In single-scan demo mode, fallback validation to available training record
    if len(val_records) == 0:
        val_records = train_records.copy()

    train_ds = SPVCNNFoveatedDataset(train_records)
    val_ds = SPVCNNFoveatedDataset(val_records)

    train_loader = DataLoader(train_ds, batch_size=cfg.get("training", {}).get("batch_size", 2), shuffle=True, collate_fn=collate_spvcnn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_spvcnn)

    # Compute training class weights
    train_weights = compute_training_class_weights(train_ds)
    print(f"Computed training-only class weights: {[round(w, 4) for w in train_weights]}")

    if cfg.get("loss", {}).get("type") == "weighted_cross_entropy":
        cfg["loss"]["class_weights"] = train_weights

    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    # Build model & load pretrained weights
    ckpt_path = cfg.get("model", {}).get("pretrained_checkpoint", "checkpoints/spvcnn_pretrained.pt")
    model = build_spvcnn(num_classes=4, in_channels=4, pretrained_path=ckpt_path, device=cfg.get("experiment", {}).get("device", "cpu"))

    trainer = SPVCNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        experiment_dir=cfg.get("training", {}).get("checkpoint_dir", "experiments/phase11_5_spvcnn_ft"),
    )

    summary = trainer.train()

    # Checkpoint reload test
    best_ckpt = Path(cfg.get("training", {}).get("checkpoint_dir", "experiments/phase11_5_spvcnn_ft")) / "best_checkpoint.pt"
    passed, orig, reloaded = trainer.reload_and_validate(best_ckpt)
    print(f"Checkpoint reload verification: {'PASS' if passed else 'FAIL'} (Original: {orig:.2f}%, Reloaded: {reloaded:.2f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
