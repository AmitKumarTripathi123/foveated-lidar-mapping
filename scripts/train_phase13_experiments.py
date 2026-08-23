"""
Phase 13: Master Experiment Runner for 3D LiDAR Semantic Segmentation.
Executes controlled optimization experiments across loss functions, class weighting,
LiDAR augmentations, and learning rate schedules.
"""

import argparse
import copy
import datetime
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from ml.data.dataset import LidarDataset
from ml.data.semanticposs_label_mapping import SemanticPOSSLabelRemapper
from ml.models.spvcnn import build_spvcnn
from ml.training.class_weights import (
    compute_training_class_frequencies,
    get_class_weights,
)
from ml.training.spvcnn_trainer import SPVCNNTrainer
from scripts.train_spvcnn_phase11_5 import SPVCNNFoveatedDataset, collate_spvcnn, get_dataset_root


def run_experiment(
    exp_name: str,
    base_config: Dict[str, Any],
    overrides: Dict[str, Any],
    train_ds: SPVCNNFoveatedDataset,
    val_ds: SPVCNNFoveatedDataset,
    class_frequencies: np.ndarray,
    device: torch.device,
    save_root: Path,
) -> Dict[str, Any]:
    """Execute a single controlled Phase 13 experiment."""
    print("\n" + "=" * 65)
    print(f"  LAUNCHING EXPERIMENT: {exp_name}")
    print("=" * 65)

    exp_cfg = copy.deepcopy(base_config)
    
    # Apply overrides
    for k, v in overrides.items():
        if isinstance(v, dict) and k in exp_cfg and isinstance(exp_cfg[k], dict):
            exp_cfg[k].update(v)
        else:
            exp_cfg[k] = v

    exp_dir = save_root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp_cfg["training"]["checkpoint_dir"] = str(exp_dir)

    # Resolve class weights
    weight_strategy = exp_cfg.get("loss", {}).get("weight_strategy")
    if weight_strategy and weight_strategy != "none":
        beta = float(exp_cfg.get("loss", {}).get("beta", 0.9999))
        weights = get_class_weights(class_frequencies, strategy=weight_strategy, beta=beta)
        exp_cfg["loss"]["class_weights"] = weights
        print(f"Using {weight_strategy} class weights: {weights}")
    elif exp_cfg.get("loss", {}).get("class_weights") is not None:
        weights = exp_cfg["loss"]["class_weights"]
        print(f"Using configured class weights: {weights}")
    else:
        exp_cfg["loss"]["class_weights"] = None
        print("Using uniform / unweighted loss.")

    # Save experiment config
    with open(exp_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(exp_cfg, f, default_flow_style=False)

    batch_size = int(exp_cfg.get("training", {}).get("batch_size", 2))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_spvcnn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_spvcnn)

    # Build fresh model from verified pretrained checkpoint
    ckpt_path = exp_cfg.get("model", {}).get("pretrained_checkpoint", "checkpoints/spvcnn_pretrained.pt")
    model = build_spvcnn(num_classes=4, in_channels=4, pretrained_path=ckpt_path, device=device)

    # Reset GPU peak memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    trainer = SPVCNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=exp_cfg,
        experiment_dir=exp_dir,
        device=device,
    )

    t0 = time.time()
    summary = trainer.train()
    wall_time = time.time() - t0

    # Reload check
    best_ckpt = exp_dir / "best_checkpoint.pt"
    passed, orig_miou, reloaded_miou = trainer.reload_and_validate(best_ckpt)
    print(f"Reload Check for {exp_name}: {'PASS' if passed else 'FAIL'} (Orig: {orig_miou:.2f}%, Reloaded: {reloaded_miou:.2f}%)")

    final_res = {
        "experiment_name": exp_name,
        "config": exp_cfg,
        "best_epoch": summary["best_epoch"],
        "best_val_miou": summary["best_val_miou"],
        "total_time_seconds": round(wall_time, 2),
        "reload_passed": passed,
        "final_metrics": summary["final_metrics"],
    }
    return final_res


def main():
    parser = argparse.ArgumentParser(description="Phase 13 Master Experiment Runner.")
    parser.add_argument("--dataset-root", type=str, default="dataset", help="Path to dataset root.")
    parser.add_argument("--save-root", type=str, default="experiments/phase13", help="Root folder for Phase 13.")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu).")
    parser.add_argument("--experiments", nargs="+", default=["all"], help="Experiments to run (A, B1, B2, C1, C2, C3, D, E, F or all).")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs per experiment.")
    args = parser.parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Phase 13 Active Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Dataset Partitions
    ds_root = Path(get_dataset_root(args.dataset_root))
    train_seqs = ["00", "01", "03", "04", "05"]
    val_seqs = ["02"]

    print(f"Indexing Training Dataset (Sequences: {train_seqs})...")
    train_records = []
    for s in train_seqs:
        s_path = ds_root / "sequences" / s
        v_dir = s_path / "velodyne"
        l_dir = s_path / "labels"
        if v_dir.is_dir() and l_dir.is_dir():
            for b_file in sorted(v_dir.glob("*.bin")):
                l_file = l_dir / f"{b_file.stem}.label"
                if l_file.is_file():
                    train_records.append((f"{s}_{b_file.stem}", str(b_file), str(l_file)))

    print(f"Indexing Validation Dataset (Sequence: {val_seqs})...")
    val_records = []
    for s in val_seqs:
        s_path = ds_root / "sequences" / s
        v_dir = s_path / "velodyne"
        l_dir = s_path / "labels"
        if v_dir.is_dir() and l_dir.is_dir():
            for b_file in sorted(v_dir.glob("*.bin")):
                l_file = l_dir / f"{b_file.stem}.label"
                if l_file.is_file():
                    val_records.append((f"{s}_{b_file.stem}", str(b_file), str(l_file)))

    train_ds = SPVCNNFoveatedDataset(train_records, foveate=True)
    val_ds = SPVCNNFoveatedDataset(val_records, foveate=True)

    assert len(train_ds) == 2488, f"Expected 2488 train frames, got {len(train_ds)}"
    assert len(val_ds) == 500, f"Expected 500 val frames, got {len(val_ds)}"
    print(f"Dataset Verified: Train={len(train_ds)}, Val={len(val_ds)}")

    # 2. Compute Training Class Frequencies strictly from training partition
    print("Computing Training-Only Class Frequencies...")
    class_freqs = compute_training_class_frequencies(train_ds, sample_fraction=0.1)
    print(f"Training Class Point Frequencies: {class_freqs.tolist()}")

    base_config = {
        "dataset": {
            "root": args.dataset_root,
            "train_sequences": train_seqs,
            "val_sequences": val_seqs,
            "target_total_frames": 2988,
        },
        "model": {
            "name": "spvcnn",
            "in_channels": 4,
            "num_classes": 4,
            "base_channels": 32,
            "voxel_size": 0.05,
            "pretrained_checkpoint": "checkpoints/spvcnn_pretrained.pt",
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "optimizer": "adamw",
            "scheduler": "cosine",
        },
        "loss": {
            "type": "weighted_cross_entropy",
            "ignore_index": 255,
            "weight_strategy": "inverse",
        },
        "augmentation": {
            "enabled": False,
            "rotation_deg": 10.0,
            "min_scale": 0.95,
            "max_scale": 1.05,
            "jitter_std": 0.01,
            "jitter_clip": 0.03,
            "flip_x_prob": 0.5,
            "flip_y_prob": 0.5,
            "translation_max": 0.1,
        },
        "collapse": {"threshold_pct": 90.0},
        "experiment": {
            "name": "phase13_experiment",
            "device": str(device),
            "seed": 42,
        },
    }

    # Define Experiment Suite
    suite = {
        "reproduction": {
            "desc": "Exp A: Phase 12 Baseline Reproduction (Weighted CE Inverse)",
            "overrides": {"loss": {"type": "weighted_cross_entropy", "weight_strategy": "inverse"}},
        },
        "weighted_ce_sqrt": {
            "desc": "Exp B1: Sqrt-Inverse Frequency Weighted Cross Entropy",
            "overrides": {"loss": {"type": "weighted_cross_entropy", "weight_strategy": "sqrt"}},
        },
        "weighted_ce_effective": {
            "desc": "Exp B2: Effective Number Class-Weighted Cross Entropy (Cui et al.)",
            "overrides": {"loss": {"type": "weighted_cross_entropy", "weight_strategy": "effective", "beta": 0.9999}},
        },
        "focal_gamma1": {
            "desc": "Exp C1: Focal Loss (gamma=1.0, unweighted)",
            "overrides": {"loss": {"type": "focal_loss", "gamma": 1.0, "weight_strategy": "none"}},
        },
        "focal_gamma2": {
            "desc": "Exp C2: Focal Loss (gamma=2.0, unweighted)",
            "overrides": {"loss": {"type": "focal_loss", "gamma": 2.0, "weight_strategy": "none"}},
        },
        "balanced_focal": {
            "desc": "Exp D: Class-Balanced Focal Loss (gamma=2.0 + Effective Weights)",
            "overrides": {"loss": {"type": "class_balanced_focal", "gamma": 2.0, "weight_strategy": "effective", "beta": 0.9999}},
        },
        "augmentation": {
            "desc": "Exp E: Training-Only LiDAR 3D Augmentation + Sqrt-Weighted CE",
            "overrides": {
                "loss": {"type": "weighted_cross_entropy", "weight_strategy": "sqrt"},
                "augmentation": {"enabled": True, "rotation_deg": 10.0, "min_scale": 0.95, "max_scale": 1.05, "jitter_std": 0.01},
            },
        },
        "lr_optimization": {
            "desc": "Exp F: Learning Rate Optimization (LR=5e-4 + Sqrt-Weighted CE + Augmentation)",
            "overrides": {
                "training": {"learning_rate": 0.0005},
                "loss": {"type": "weighted_cross_entropy", "weight_strategy": "sqrt"},
                "augmentation": {"enabled": True},
            },
        },
    }

    selected_exps = suite.keys() if "all" in args.experiments else [k for k in args.experiments if k in suite]

    results = []
    for exp_id in selected_exps:
        info = suite[exp_id]
        print(f"\n>>> Running {info['desc']}")
        res = run_experiment(
            exp_name=exp_id,
            base_config=base_config,
            overrides=info["overrides"],
            train_ds=train_ds,
            val_ds=val_ds,
            class_frequencies=class_freqs,
            device=device,
            save_root=save_root,
        )
        results.append(res)

    # 3. Select Best Model
    results_sorted = sorted(
        results,
        key=lambda r: (
            r["best_val_miou"],
            r["final_metrics"]["per_class_iou"].get("3", 0.0),
        ),
        reverse=True,
    )
    best_res = results_sorted[0]
    print("\n" + "=" * 65)
    print(f"  PHASE 13 BEST MODEL SELECTED: {best_res['experiment_name']}")
    print(f"  Best Validation mIoU: {best_res['best_val_miou']:.2f}% (Dynamic IoU: {best_res['final_metrics']['per_class_iou'].get('3', 0.0):.2f}%)")
    print("=" * 65)

    # Copy best model to best_model/
    best_dir = save_root / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    src_ckpt = save_root / best_res["experiment_name"] / "best_checkpoint.pt"
    dst_ckpt = best_dir / "best_checkpoint.pt"
    if src_ckpt.exists():
        shutil.copy2(src_ckpt, dst_ckpt)

    # Save experiment comparison JSON
    comparison_summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "phase12_baseline_miou": 53.59,
        "phase13_best_experiment": best_res["experiment_name"],
        "phase13_best_miou": best_res["best_val_miou"],
        "delta_percentage_points": round(best_res["best_val_miou"] - 53.59, 2),
        "experiments": results,
    }

    with open(save_root / "experiment_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)

    print(f"Saved comparison summary to: {save_root / 'experiment_comparison.json'}")


if __name__ == "__main__":
    main()
