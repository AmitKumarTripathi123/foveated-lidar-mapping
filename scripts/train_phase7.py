#!/usr/bin/env python3
"""Phase 7 Multi-Frame PointNet++ Training & Diagnostic Engine.

Usage:
    # Experiment A: Baseline Plain Cross-Entropy
    python scripts/train_phase7.py --experiment phase7_baseline_ce --epochs 10

    # Experiment B: Class-Weighted Cross-Entropy
    python scripts/train_phase7.py --experiment phase7_weighted_ce --epochs 10 --weighted-loss

    # Experiment C: Class-Weighted Cross-Entropy + 3D Augmentation
    python scripts/train_phase7.py --experiment phase7_weighted_ce_aug --epochs 10 --weighted-loss --augmentation
"""

import argparse
import json
import sys
from pathlib import Path
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.dataset import lidar_collate_fn
from ml.data.foveated_dataset import FoveatedLidarDataset
from ml.data.manifest import discover_dataset
from ml.models.pointnet2 import build_model
from ml.training.losses import get_loss_function, compute_class_weights
from ml.training.augmentation import LidarAugmentor
from ml.training.trainer import PointNet2Trainer


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Phase 7 Multi-Frame PointNet++ Training.")
    parser.add_argument("--config", type=str, default="configs/phase7_training.yaml", help="Path to training config")
    parser.add_argument("--experiment", type=str, default=None, help="Experiment name")
    parser.add_argument("--epochs", type=int, default=None, help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--num-points", type=int, default=None, help="Point resolution")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--weighted-loss", action="store_true", help="Enable training-set class weighting")
    parser.add_argument("--augmentation", action="store_true", help="Enable training-only 3D augmentation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--force-raw", action="store_true", help="Force loading raw dataset bypass cache")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if cfg_path.is_file():
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    exp_name = args.experiment or config.get("experiment", {}).get("name", "phase7_baseline_ce")
    config.setdefault("experiment", {})["name"] = exp_name
    config["experiment"]["seed"] = args.seed

    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        config.setdefault("training", {})["learning_rate"] = args.lr
    if args.num_points is not None:
        config.setdefault("dataset", {})["num_points"] = args.num_points
    if args.weighted_loss:
        config.setdefault("loss", {})["type"] = "weighted_cross_entropy"
    if args.augmentation:
        config.setdefault("augmentation", {})["enabled"] = True

    seed = config.get("experiment", {}).get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_points = config.get("dataset", {}).get("num_points", 1024)
    batch_size = config.get("training", {}).get("batch_size", 1)

    cached_train_dir = Path(config.get("dataset", {}).get("cached_train_dir", "processed/train"))
    cached_val_dir = Path(config.get("dataset", {}).get("cached_val_dir", "processed/val"))

    manifest = discover_dataset(config.get("dataset", {}).get("dataset_root", "dataset"))
    expected_train_count = len(manifest.get("train", []))
    expected_val_count = len(manifest.get("val", []))

    cached_train_files = list(cached_train_dir.glob("*_pts.npy")) if cached_train_dir.is_dir() else []
    cached_val_files = list(cached_val_dir.glob("*_pts.npy")) if cached_val_dir.is_dir() else []

    if (
        not args.force_raw
        and len(cached_train_files) >= expected_train_count > 0
        and len(cached_val_files) >= expected_val_count > 0
    ):
        print(f"[CACHE MATCH] Loading full preprocessed dataset cache: {len(cached_train_files)} train frames, {len(cached_val_files)} val frames")
        train_dataset = FoveatedLidarDataset(
            cached_dir=cached_train_dir, target_num_points=num_points, to_tensor=True, seed=seed
        )
        val_dataset = FoveatedLidarDataset(
            cached_dir=cached_val_dir, target_num_points=num_points, to_tensor=True, seed=seed + 1000
        )
    else:
        print(f"[RAW DISCOVERY] Loading raw dataset manifest: {expected_train_count} train frames, {expected_val_count} val frames (Cache count was: train={len(cached_train_files)}, val={len(cached_val_files)})")
        train_dataset = FoveatedLidarDataset(
            raw_manifest=manifest["train"], target_num_points=num_points, to_tensor=True, seed=seed
        )
        val_dataset = FoveatedLidarDataset(
            raw_manifest=manifest["val"], target_num_points=num_points, to_tensor=True, seed=seed + 1000
        )


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lidar_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lidar_collate_fn)

    class_weights = None
    if config.get("loss", {}).get("type") == "weighted_cross_entropy":
        raw_train_sample = train_dataset[0]
        train_labels = raw_train_sample["labels"].numpy()
        valid_labels = train_labels[train_labels != 255]
        unique_c, counts = np.unique(valid_labels, return_counts=True)
        train_class_counts = {int(c): int(cnt) for c, cnt in zip(unique_c, counts)}

        strat = config.get("loss", {}).get("weighting_strategy", "inverse_frequency")
        class_weights = compute_class_weights(train_class_counts, num_classes=4, strategy=strat)
        print(f"Computed Phase 7 Training Class Weights ({strat}): {class_weights.tolist()}")

    model = build_model(name="pointnet2_semseg", num_classes=4, in_channels=4)

    aug_cfg = config.get("augmentation", {})
    augmentor = None
    if aug_cfg.get("enabled", False):
        augmentor = LidarAugmentor(
            enabled=True,
            rotation_range=tuple(aug_cfg.get("rotation_range", [-15.0, 15.0])),
            scale_range=tuple(aug_cfg.get("scale_range", [0.95, 1.05])),
            jitter_std=float(aug_cfg.get("jitter_std", 0.01)),
            seed=seed,
        )

    ignore_idx = int(config.get("loss", {}).get("ignore_index", 255))
    criterion = get_loss_function(
        loss_type=config.get("loss", {}).get("type", "cross_entropy"),
        class_weights=class_weights,
        ignore_index=ignore_idx,
    )

    exp_dir = Path("experiments") / exp_name
    trainer = PointNet2Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        experiment_dir=exp_dir,
        criterion=criterion,
        augmentor=augmentor,
    )

    summary = trainer.train()
    return 0 if summary["best_val_miou"] >= 0.0 else 1


if __name__ == "__main__":
    sys.exit(main())
