#!/usr/bin/env python3
"""PointNet++ Training CLI Script on Foveated LiDAR Data (Master Task).

Usage:
    # Experiment A: Plain Cross-Entropy Baseline
    python scripts/train.py --experiment baseline_ce --epochs 10 --num-points 1024

    # Experiment B: Class-Weighted Cross-Entropy
    python scripts/train.py --experiment weighted_ce --epochs 10 --num-points 1024 --weighted-loss

    # Experiment C: Class-Weighted Cross-Entropy + Training Augmentation
    python scripts/train.py --experiment weighted_ce_aug --epochs 10 --num-points 1024 --weighted-loss --augmentation
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

from ml.data.foveated_dataset import FoveatedLidarDataset
from ml.data.dataset import lidar_collate_fn
from ml.data.manifest import discover_dataset
from ml.models.pointnet2 import build_model
from ml.training.losses import get_loss_function, compute_class_weights
from ml.training.augmentation import LidarAugmentor
from ml.training.trainer import PointNet2Trainer


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Train PointNet++ Semantic Segmentation on Foveated LiDAR.")
    parser.add_argument("--config", type=str, default="ml/configs/training.yaml", help="Path to training.yaml")
    parser.add_argument("--experiment", type=str, default=None, help="Experiment name (e.g. baseline_ce)")
    parser.add_argument("--epochs", type=int, default=None, help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--num-points", type=int, default=None, help="Point resolution (default: 1024)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--weighted-loss", action="store_true", help="Enable training-set class weighting")
    parser.add_argument("--augmentation", action="store_true", help="Enable training-only 3D augmentation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # 1. Load Base Config
    cfg_path = Path(args.config)
    if cfg_path.is_file():
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    # Override CLI flags
    exp_name = args.experiment or config.get("experiment", {}).get("name", "baseline_ce")
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

    # 2. Set Reproducibility Seeds
    seed = config.get("experiment", {}).get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_points = config.get("dataset", {}).get("num_points", 1024)
    batch_size = config.get("training", {}).get("batch_size", 1)

    # 3. Create Datasets via Foveated Preprocessing Cache
    cached_train_dir = Path("processed/train")
    cached_val_dir = Path("processed/val")

    if cached_train_dir.is_dir() and len(list(cached_train_dir.glob("*_pts.npy"))) > 0:
        train_dataset = FoveatedLidarDataset(
            cached_dir=cached_train_dir, target_num_points=num_points, to_tensor=True, seed=seed
        )
        val_dataset = FoveatedLidarDataset(
            cached_dir=cached_val_dir, target_num_points=num_points, to_tensor=True, seed=seed + 1000
        )
    else:
        manifest = discover_dataset("dataset")
        train_dataset = FoveatedLidarDataset(
            raw_manifest=manifest["train"], target_num_points=num_points, to_tensor=True, seed=seed
        )
        val_dataset = FoveatedLidarDataset(
            raw_manifest=manifest["val"], target_num_points=num_points, to_tensor=True, seed=seed + 1000
        )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lidar_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lidar_collate_fn
    )

    # 4. Compute Training Split Class Distribution & Weights if requested
    class_weights = None
    if config.get("loss", {}).get("type") == "weighted_cross_entropy":
        raw_train_sample = train_dataset[0]
        train_labels = raw_train_sample["labels"].numpy()
        valid_labels = train_labels[train_labels != 255]
        unique_c, counts = np.unique(valid_labels, return_counts=True)
        train_class_counts = {int(c): int(cnt) for c, cnt in zip(unique_c, counts)}

        strat = config.get("loss", {}).get("weighting_strategy", "inverse_frequency")
        class_weights = compute_class_weights(train_class_counts, num_classes=4, strategy=strat)
        print(f"Computed Training Class Weights ({strat}): {class_weights.tolist()}")

    # 5. Build Model & Augmentor
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

    # 6. Loss Function
    ignore_idx = int(config.get("loss", {}).get("ignore_index", 255))
    criterion = get_loss_function(
        loss_type=config.get("loss", {}).get("type", "cross_entropy"),
        class_weights=class_weights,
        ignore_index=ignore_idx,
    )

    # 7. Execute Trainer
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
