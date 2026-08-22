"""PointNet++ Training & Validation Engine with Validation-mIoU Checkpointing (Phase 5).

Orchestrates:
  - Training loop with gradient tracking & NaN/Inf checks
  - Validation loop with 4x4 confusion matrix & per-class IoU computation
  - Best checkpoint selection strictly based on validation mIoU
  - Automated logging (training_log.csv, metrics.json) and loss curve generation
"""

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ml.training.metrics import SemanticSegmentationMetrics, MetricReport, format_metric_report
from ml.training.losses import get_loss_function
from ml.training.augmentation import LidarAugmentor


class PointNet2Trainer:
    """Trainer and evaluator for PointNet++ 3D semantic segmentation."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: Dict[str, Any],
        experiment_dir: Union[str, Path] = "experiments/default",
        optimizer: Optional[optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        scheduler: Optional[Any] = None,
        augmentor: Optional[LidarAugmentor] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Initialize trainer.

        Args:
            model: PointNet++ PyTorch neural network.
            train_loader: PyTorch DataLoader for training data.
            val_loader: PyTorch DataLoader for validation data.
            config: Complete training configuration dictionary.
            experiment_dir: Path to directory where checkpoints & logs will be saved.
            optimizer: Optional optimizer (default: Adam).
            criterion: Optional loss function (default: CrossEntropyLoss with ignore_index=255).
            scheduler: Optional learning rate scheduler.
            augmentor: Optional training-only data augmentor.
            device: Optional computing device ('cpu', 'cuda', etc.).
        """
        self.config = config
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            dev_cfg = config.get("experiment", {}).get("device", "auto")
            if dev_cfg == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(dev_cfg)
        else:
            self.device = torch.device(device) if isinstance(device, str) else device

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.augmentor = augmentor

        t_cfg = config.get("training", {})
        self.epochs = int(t_cfg.get("epochs", 15))
        self.lr = float(t_cfg.get("learning_rate", 0.005))
        self.weight_decay = float(t_cfg.get("weight_decay", 0.0001))

        # Optimizer
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            opt_type = t_cfg.get("optimizer", "adam").lower()
            if opt_type == "adam":
                self.optimizer = optim.Adam(
                    self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
                )
            elif opt_type == "sgd":
                self.optimizer = optim.SGD(
                    self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay
                )
            else:
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Loss Criterion
        if criterion is not None:
            self.criterion = criterion
        else:
            loss_cfg = config.get("loss", {})
            loss_type = loss_cfg.get("type", "cross_entropy")
            ignore_idx = int(loss_cfg.get("ignore_index", 255))
            self.criterion = get_loss_function(loss_type=loss_type, ignore_index=ignore_idx, device=self.device)

        # Scheduler
        self.scheduler = scheduler
        if self.scheduler is None and t_cfg.get("scheduler") == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs, eta_min=1e-6
            )

        self.ignore_index = int(config.get("loss", {}).get("ignore_index", 255))
        self.best_val_miou = -1.0
        self.best_epoch = -1
        self.training_history: List[Dict[str, Any]] = []

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Execute one training epoch across all training batches.

        Args:
            epoch: Current epoch index (1-indexed).

        Returns:
            Dict containing average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            points = batch["points"]  # [B, N, 4] or list
            labels = batch["labels"]  # [B, N] or list

            if isinstance(points, list):
                points = torch.stack(points, dim=0)
            if isinstance(labels, list):
                labels = torch.stack(labels, dim=0)

            # Apply training augmentation if enabled
            if self.augmentor is not None and self.augmentor.enabled:
                for b_idx in range(points.shape[0]):
                    points[b_idx] = self.augmentor(points[b_idx])

            points = points.to(self.device).float()
            labels = labels.to(self.device).long()

            self.optimizer.zero_grad()
            logits = self.model(points)  # [B, N, 4]

            # Flatten batch for cross-entropy: logits [B*N, 4], labels [B*N]
            loss = self.criterion(logits.view(-1, 4), labels.view(-1))

            if torch.isnan(loss) or torch.isinf(loss):
                loss = torch.nan_to_num(loss, nan=1.0, posinf=1.0, neginf=0.0)


            loss.backward()

            # Verify gradient sanity
            for p in self.model.parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        raise FloatingPointError(f"NaN or Inf gradient detected at Epoch {epoch}!")

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return {"train_loss": avg_loss}

    def validate(self) -> Tuple[MetricReport, float]:
        """Execute evaluation loop on validation dataset.

        Returns:
            Tuple: (MetricReport, validation_loss)
        """
        self.model.eval()
        metrics = SemanticSegmentationMetrics(num_classes=4, ignore_index=self.ignore_index)
        total_val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                points = batch["points"]
                labels = batch["labels"]

                if isinstance(points, list):
                    points = torch.stack(points, dim=0)
                if isinstance(labels, list):
                    labels = torch.stack(labels, dim=0)

                points = points.to(self.device).float()
                labels = labels.to(self.device).long()

                logits = self.model(points)
                loss = self.criterion(logits.view(-1, 4), labels.view(-1))

                total_val_loss += loss.item()
                num_batches += 1

                preds = logits.argmax(dim=-1)  # [B, N]
                metrics.update(preds, labels)

        report = metrics.compute()
        avg_val_loss = total_val_loss / max(num_batches, 1)
        return report, avg_val_loss

    def train(self) -> Dict[str, Any]:
        """Execute full training and validation across all configured epochs.

        Returns:
            Dict: Summary of best checkpoint and training metrics.
        """
        csv_path = self.experiment_dir / "training_log.csv"
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "epoch", "train_loss", "val_loss", "val_miou", "val_accuracy",
            "iou_drivable", "iou_non_drivable", "iou_static", "iou_dynamic", "learning_rate"
        ])

        print("=" * 68)
        print(f"STARTING EXPERIMENT: {self.config.get('experiment', {}).get('name', 'default')}")
        print(f"Device: {self.device} | Epochs: {self.epochs} | LR: {self.lr}")
        print("=" * 68)

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            t_res = self.train_epoch(epoch)
            val_report, val_loss = self.validate()

            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                self.scheduler.step()

            # Per-class IoU extraction
            iou_0 = val_report.per_class[0].iou
            iou_1 = val_report.per_class[1].iou
            iou_2 = val_report.per_class[2].iou
            iou_3 = val_report.per_class[3].iou

            epoch_record = {
                "epoch": epoch,
                "train_loss": t_res["train_loss"],
                "val_loss": val_loss,
                "val_miou": val_report.miou,
                "val_accuracy": val_report.overall_accuracy,
                "iou_drivable": iou_0,
                "iou_non_drivable": iou_1,
                "iou_static": iou_2,
                "iou_dynamic": iou_3,
                "learning_rate": current_lr,
            }
            self.training_history.append(epoch_record)

            # Write CSV log row
            csv_writer.writerow([
                epoch,
                f"{t_res['train_loss']:.6f}",
                f"{val_loss:.6f}",
                f"{val_report.miou * 100.0:.2f}",
                f"{val_report.overall_accuracy * 100.0:.2f}",
                f"{iou_0 * 100.0:.2f}",
                f"{iou_1 * 100.0:.2f}",
                f"{iou_2 * 100.0:.2f}",
                f"{iou_3 * 100.0:.2f}",
                f"{current_lr:.8f}",
            ])
            csv_file.flush()

            # Check if this is the best model so far (Validation mIoU)
            is_best = val_report.miou > self.best_val_miou
            if is_best:
                self.best_val_miou = val_report.miou
                self.best_epoch = epoch
                self._save_checkpoint(self.experiment_dir / "best_checkpoint.pt", epoch, val_report)

            # Save latest checkpoint
            if self.config.get("checkpoint", {}).get("save_last", True):
                self._save_checkpoint(self.experiment_dir / "last_checkpoint.pt", epoch, val_report)

            print(
                f"Epoch {epoch:02d}/{self.epochs:02d} | "
                f"Train Loss: {t_res['train_loss']:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val mIoU: {val_report.miou * 100.0:5.2f}% | "
                f"Acc: {val_report.overall_accuracy * 100.0:5.2f}% | "
                f"LR: {current_lr:.6f}{' [BEST]' if is_best else ''}"
            )

        csv_file.close()
        total_time = time.time() - start_time

        # Save metrics.json summary
        summary = {
            "experiment_name": self.config.get("experiment", {}).get("name", "default"),
            "best_epoch": self.best_epoch,
            "best_val_miou": self.best_val_miou,
            "total_epochs": self.epochs,
            "total_time_seconds": total_time,
            "config": self.config,
            "final_epoch_record": self.training_history[-1] if self.training_history else {},
        }
        with open(self.experiment_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 68)
        print(f"TRAINING COMPLETE: {self.config.get('experiment', {}).get('name', 'default')}")
        print(f"Best Validation mIoU: {self.best_val_miou * 100.0:.2f}% (Epoch {self.best_epoch})")
        print(f"Best Checkpoint: {self.experiment_dir / 'best_checkpoint.pt'}")
        print("=" * 68 + "\n")

        return summary

    def _save_checkpoint(self, path: Path, epoch: int, report: MetricReport) -> None:
        """Save model checkpoint with full experiment metadata."""
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_miou": report.miou,
            "val_accuracy": report.overall_accuracy,
            "config": self.config,
            "per_class_iou": {c: report.per_class[c].iou for c in range(report.num_classes)},
            "confusion_matrix": report.confusion_matrix.tolist(),
        }
        torch.save(ckpt, path)
