"""SPVCNN Fine-Tuning Engine and Scientific Validator (Phase 11.5).

Implements end-to-end multi-frame fine-tuning for SPVCNN on SemanticPOSS foveated data:
  - Input: Foveated point clouds [N, 4] and remapped SIH labels [N]
  - Loss: Cross-Entropy with ignore_index=255 and optional inverse-frequency weighting
  - Metrics: Confusion matrix, per-class IoU (0, 1, 2, 3), mIoU, precision, recall, accuracy
  - Model Collapse Diagnostic: Dominant class proportion and prediction entropy
  - Checkpoint Reload Validator: Strict reproducibility assertion
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
from torch.utils.data import DataLoader

from ml.data.spvcnn_adapter import SPVCNNInputAdapter
from ml.models.spvcnn import SPVCNN, load_spvcnn_checkpoint


class SPVCNNTrainer:
    """Trainer for fine-tuning SPVCNN on 3D LiDAR semantic segmentation."""

    def __init__(
        self,
        model: SPVCNN,
        train_loader: Optional[DataLoader],
        val_loader: Optional[DataLoader],
        config: Dict[str, Any],
        experiment_dir: Union[str, Path] = "experiments/spvcnn_phase11_5",
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.config = config
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            cfg_dev = config.get("experiment", {}).get("device")
            if cfg_dev:
                self.device = torch.device(cfg_dev)
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 10)
        self.lr = float(train_cfg.get("learning_rate", 0.001))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-4))
        self.ignore_index = int(config.get("loss", {}).get("ignore_index", 255))
        self.num_classes = int(config.get("model", {}).get("num_classes", 4))
        self.voxel_size = float(config.get("model", {}).get("voxel_size", 0.05))

        class_weights = config.get("loss", {}).get("class_weights")
        if class_weights is not None:
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights_tensor, ignore_index=self.ignore_index)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        opt_name = train_cfg.get("optimizer", "adam").lower()
        if opt_name == "adamw":
            self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            self.optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(self.epochs, 1), eta_min=1e-6)
        self.input_adapter = SPVCNNInputAdapter(voxel_size=self.voxel_size)

        self.history: List[Dict[str, Any]] = []
        self.best_val_miou: float = 0.0
        self.best_epoch: int = 0

    def train_epoch(self) -> float:
        if self.train_loader is None or len(self.train_loader) == 0:
            return 0.0

        self.model.train()
        total_loss = 0.0
        total_batches = 0
        num_total_batches = len(self.train_loader)

        for i, batch in enumerate(self.train_loader):
            pts_batch, lbls_batch = self._unpack_batch(batch)
            if pts_batch is None:
                continue

            self.optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            for pts, lbls in zip(pts_batch, lbls_batch):
                pts_t = pts.to(self.device).float()
                lbls_t = lbls.to(self.device).long()

                bundle = self.input_adapter.prepare_input(pts_t, device=self.device)
                logits = self.model(
                    features=bundle["features"],
                    point_to_voxel_idx=bundle["point_to_voxel_idx"],
                    num_voxels=bundle["num_voxels"],
                )

                loss = self.criterion(logits, lbls_t)
                batch_loss = batch_loss + loss

            batch_loss = batch_loss / max(len(pts_batch), 1)
            batch_loss.backward()
            self.optimizer.step()

            total_loss += batch_loss.item()
            total_batches += 1

            if (i + 1) % 200 == 0 or (i + 1) == num_total_batches:
                avg_so_far = total_loss / max(total_batches, 1)
                print(f"  [Train Batch {i+1:4d}/{num_total_batches:4d}] Loss: {avg_so_far:.4f}")

        self.scheduler.step()
        return total_loss / max(total_batches, 1)

    def evaluate(self, data_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        loader = data_loader if data_loader is not None else self.val_loader
        if loader is None or len(loader) == 0:
            return {
                "val_loss": 0.0,
                "val_miou": 0.0,
                "overall_accuracy": 0.0,
                "per_class_iou": {str(c): 0.0 for c in range(self.num_classes)},
                "per_class_precision": {str(c): 0.0 for c in range(self.num_classes)},
                "per_class_recall": {str(c): 0.0 for c in range(self.num_classes)},
                "confusion_matrix": [[0]*self.num_classes for _ in range(self.num_classes)],
                "evaluated_points": 0,
                "supervised_points": 0,
                "ignored_points": 0,
                "dominant_class": None,
                "dominant_class_pct": 0.0,
                "prediction_entropy": 0.0,
                "collapse_warning": False,
            }

        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        num_val_batches = len(loader)

        cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        pred_counts = np.zeros(self.num_classes, dtype=np.int64)
        total_pts = 0
        total_supervised = 0
        total_ignored = 0

        with torch.no_grad():
            for i, batch in enumerate(loader):
                pts_batch, lbls_batch = self._unpack_batch(batch)
                if pts_batch is None:
                    continue

                for pts, lbls in zip(pts_batch, lbls_batch):
                    pts_t = pts.to(self.device).float()
                    lbls_t = lbls.to(self.device).long()

                    bundle = self.input_adapter.prepare_input(pts_t, device=self.device)
                    logits = self.model(
                        features=bundle["features"],
                        point_to_voxel_idx=bundle["point_to_voxel_idx"],
                        num_voxels=bundle["num_voxels"],
                    )

                    loss = self.criterion(logits, lbls_t)
                    total_loss += loss.item()
                    total_batches += 1

                    probs = F.softmax(logits, dim=-1)
                    preds = torch.argmax(probs, dim=-1).cpu().numpy().astype(np.int64)
                    gts = lbls_t.cpu().numpy().astype(np.int64)

                    total_pts += int(len(gts))
                    counts = np.bincount(preds, minlength=self.num_classes)
                    pred_counts[:len(counts)] += counts

                    valid = (gts != self.ignore_index) & (gts >= 0) & (gts < self.num_classes)
                    gts_v = gts[valid]
                    preds_v = preds[valid]

                    total_supervised += int(len(gts_v))
                    total_ignored += int(np.sum(gts == self.ignore_index))

                    for t in range(self.num_classes):
                        for p in range(self.num_classes):
                            cm[t, p] += int(np.sum((gts_v == t) & (preds_v == p)))

                if (i + 1) % 100 == 0 or (i + 1) == num_val_batches:
                    print(f"  [Val Batch {i+1:3d}/{num_val_batches:3d}] Processed...")

        assert int(np.sum(cm)) == total_supervised, f"Confusion matrix sum {int(np.sum(cm))} != supervised count {total_supervised}"

        ious = {}
        precisions = {}
        recalls = {}

        for c in range(self.num_classes):
            tp = int(cm[c, c])
            fp = int(np.sum(cm[:, c]) - tp)
            fn = int(np.sum(cm[c, :]) - tp)
            union = tp + fp + fn
            ious[str(c)] = float(tp / union * 100.0) if union > 0 else 0.0
            precisions[str(c)] = float(tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
            recalls[str(c)] = float(tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0

        val_miou = float(np.mean(list(ious.values()))) if ious else 0.0
        acc = float(np.trace(cm) / total_supervised * 100.0) if total_supervised > 0 else 0.0

        total_preds_count = int(np.sum(pred_counts))
        if total_preds_count > 0:
            dom_idx = int(np.argmax(pred_counts))
            dom_cls = dom_idx
            dom_pct = float(np.max(pred_counts) / total_preds_count * 100.0)
            p_dist = pred_counts / total_preds_count
            entropy = float(-np.sum(p_dist * np.log2(p_dist + 1e-12)))
        else:
            dom_cls = None
            dom_pct = 0.0
            entropy = 0.0

        collapse_threshold = float(self.config.get("collapse", {}).get("threshold_pct", 90.0))
        collapse_warning = dom_pct >= collapse_threshold

        # Measure GPU memory
        peak_alloc_mb = float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
        peak_res_mb = float(torch.cuda.max_memory_reserved() / (1024**2)) if torch.cuda.is_available() else 0.0

        return {
            "val_loss": round(float(total_loss / max(total_batches, 1)), 4),
            "val_miou": round(val_miou, 2),
            "overall_accuracy": round(acc, 2),
            "per_class_iou": {k: round(v, 2) for k, v in ious.items()},
            "per_class_precision": {k: round(v, 2) for k, v in precisions.items()},
            "per_class_recall": {k: round(v, 2) for k, v in recalls.items()},
            "confusion_matrix": cm.tolist(),
            "evaluated_points": total_pts,
            "supervised_points": total_supervised,
            "ignored_points": total_ignored,
            "dominant_class": dom_cls,
            "dominant_class_pct": round(dom_pct, 2),
            "prediction_entropy": round(entropy, 4),
            "collapse_warning": collapse_warning,
            "peak_vram_allocated_mb": round(peak_alloc_mb, 2),
            "peak_vram_reserved_mb": round(peak_res_mb, 2),
        }

    def train(self) -> Dict[str, Any]:
        print(f"=== Starting SPVCNN Fine-Tuning on {self.device} for {self.epochs} epochs ===")
        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            t_loss = self.train_epoch()
            val_metrics = self.evaluate()

            v_loss = val_metrics["val_loss"]
            v_miou = val_metrics["val_miou"]
            acc = val_metrics["overall_accuracy"]
            cur_lr = self.optimizer.param_groups[0]["lr"]

            is_best = v_miou > self.best_val_miou
            if is_best:
                self.best_val_miou = v_miou
                self.best_epoch = epoch
                self.save_checkpoint("best_checkpoint.pt", epoch, val_metrics)

            self.save_checkpoint("last_checkpoint.pt", epoch, val_metrics)

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(t_loss, 4),
                "val_loss": round(v_loss, 4),
                "val_miou": v_miou,
                "overall_accuracy": acc,
                "lr": cur_lr,
                "is_best": is_best,
            }
            self.history.append(epoch_record)

            best_tag = " [BEST]" if is_best else ""
            print(f"Epoch {epoch:02d}/{self.epochs:02d} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Val mIoU: {v_miou:5.2f}% | Acc: {acc:5.2f}% | LR: {cur_lr:.6f}{best_tag}")

        total_time = time.time() - start_time
        summary = {
            "total_epochs": self.epochs,
            "best_epoch": self.best_epoch,
            "best_val_miou": self.best_val_miou,
            "total_time_seconds": round(total_time, 2),
            "history": self.history,
            "final_metrics": self.evaluate(),
        }

        with open(self.experiment_dir / "training_history.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"=== SPVCNN Fine-Tuning Complete (Best Val mIoU: {self.best_val_miou:.2f}% at Epoch {self.best_epoch}) ===\n")
        return summary

    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict[str, Any]) -> str:
        ckpt_path = self.experiment_dir / filename
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }, ckpt_path)
        return str(ckpt_path)

    def reload_and_validate(self, checkpoint_path: Union[str, Path], tolerance: float = 1e-4) -> Tuple[bool, float, float]:
        ckpt_p = Path(checkpoint_path)
        if not ckpt_p.is_file():
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_p}")

        state = torch.load(ckpt_p, map_location="cpu")
        original_miou = float(state.get("metrics", {}).get("val_miou", 0.0))

        fresh_model = SPVCNN(num_classes=self.num_classes, in_channels=4, base_channels=32)
        fresh_model.load_state_dict(state["model_state_dict"])
        fresh_model.to(self.device)

        reloaded_trainer = SPVCNNTrainer(
            model=fresh_model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            config=self.config,
            experiment_dir=self.experiment_dir,
            device=self.device,
        )
        reloaded_metrics = reloaded_trainer.evaluate()
        reloaded_miou = float(reloaded_metrics["val_miou"])

        diff = abs(original_miou - reloaded_miou)
        passed = diff <= tolerance
        return passed, original_miou, reloaded_miou

    def _unpack_batch(self, batch: Any) -> Tuple[Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 2:
                pts, lbls = batch[0], batch[1]
                if isinstance(pts, torch.Tensor) and pts.ndim == 3:
                    return [pts[b] for b in range(pts.shape[0])], [lbls[b] for b in range(lbls.shape[0])]
                elif isinstance(pts, list):
                    return pts, lbls
                elif isinstance(pts, torch.Tensor) and pts.ndim == 2:
                    return [pts], [lbls]
        elif isinstance(batch, dict):
            pts = batch.get("points")
            lbls = batch.get("labels")
            if isinstance(pts, torch.Tensor) and pts.ndim == 3:
                return [pts[b] for b in range(pts.shape[0])], [lbls[b] for b in range(lbls.shape[0])]
            elif isinstance(pts, torch.Tensor) and pts.ndim == 2:
                return [pts], [lbls]
        return None, None
