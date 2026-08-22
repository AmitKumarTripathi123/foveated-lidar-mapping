"""
Phase 2 AI Model Trainer.
"""
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from phase2.models.point_seg_net import FoveatedPointSegNet
from phase2.dataset import Phase2Dataset
from phase2.metrics.semantic_evaluator import Phase2SemanticEvaluator
from src.types import SuperClass


class Phase2Trainer:
    def __init__(
        self,
        model: Optional[FoveatedPointSegNet] = None,
        train_dataset: Optional[Phase2Dataset] = None,
        val_dataset: Optional[Phase2Dataset] = None,
        lr: float = 3e-3,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        device: Optional[str] = "cpu",
        checkpoint_dir: str = "checkpoints"
    ):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = model if model is not None else FoveatedPointSegNet()
        self.model.to(self.device)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if class_weights is None:
            # Default balanced weights: 0: drivable (2.5), 1: non-drivable (1.5), 2: static (0.8), 3: dynamic (4.0)
            class_weights = torch.tensor([2.5, 1.5, 0.8, 4.0], dtype=torch.float32)

        cw = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=cw, ignore_index=SuperClass.IGNORE_LABEL)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.evaluator = Phase2SemanticEvaluator()

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_points = 0

        for batch in dataloader:
            pts = batch["points"][0].to(self.device)
            lbls = batch["labels"][0].to(self.device)

            valid_mask = (lbls != SuperClass.IGNORE_LABEL)
            if valid_mask.sum() == 0:
                continue

            self.optimizer.zero_grad()
            logits = self.model(pts)
            loss = self.criterion(logits, lbls)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item()) * len(pts)
            total_points += len(pts)

        avg_loss = total_loss / max(total_points, 1)
        return {"train_loss": avg_loss}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        total_points = 0

        all_preds, all_targets, all_probs, all_ranges = [], [], [], []

        for batch in dataloader:
            pts = batch["points"][0].to(self.device)
            lbls = batch["labels"][0].to(self.device)

            logits = self.model(pts)
            loss = self.criterion(logits, lbls)

            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            total_loss += float(loss.item()) * len(pts)
            total_points += len(pts)

            r = torch.sqrt(pts[:, 0]**2 + pts[:, 1]**2)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(lbls.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_ranges.append(r.cpu().numpy())

        preds_cat = np.concatenate(all_preds) if all_preds else np.empty(0)
        targets_cat = np.concatenate(all_targets) if all_targets else np.empty(0)
        probs_cat = np.concatenate(all_probs) if all_probs else np.empty((0, 4))
        ranges_cat = np.concatenate(all_ranges) if all_ranges else np.empty(0)

        eval_res = self.evaluator.evaluate(preds_cat, targets_cat, probs_cat, ranges_cat)
        eval_res["val_loss"] = total_loss / max(total_points, 1)
        return eval_res

    def fit(self, epochs: int = 25, batch_size: int = 1) -> Dict[str, Any]:
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False) if self.val_dataset else None

        best_miou = -1.0
        history = []

        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            train_metrics = self.train_epoch(train_loader)
            dt = time.perf_counter() - t0

            val_metrics = self.validate(val_loader) if val_loader else {}
            miou = val_metrics.get("mIoU", 0.0)

            rec = {
                "epoch": epoch,
                "train_loss": train_metrics["train_loss"],
                "val_loss": val_metrics.get("val_loss", 0.0),
                "mIoU": miou,
                "epoch_time_s": dt
            }
            history.append(rec)

            if miou > best_miou:
                best_miou = miou
                self.save_checkpoint("best_model.pth", metadata={"epoch": epoch, "mIoU": best_miou})

        self.save_checkpoint("latest_model.pth", metadata={"epoch": epochs, "mIoU": miou})
        return {"history": history, "best_mIoU": best_miou}

    def save_checkpoint(self, filename: str, metadata: Optional[Dict[str, Any]] = None):
        ckpt_path = self.checkpoint_dir / filename
        data = {
            "model_state_dict": self.model.state_dict(),
            "metadata": metadata or {}
        }
        torch.save(data, str(ckpt_path))

    def load_checkpoint(self, filename: str):
        ckpt_path = self.checkpoint_dir / filename
        data = torch.load(str(ckpt_path), map_location=self.device)
        self.model.load_state_dict(data["model_state_dict"])
        return data.get("metadata", {})
