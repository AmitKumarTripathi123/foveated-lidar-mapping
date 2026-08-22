import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.types import SuperClass, PointCloudFrame
from src.range_filter import RangeFilter
from phase2.dataset import remap_poss_labels, SEMANTICPOSS_TO_PROJECT
from phase2.models.spvcnn import SPVCNN, build_spvcnn, load_spvcnn_checkpoint
from phase2.models.spvcnn_adapter import SPVCNNInputAdapter, SPVCNNLabelAdapter
from phase2.metrics.semantic_evaluator import Phase2SemanticEvaluator
from tools.dataset_manager import build_full_manifest

class DistillationLoss(nn.Module):
    """
    Combined Knowledge Distillation Loss for SPVCNN Student:
    L_total = alpha * L_CE(student_logits, gt) + (1 - alpha) * T^2 * KL(student_soft, teacher_soft)
    """
    def __init__(self, alpha: float = 0.5, temperature: float = 3.0, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=SuperClass.IGNORE_LABEL)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_ce = self.ce_loss(student_logits, targets)

        valid_mask = (targets != SuperClass.IGNORE_LABEL)
        if valid_mask.sum() > 0:
            s_valid = student_logits[valid_mask]
            t_valid = teacher_logits[valid_mask]

            s_soft = F.log_softmax(s_valid / self.temperature, dim=-1)
            t_soft = F.softmax(t_valid / self.temperature, dim=-1)

            loss_kd = self.kl_loss(s_soft, t_soft) * (self.temperature ** 2)
        else:
            loss_kd = torch.tensor(0.0, device=student_logits.device)

        loss_total = self.alpha * loss_ce + (1.0 - self.alpha) * loss_kd
        return loss_total, loss_ce, loss_kd

class SemanticPOSSDistillationDataset(Dataset):
    """Dataset streaming SemanticPOSS point clouds with 4-class authoritative mapping."""
    def __init__(self, file_pairs: List[Tuple[str, str]], voxel_size: float = 0.08, augment: bool = False):
        self.file_pairs = file_pairs
        self.voxel_size = float(voxel_size)
        self.augment = augment
        self.range_filter = RangeFilter(min_range=0.5, max_range=100.0)
        self.input_adapter = SPVCNNInputAdapter(voxel_size=voxel_size)

    def __len__(self) -> int:
        return len(self.file_pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        bin_path, lbl_path = self.file_pairs[idx]
        pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        lbls_raw = np.fromfile(lbl_path, dtype=np.uint32)
        n = min(len(pts), len(lbls_raw))
        pts, lbls_raw = pts[:n].copy(), lbls_raw[:n].copy()

        # Authoritative SemanticPOSS 4-Class Mapping
        lbls = remap_poss_labels(lbls_raw)

        raw_frame = PointCloudFrame(points=pts, labels=lbls.astype(np.uint32))
        filt_frame, _ = self.range_filter.filter_frame(raw_frame)
        filt_pts = filt_frame.points.copy()
        filt_lbls = filt_frame.labels.copy()

        # Data augmentation on train
        if self.augment and len(filt_pts) > 0:
            theta = float(np.random.uniform(-np.pi / 12, np.pi / 12))
            c, s = float(np.cos(theta)), float(np.sin(theta))
            xyz = filt_pts[:, :3].copy()
            rot_x = xyz[:, 0] * c - xyz[:, 1] * s
            rot_y = xyz[:, 0] * s + xyz[:, 1] * c
            filt_pts[:, 0] = rot_x.astype(np.float32)
            filt_pts[:, 1] = rot_y.astype(np.float32)
            filt_pts[:, :3] += np.random.normal(0, 0.01, size=filt_pts[:, :3].shape).astype(np.float32)

        bundle = self.input_adapter.prepare_input(filt_pts)
        bundle["labels"] = torch.from_numpy(filt_lbls).long()
        bundle["file_path"] = bin_path
        return bundle

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return batch[0]

def train_distillation(
    teacher_ckpt: str = "checkpoints/best_spvcnn.pt",
    output_dir: str = "experiments/spvcnn_16ch_distilled",
    epochs: int = 5,
    lr: float = 0.003,
    voxel_size: float = 0.08,
    device: str = "cpu",
    max_train_frames: Optional[int] = 100,
    max_val_frames: Optional[int] = 20
):
    print("=" * 80)
    print("  SPVCNN 16-CHANNEL KNOWLEDGE DISTILLATION TRAINING ENGINE")
    print("=" * 80)

    dev = torch.device(device)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Instantiate & Load Frozen 32-channel Teacher
    print(f"Loading Teacher SPVCNN (32 channels) from {teacher_ckpt}...")
    teacher = build_spvcnn(num_classes=4, in_channels=4, pretrained_path=teacher_ckpt, device=dev)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Teacher Parameters: {sum(p.numel() for p in teacher.parameters()):,d} (FROZEN)")

    # 2. Instantiate 16-channel Student
    print(f"Initializing Student SPVCNN (16 channels, base_channels=16)...")
    student = SPVCNN(num_classes=4, in_channels=4, base_channels=16).to(dev)
    student_params = sum(p.numel() for p in student.parameters())
    print(f"Student Parameters: {student_params:,d} (Compression: {student_params / 138514 * 100:.1f}% of teacher)")

    # 3. Discover Dataset Manifest
    dataset_root = repo_root / "dataset"
    manifest = build_full_manifest(str(dataset_root))
    train_pairs = manifest.get("train_pairs", [])
    val_pairs = manifest.get("val_pairs", [])

    if not train_pairs:
        # Fallback to direct sequence 00 scan discovery
        velo_00 = sorted((dataset_root / "sequences/00/velodyne").glob("*.bin"))
        lbl_00 = sorted((dataset_root / "sequences/00/labels").glob("*.label"))
        train_pairs = [(str(v), str(l)) for v, l in zip(velo_00, lbl_00)]
        val_pairs = train_pairs[:20]

    if max_train_frames:
        train_pairs = train_pairs[:max_train_frames]
    if max_val_frames:
        val_pairs = val_pairs[:max_val_frames]

    print(f"Train Frames: {len(train_pairs)} | Validation Frames: {len(val_pairs)}")

    train_ds = SemanticPOSSDistillationDataset(train_pairs, voxel_size=voxel_size, augment=True)
    val_ds = SemanticPOSSDistillationDataset(val_pairs, voxel_size=voxel_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # 4. Class-balanced Weights & Loss
    weights = torch.tensor([2.0, 3.0, 1.0, 1.5], dtype=torch.float32, device=dev)
    criterion = DistillationLoss(alpha=0.5, temperature=3.0, class_weights=weights)
    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    evaluator = Phase2SemanticEvaluator(num_classes=4)

    best_miou = 0.0
    best_oa = 0.0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        student.train()
        train_loss = 0.0
        train_points = 0

        for b_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            feats = batch["features"].to(dev)
            p2v = batch["point_to_voxel_idx"].to(dev)
            n_vox = batch["num_voxels"]
            targets = batch["labels"].to(dev)

            # Teacher inference
            with torch.no_grad():
                teacher_logits = teacher(feats, p2v, n_vox)

            # Student forward
            student_logits = student(feats, p2v, n_vox)

            loss, loss_ce, loss_kd = criterion(student_logits, teacher_logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(targets)
            train_points += len(targets)

            if (b_idx + 1) % 25 == 0 or (b_idx + 1) == len(train_loader):
                cur_avg = train_loss / max(train_points, 1)
                print(f"  Epoch {ep:2d}/{epochs:2d} | Train Frame [{b_idx+1:3d}/{len(train_loader):3d}] | Loss: {cur_avg:.4f} (CE: {loss_ce.item():.4f}, KD: {loss_kd.item():.4f}) | {time.time()-t0:.1f}s")

        scheduler.step()
        train_loss_avg = train_loss / max(train_points, 1)

        # Validation Step
        student.eval()
        val_preds_list, val_targets_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                feats = batch["features"].to(dev)
                p2v = batch["point_to_voxel_idx"].to(dev)
                n_vox = batch["num_voxels"]
                targets = batch["labels"].numpy()

                logits = student(feats, p2v, n_vox)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=-1)

                val_preds_list.append(preds)
                val_targets_list.append(targets)

        val_preds_cat = np.concatenate(val_preds_list)
        val_targs_cat = np.concatenate(val_targets_list)

        metrics = evaluator.evaluate(val_preds_cat, val_targs_cat)
        val_miou = metrics.get("mIoU", 0.0)
        val_oa = metrics.get("overall_accuracy", 0.0)
        dt = time.time() - t0

        print(f"\n>> Epoch {ep:2d}/{epochs:2d} Summary [{dt:.1f}s] | Loss: {train_loss_avg:.4f} | Val OA: {val_oa*100:5.2f}% | Val mIoU: {val_miou*100:5.2f}%\n")

        if val_miou >= best_miou:
            best_miou = val_miou
            best_oa = val_oa
            ckpt_dict = {
                "epoch": ep,
                "model_state_dict": student.state_dict(),
                "val_miou": val_miou,
                "val_oa": val_oa,
                "model_type": "SPVCNN_16Class_Distilled",
                "base_channels": 16,
                "num_classes": 4
            }
            torch.save(ckpt_dict, out_dir / "best_model.pt")
            torch.save(ckpt_dict, repo_root / "checkpoints/spvcnn_student_16ch.pt")

    print("=" * 80)
    print(f"Distillation Complete! Best Val mIoU: {best_miou*100:.2f}% | Accuracy: {best_oa*100:.2f}%")
    print(f"Saved to {out_dir}/best_model.pt and checkpoints/spvcnn_student_16ch.pt")
    print("=" * 80)

    return {
        "best_miou": best_miou,
        "best_oa": best_oa,
        "checkpoint": str(repo_root / "checkpoints/spvcnn_student_16ch.pt")
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--voxel-size", type=float, default=0.08)
    parser.add_argument("--max-train", type=int, default=50)
    parser.add_argument("--max-val", type=int, default=10)
    args = parser.parse_args()

    train_distillation(
        epochs=args.epochs,
        lr=args.lr,
        voxel_size=args.voxel_size,
        max_train_frames=args.max_train,
        max_val_frames=args.max_val
    )
