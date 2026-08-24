"""
Canonical SPVCNN Inference Predictor (SIH PS 26130).
Loads certified frozen checkpoint with cryptographic SHA256 validation.
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from ml.data.spvcnn_adapter import SPVCNNInputAdapter
from ml.models.spvcnn import SPVCNN, build_spvcnn


def verify_sha256(filepath: Path, expected_hash: str) -> bool:
    """Validate SHA256 checksum of checkpoint file."""
    if not filepath.is_file():
        return False
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest() == expected_hash


class CanonicalPredictor:
    """Production SPVCNN predictor executing GPU tensor-core accelerated inference."""

    def __init__(self, config_path: Union[str, Path] = "configs/system_config.yaml"):
        cfg_file = Path(config_path)
        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        m_cfg = self.cfg.get("model", {})
        self.device = torch.device(m_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

        # Resolve checkpoint path relative to repo root
        root = cfg_file.parent.parent
        ckpt_p = root / m_cfg.get("checkpoint_path", "experiments/phase12_full_semanticposs_spvcnn/best_checkpoint.pt")
        expected_sha = m_cfg.get("checkpoint_sha256", "b15c6dfb2f20d1dce4febc47be67f9d50b86a0af72f1895176c6a6ee58bca142")

        if not verify_sha256(ckpt_p, expected_sha):
            raise ValueError(f"CRITICAL: Checkpoint SHA256 mismatch for {ckpt_p}!")

        if m_cfg.get("allow_tf32", True) and self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.model = build_spvcnn(
            num_classes=m_cfg.get("num_classes", 4),
            in_channels=m_cfg.get("in_channels", 4),
            pretrained_path=str(ckpt_p),
            device=self.device,
        )
        self.model.eval()
        self.input_adapter = SPVCNNInputAdapter(voxel_size=0.05)

    def predict(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform semantic point-cloud inference.

        Args:
            points: (N, 4) float32 array [x, y, z, intensity].

        Returns:
            Tuple[predicted_classes (N,), confidences (N,)]
        """
        if points.shape[0] == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)

        pts_tensor = torch.from_numpy(points).to(self.device).float()
        bundle = self.input_adapter.prepare_input(pts_tensor, device=self.device)

        with torch.inference_mode():
            logits = self.model(
                features=bundle["features"],
                point_to_voxel_idx=bundle["point_to_voxel_idx"],
                num_voxels=bundle["num_voxels"],
            )
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1).cpu().numpy().astype(np.int64)
            confs = torch.max(probs, dim=-1).values.cpu().numpy().astype(np.float32)

        return preds, confs
