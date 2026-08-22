"""
Foveated Point Cloud Semantic Segmentation Network (FoveatedPointSegNet).
Lightweight, distance-aware multi-scale point-wise segmentation network for 4-class navigation prediction:
0: drivable_terrain, 1: non_drivable_terrain, 2: static_obstacle, 3: dynamic_object.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.ln1 = nn.LayerNorm(channels)
        self.fc2 = nn.Linear(channels, channels)
        self.ln2 = nn.LayerNorm(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.ln1(self.fc1(x)))
        out = self.ln2(self.fc2(out))
        return self.act(out + residual)


class FoveatedPointSegNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        embed_dim: int = 64,
        hidden_dims: Tuple[int, ...] = (128, 256, 128)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Input projection takes [x_norm, y_norm, z_norm, intensity, r_norm] (5 features)
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels + 1, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        self.enc1 = nn.Sequential(
            nn.Linear(embed_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            ResidualBlock(hidden_dims[0])
        )

        self.enc2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
            ResidualBlock(hidden_dims[1])
        )

        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU()
        )

        dec_in = hidden_dims[1] * 2 + hidden_dims[0]
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, hidden_dims[2]),
            nn.LayerNorm(hidden_dims[2]),
            nn.GELU(),
            ResidualBlock(hidden_dims[2]),
            nn.Linear(hidden_dims[2], 64),
            nn.LayerNorm(64),
            nn.GELU()
        )

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        orig_dim = points.dim()
        if orig_dim == 2:
            pts = points
            N = pts.shape[0]
            if N == 0:
                return torch.empty((0, self.num_classes), device=points.device, dtype=points.dtype)

            # Feature normalization for robust multi-scale perception
            x_norm = pts[:, 0:1] / 50.0
            y_norm = pts[:, 1:2] / 50.0
            z_norm = pts[:, 2:3] / 3.0
            i_norm = pts[:, 3:4]
            r = torch.sqrt(pts[:, 0:1]**2 + pts[:, 1:2]**2) / 50.0

            feat = torch.cat([x_norm, y_norm, z_norm, i_norm, r], dim=-1)

            f0 = self.input_proj(feat)
            f1 = self.enc1(f0)
            f2 = self.enc2(f1)

            global_max = torch.max(f2, dim=0, keepdim=True)[0].expand(N, -1)
            global_feat = self.global_mlp(global_max)

            f_cat = torch.cat([f2, global_feat, f1], dim=-1)
            f_dec = self.decoder(f_cat)
            logits = self.classifier(f_dec)
            return logits

        elif orig_dim == 3:
            B, N, C = points.shape
            pts_flat = points.view(-1, C)
            logits_flat = self.forward(pts_flat)
            return logits_flat.view(B, N, self.num_classes)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got dim={orig_dim}")

    @torch.no_grad()
    def predict(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.eval()
        logits = self.forward(points)
        probs = F.softmax(logits, dim=-1)
        conf, pred_cls = torch.max(probs, dim=-1)
        return {
            "logits": logits,
            "probabilities": probs,
            "predicted_class": pred_cls,
            "confidence": conf
        }
