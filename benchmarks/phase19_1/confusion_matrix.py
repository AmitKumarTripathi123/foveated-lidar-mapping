"""
Phase 19.1 Confusion Matrix Generator & Heatmap Visualizer.
Exports raw and normalized confusion matrices for Global, Near, Mid, and Far zones.
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CLASS_LABELS = ["Drivable", "Non-Drivable", "Static", "Dynamic"]


def normalize_confusion_matrix(cm: np.ndarray) -> np.ndarray:
    """Normalize confusion matrix row-wise (recall/true class distribution)."""
    row_sums = cm.sum(axis=1, keepdims=True).astype(np.float64)
    row_sums[row_sums == 0] = 1.0
    return cm / row_sums


def plot_4panel_confusion_matrices(
    cms: Dict[str, np.ndarray],
    out_png: Path,
):
    """Render a 4-panel normalized confusion matrix heatmap figure."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=150)
    fig.suptitle("SIH PS 26130 — Global & Distance-Stratified Confusion Matrices (Phase 19.1)", fontsize=14, fontweight="bold")

    panel_keys = [
        ("global", "1. Global Confusion Matrix (0–100m)"),
        ("near_0_10m", "2. Near Zone (0–10m @ 5cm)"),
        ("mid_10_40m", "3. Mid Zone (10–40m @ 15cm)"),
        ("far_40_100m", "4. Far Zone (40–100m @ 50cm)"),
    ]

    for idx, (key, title) in enumerate(panel_keys):
        ax = axes[idx // 2, idx % 2]
        cm = cms.get(key, np.zeros((4, 4), dtype=np.int64))
        norm_cm = normalize_confusion_matrix(cm)

        im = ax.imshow(norm_cm, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(CLASS_LABELS, fontsize=9)
        ax.set_yticklabels(CLASS_LABELS, fontsize=9)
        ax.set_xlabel("Predicted Class", fontsize=10)
        ax.set_ylabel("True Class", fontsize=10)

        # Annotate text
        for i in range(4):
            for j in range(4):
                val = norm_cm[i, j]
                count = cm[i, j]
                text_color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val*100:.1f}%\n({count:,})", ha="center", va="center", color=text_color, fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_png, dpi=150)
    plt.close()
