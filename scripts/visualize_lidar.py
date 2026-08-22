#!/usr/bin/env python3
"""Lightweight LiDAR point cloud sanity visualizer for Phase 1.

Supports Open3D (if installed) or Matplotlib 3D/2D scatter plot as fallback.
Can also save a sanity visualization image without opening a GUI window.

Usage:
    python scripts/visualize_lidar.py \
        --scan dataset/sequences/00/velodyne/000000.bin \
        --save-image lidar_sanity_check.png
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.dataset import load_point_cloud, load_labels


def visualize_with_open3d(points: np.ndarray, labels: np.ndarray = None) -> bool:
    """Attempt visualization with Open3D."""
    try:
        import open3d as o3d
    except ImportError:
        return False

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    if labels is not None:
        # Simple color hashing for raw semantic IDs (Phase 1 sanity only)
        unique_labels = np.unique(labels)
        np.random.seed(42)
        color_map = {lbl: np.random.uniform(0.1, 0.9, size=3) for lbl in unique_labels}
        colors = np.array([color_map[lbl] for lbl in labels])
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Color by intensity
        intensities = points[:, 3]
        norm_int = (intensities - intensities.min()) / (intensities.max() - intensities.min() + 1e-6)
        colors = np.column_stack([norm_int, norm_int, norm_int])
        pcd.colors = o3d.utility.Vector3dVector(colors)

    print("Opening Open3D visualizer window...")
    o3d.visualization.draw_geometries([pcd], window_name="LiDAR Sanity Visualizer (Phase 1)")
    return True


def visualize_with_matplotlib(
    points: np.ndarray,
    labels: np.ndarray = None,
    save_path: str = None,
    max_points: int = 15000,
) -> None:
    """Fallback visualizer using Matplotlib."""
    import matplotlib.pyplot as plt

    # Downsample points for responsive plotting
    n_pts = len(points)
    if n_pts > max_points:
        idx = np.random.choice(n_pts, max_points, replace=False)
        sub_points = points[idx]
        sub_labels = labels[idx] if labels is not None else None
    else:
        sub_points = points
        sub_labels = labels

    x = sub_points[:, 0]
    y = sub_points[:, 1]
    z = sub_points[:, 2]
    intensity = sub_points[:, 3]

    fig = plt.figure(figsize=(14, 6))

    # Subplot 1: Bird's Eye View (X-Y)
    ax1 = fig.add_subplot(1, 2, 1)
    if sub_labels is not None:
        scatter1 = ax1.scatter(y, x, c=sub_labels, cmap="tab20", s=1, alpha=0.7)
        plt.colorbar(scatter1, ax=ax1, label="Raw Semantic Label ID")
    else:
        scatter1 = ax1.scatter(y, x, c=intensity, cmap="viridis", s=1, alpha=0.7)
        plt.colorbar(scatter1, ax=ax1, label="Intensity")
    ax1.set_title("LiDAR Top-Down View (Port/Starboard vs Forward)")
    ax1.set_xlabel("Y: Left (+) / Right (-) [m]")
    ax1.set_ylabel("X: Forward (+) / Rear (-) [m]")
    ax1.set_aspect("equal", "datalim")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Side Profile View (X-Z)
    ax2 = fig.add_subplot(1, 2, 2)
    if sub_labels is not None:
        scatter2 = ax2.scatter(x, z, c=sub_labels, cmap="tab20", s=1, alpha=0.7)
        plt.colorbar(scatter2, ax=ax2, label="Raw Semantic Label ID")
    else:
        scatter2 = ax2.scatter(x, z, c=intensity, cmap="viridis", s=1, alpha=0.7)
        plt.colorbar(scatter2, ax=ax2, label="Intensity")
    ax2.set_title("LiDAR Side Profile (Forward vs Height)")
    ax2.set_xlabel("X: Forward (+) / Rear (-) [m]")
    ax2.set_ylabel("Z: Upward (+) / Down (-) [m]")
    ax2.set_aspect("equal", "datalim")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_path:
        out_path = Path(save_path)
        plt.savefig(out_path, dpi=150)
        print(f"[OK] Sanity visualization saved to: {out_path.resolve()}")
    else:
        print("Displaying Matplotlib figure...")
        plt.show()


def main() -> int:
    """CLI entrypoint for visualization."""
    parser = argparse.ArgumentParser(description="LiDAR Sanity Visualizer (Phase 1)")
    parser.add_argument(
        "--scan",
        type=str,
        default="dataset/sequences/00/velodyne/000000.bin",
        help="Path to .bin file",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="dataset/sequences/00/labels/000000.label",
        help="Path to .label file",
    )
    parser.add_argument(
        "--save-image",
        type=str,
        default=None,
        help="Path to save output visualization image (headless friendly)",
    )
    parser.add_argument(
        "--prefer-open3d",
        action="store_true",
        help="Attempt Open3D interactive viewer first",
    )

    args = parser.parse_args()

    points = load_point_cloud(args.scan)
    labels = load_labels(args.label) if Path(args.label).is_file() else None

    print(f"Loaded {len(points):,} points from {args.scan}")
    if labels is not None:
        print(f"Loaded {len(labels):,} labels from {args.label}")

    if args.prefer_open3d:
        if visualize_with_open3d(points, labels):
            return 0
        print("[!] Open3D unavailable. Falling back to Matplotlib...")

    visualize_with_matplotlib(points, labels, save_path=args.save_image)
    return 0


if __name__ == "__main__":
    sys.exit(main())
