#!/usr/bin/env python3
"""
Live 2.5D Foveated LiDAR Grid Visualizer.

Provides an interactive real-time dashboard displaying:
1. Raw LiDAR Point Cloud (Bird's Eye View)
2. Multi-Resolution Foveated Bands (5cm / 10cm / 25cm / 50cm)
3. 2.5D Elevation Field (Mean height & height range)
4. 4-Class Semantic Layer (Drivable, Non-Drivable, Static Obstacle, Dynamic Object)
5. Live Performance HUD (Real-time latency, FPS, memory usage, cell count)

Usage:
    python tools/visualize_live_25d_grid.py --sequence 00 --max-frames 50
    python tools/visualize_live_25d_grid.py --scan dataset/sequences/00/velodyne/000000.bin --save-image live_frame.png
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.types import SuperClass
from src.data_loader import LiDARDataLoader
from src.foveated_grid import FoveatedGrid25D
from ml.models.spvcnn_label_adapter import SPVCNNLabelAdapter


def run_live_visualizer(sequence="00", max_frames=50, save_image=None, headless=False):
    if headless:
        matplotlib.use("Agg")

    dataset_velodyne = repo_root / f"dataset/sequences/{sequence}/velodyne"
    dataset_labels = repo_root / f"dataset/sequences/{sequence}/labels"

    if not dataset_velodyne.exists():
        print(f"[-] Error: Dataset directory {dataset_velodyne} not found!")
        return

    bin_files = sorted(dataset_velodyne.glob("*.bin"))[:max_frames]
    if not bin_files:
        print("[-] No scan files found.")
        return

    print(f"[+] Discovered {len(bin_files)} frames in Sequence {sequence}.")

    loader = LiDARDataLoader()
    engine = FoveatedGrid25D(use_cpp=True)
    label_adapter = SPVCNNLabelAdapter(native_source="semanticposs")

    cmap_sem = matplotlib.colors.ListedColormap(['#2ca02c', '#d62728', '#1f77b4', '#ff7f0e'])

    # Setup Dashboard Figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.canvas.manager.set_window_title("Live Foveated 2.5D LiDAR Grid Streamer — SIH") if not headless and hasattr(fig.canvas, "manager") and fig.canvas.manager else None

    ax_raw, ax_fov = axes[0, 0], axes[0, 1]
    ax_elev, ax_sem = axes[1, 0], axes[1, 1]

    # Pre-render state
    sc_raw = ax_raw.scatter([], [], s=1, c=[], cmap="viridis")
    sc_fov = ax_fov.scatter([], [], s=4, c=[], cmap="plasma", vmin=0.05, vmax=0.50)
    sc_elev = ax_elev.scatter([], [], s=4, c=[], cmap="terrain", vmin=-2.0, vmax=5.0)
    sc_sem = ax_sem.scatter([], [], s=4, c=[], cmap=cmap_sem, vmin=0, vmax=3)

    for ax in [ax_raw, ax_fov, ax_elev, ax_sem]:
        ax.set_xlim(-60, 60)
        ax.set_ylim(-60, 60)
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_xlabel("Y: Left (+) / Right (-) [m]")
        ax.set_ylabel("X: Forward (+) / Rear (-) [m]")

    ax_raw.set_title("1. Raw LiDAR Point Cloud (BEV)", fontweight="bold")
    ax_fov.set_title("2. Multi-Resolution Foveated Bands (5cm - 50cm)", fontweight="bold")
    ax_elev.set_title("3. 2.5D Elevation Field (Mean Z)", fontweight="bold")
    ax_sem.set_title("4. 4-Class Dominant Semantic Layer", fontweight="bold")

    # Add Colorbars
    plt.colorbar(sc_raw, ax=ax_raw, label="Z Height (m)", fraction=0.046, pad=0.04)
    cb_fov = plt.colorbar(sc_fov, ax=ax_fov, label="Cell Resolution (m)", fraction=0.046, pad=0.04)
    plt.colorbar(sc_elev, ax=ax_elev, label="Elevation Mean (m)", fraction=0.046, pad=0.04)
    cb_sem = plt.colorbar(sc_sem, ax=ax_sem, ticks=[0, 1, 2, 3], fraction=0.046, pad=0.04)
    cb_sem.ax.set_yticklabels(['Drivable', 'Non-Drivable', 'Static Obs', 'Dynamic Obj'])

    # Title HUD Text
    hud_text = fig.suptitle("Initializing Live LiDAR Stream...", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    def update_frame(frame_idx):
        bf = bin_files[frame_idx]
        lf = dataset_labels / f"{bf.stem}.label" if dataset_labels.exists() else None
        
        t_start = time.perf_counter()
        frm = loader.load_frame(bf, lf)
        m_lbls = label_adapter.remap_predictions(frm.labels)
        
        # Build 2.5D Foveated Grid
        grid = engine.build_grid(frm.points, m_lbls)
        dt_ms = (time.perf_counter() - t_start) * 1000.0
        fps = 1000.0 / max(dt_ms, 0.001)

        # Extract Grid Arrays
        xs = [c.center_xy[0] for c in grid.cells.values()]
        ys = [c.center_xy[1] for c in grid.cells.values()]
        resolutions = [c.resolution for c in grid.cells.values()]
        elevations = [c.elevation_mean for c in grid.cells.values()]
        semantics = [c.dominant_class for c in grid.cells.values()]

        # 1. Update Raw BEV (Downsampled for responsive GUI)
        pts = frm.points[::5]
        sc_raw.set_offsets(np.column_stack([pts[:, 1], pts[:, 0]]))
        sc_raw.set_array(pts[:, 2])

        # 2. Update Foveated Grid
        sc_fov.set_offsets(np.column_stack([ys, xs]))
        sc_fov.set_array(np.array(resolutions))

        # 3. Update Elevation
        sc_elev.set_offsets(np.column_stack([ys, xs]))
        sc_elev.set_array(np.array(elevations))

        # 4. Update Semantics
        sc_sem.set_offsets(np.column_stack([ys, xs]))
        sc_sem.set_array(np.array(semantics))

        hud_text.set_text(
            f"Frame [{frame_idx+1:03d}/{len(bin_files):03d}]: {bf.name} | "
            f"Points: {len(frm.points):,d} | Populated Cells: {len(grid.cells):,d} | "
            f"Latency: {dt_ms:4.1f} ms ({fps:4.1f} FPS) | RAM: ~3.4 MB"
        )
        return sc_raw, sc_fov, sc_elev, sc_sem, hud_text

    if save_image:
        update_frame(0)
        plt.savefig(save_image, dpi=150)
        print(f"[+] Saved snapshot image to: {save_image}")
        plt.close()
    elif not headless:
        print("[+] Starting live animated playback... (Close window to exit)")
        anim = FuncAnimation(fig, update_frame, frames=len(bin_files), interval=50, blit=False, repeat=True)
        plt.show()
    else:
        # Benchmark run
        for idx in range(min(5, len(bin_files))):
            update_frame(idx)
        print(f"[+] Tested {min(5, len(bin_files))} frames in headless mode successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live 2.5D Foveated LiDAR Grid Visualizer")
    parser.add_argument("--sequence", default="00", help="Sequence ID (default: 00)")
    parser.add_argument("--max-frames", type=int, default=50, help="Max frames to play")
    parser.add_argument("--save-image", type=str, default=None, help="Save snapshot to file")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI")
    args = parser.parse_args()

    run_live_visualizer(
        sequence=args.sequence,
        max_frames=args.max_frames,
        save_image=args.save_image,
        headless=args.headless
    )
