"""
Visualization Exporter Module.
Generates comprehensive 2D Birds-Eye-View (BEV), 2.5D Elevation, Semantic Super-class,
and Before-vs-After comparison plots.
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches

from src.types import PointCloudFrame, SuperClass, FoveationBand
from src.metrics.elevation_preservation import ElevationPreservationValidator


# Super-class color palette
CLASS_COLORS = {
    SuperClass.DRIVABLE_TERRAIN: "#2ca02c",       # Green
    SuperClass.NON_DRIVABLE_TERRAIN: "#d62728",   # Crimson Red / Orange
    SuperClass.STATIC_OBSTACLE: "#1f77b4",        # Blue
    SuperClass.DYNAMIC_OBJECT: "#ff7f0e",         # Amber / Orange
    SuperClass.IGNORE_LABEL: "#7f7f7f"            # Gray
}

CLASS_NAMES = {
    SuperClass.DRIVABLE_TERRAIN: "Drivable Terrain",
    SuperClass.NON_DRIVABLE_TERRAIN: "Non-Drivable Terrain",
    SuperClass.STATIC_OBSTACLE: "Static Obstacle",
    SuperClass.DYNAMIC_OBJECT: "Dynamic Object",
    SuperClass.IGNORE_LABEL: "Ignore / Outlier"
}


class VisualizationExporter:
    """
    Generates publication-quality diagnostic visualizations for LiDAR processing.
    """

    def __init__(self, output_dir: Union[str, Path] = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_comparison_suite(
        self,
        raw_frame: PointCloudFrame,
        filtered_frame: PointCloudFrame,
        foveated_frame: PointCloudFrame,
        bands: List[FoveationBand],
        prefix: str = "frame_000000"
    ) -> Dict[str, str]:
        """
        Generates and saves the full comparison suite.
        Returns paths to generated image files.
        """
        out_paths = {}

        # 1. 4-Panel Pipeline Progression (Raw -> Filtered -> Foveated -> Semantic)
        p1 = self.output_dir / f"{prefix}_pipeline_progression.png"
        self._plot_pipeline_progression(raw_frame, filtered_frame, foveated_frame, p1)
        out_paths["pipeline_progression"] = str(p1)

        # 2. Multi-Band Foveation Breakdown (with range rings)
        p2 = self.output_dir / f"{prefix}_foveation_bands.png"
        self._plot_foveation_bands(filtered_frame, foveated_frame, bands, p2)
        out_paths["foveation_bands"] = str(p2)

        # 3. 2.5D Elevation Heatmap Comparison
        p3 = self.output_dir / f"{prefix}_elevation_comparison.png"
        self._plot_elevation_comparison(filtered_frame, foveated_frame, p3)
        out_paths["elevation_comparison"] = str(p3)

        # 4. Obstacle & Dynamic Object Detail
        p4 = self.output_dir / f"{prefix}_obstacle_preservation.png"
        self._plot_obstacle_preservation(filtered_frame, foveated_frame, p4)
        out_paths["obstacle_preservation"] = str(p4)

        return out_paths

    def _plot_pipeline_progression(
        self,
        raw: PointCloudFrame,
        filt: PointCloudFrame,
        fov: PointCloudFrame,
        save_path: Path
    ):
        fig, axes = plt.subplots(1, 4, figsize=(24, 6), facecolor="#111111")

        def _scatter_bev(ax, frame, title, color_mode="height"):
            ax.set_facecolor("#181818")
            pts = frame.points
            if pts is not None and len(pts) > 0:
                x = pts[:, 0]
                y = pts[:, 1]
                z = pts[:, 2]

                if color_mode == "height":
                    sc = ax.scatter(y, x, c=z, s=0.4, cmap="viridis", vmin=-2.0, vmax=3.0, alpha=0.8)
                elif color_mode == "semantic":
                    lbls = frame.labels if frame.labels is not None else np.zeros(len(pts))
                    colors = [CLASS_COLORS.get(l, "#7f7f7f") for l in lbls]
                    ax.scatter(y, x, c=colors, s=0.8, alpha=0.9)

            ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)
            ax.set_xlabel("+Y (Left / Right) [m]", color="#cccccc")
            ax.set_ylabel("+X (Forward) [m]", color="#cccccc")
            ax.set_xlim(-60, 60)
            ax.set_ylim(-20, 105)
            ax.tick_params(colors="#cccccc")
            ax.grid(True, color="#333333", linestyle="--", alpha=0.5)

        _scatter_bev(axes[0], raw, f"1. Raw LiDAR Cloud ({raw.num_points:,} pts)")
        _scatter_bev(axes[1], filt, f"2. Range-Filtered (0-100m) ({filt.num_points:,} pts)")
        _scatter_bev(axes[2], fov, f"3. Foveated Cloud ({fov.num_points:,} pts, {round((1 - fov.num_points/max(raw.num_points, 1))*100, 1)}% red.)")
        _scatter_bev(axes[3], fov, "4. Semantic Super-Classes (Foveated Representation)", color_mode="semantic")

        # Add custom legend for semantic classes
        legend_elements = [
            patches.Patch(facecolor=color, edgecolor="none", label=CLASS_NAMES[c])
            for c, color in CLASS_COLORS.items()
        ]
        axes[3].legend(handles=legend_elements, loc="upper right", facecolor="#222222", edgecolor="#555555", labelcolor="white", fontsize=9)

        plt.suptitle("Foveated LiDAR Pipeline: Progression & Compression Flow", color="white", fontsize=16, y=1.03, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    def _plot_foveation_bands(
        self,
        filt: PointCloudFrame,
        fov: PointCloudFrame,
        bands: List[FoveationBand],
        save_path: Path
    ):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor="#111111")

        for idx, (ax, frame, title) in enumerate([
            (axes[0], filt, f"Uniform Range-Filtered ({filt.num_points:,} pts)"),
            (axes[1], fov, f"Distance-Aware Foveated ({fov.num_points:,} pts)")
        ]):
            ax.set_facecolor("#181818")
            pts = frame.points
            if pts is not None and len(pts) > 0:
                x = pts[:, 0]
                y = pts[:, 1]
                z = pts[:, 2]
                ax.scatter(y, x, c=z, s=0.5, cmap="plasma", vmin=-2.0, vmax=3.0, alpha=0.7)

            # Draw range band rings
            for b in bands:
                theta = np.linspace(-np.pi, np.pi, 200)
                circ_x = b.max_range * np.cos(theta)
                circ_y = b.max_range * np.sin(theta)
                # Keep forward/visible sector
                ax.plot(circ_y, circ_x, color="#00ffcc", linestyle=":", linewidth=1.2, alpha=0.8)
                ax.text(0, b.max_range - 2, f"{b.name} ({b.voxel_size}m voxel)",
                        color="#00ffcc", fontsize=10, ha="center", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="#000000", alpha=0.7, edgecolor="none"))

            ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=10)
            ax.set_xlabel("+Y (Left / Right) [m]", color="#cccccc")
            ax.set_ylabel("+X (Forward) [m]", color="#cccccc")
            ax.set_xlim(-60, 60)
            ax.set_ylim(-10, 105)
            ax.tick_params(colors="#cccccc")
            ax.grid(True, color="#333333", linestyle="--", alpha=0.5)

        plt.suptitle("Distance-Aware Foveation Band Partitioning (0-10m, 10-40m, 40-100m)", color="white", fontsize=16, y=0.98, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    def _plot_elevation_comparison(
        self,
        filt: PointCloudFrame,
        fov: PointCloudFrame,
        save_path: Path
    ):
        validator = ElevationPreservationValidator(grid_resolution=0.25, max_range=100.0)
        report = validator.evaluate(filt, fov, return_grids=True)

        fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor="#111111")

        # Helper to crop grids to forward quadrant [-50, 50] lateral, [0, 100] forward
        # Grid range is [-100, 100]
        res = 0.25
        idx_x0 = int((0 - (-100)) / res)
        idx_x1 = int((100 - (-100)) / res)
        idx_y0 = int((-50 - (-100)) / res)
        idx_y1 = int((50 - (-100)) / res)

        raw_sub = report.raw_elevation_grid[idx_x0:idx_x1, idx_y0:idx_y1]
        fov_sub = report.fov_elevation_grid[idx_x0:idx_x1, idx_y0:idx_y1]
        diff_sub = np.abs(raw_sub - fov_sub)

        extent = [-50, 50, 0, 100]

        im0 = axes[0].imshow(np.rot90(raw_sub), extent=extent, cmap="terrain", vmin=-2.0, vmax=4.0, origin="lower")
        axes[0].set_title("Raw 2.5D Elevation Grid (Z max)", color="white", fontsize=13, fontweight="bold")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04).ax.tick_params(colors="white")

        im1 = axes[1].imshow(np.rot90(fov_sub), extent=extent, cmap="terrain", vmin=-2.0, vmax=4.0, origin="lower")
        axes[1].set_title("Foveated 2.5D Elevation Grid (Z max)", color="white", fontsize=13, fontweight="bold")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04).ax.tick_params(colors="white")

        im2 = axes[2].imshow(np.rot90(diff_sub), extent=extent, cmap="inferno", vmin=0.0, vmax=0.5, origin="lower")
        axes[2].set_title(f"Elevation Error (|Z_raw - Z_fov|)\nRMSE: {report.overall_rmse}m, p95: {report.overall_p95_error}m",
                          color="white", fontsize=13, fontweight="bold")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04).ax.tick_params(colors="white")

        for ax in axes:
            ax.set_facecolor("#000000")
            ax.set_xlabel("+Y (Left / Right) [m]", color="#cccccc")
            ax.set_ylabel("+X (Forward) [m]", color="#cccccc")
            ax.tick_params(colors="#cccccc")

        plt.suptitle("2.5D Elevation Preservation & Vertical Fidelity Analysis", color="white", fontsize=16, y=0.98, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    def _plot_obstacle_preservation(
        self,
        filt: PointCloudFrame,
        fov: PointCloudFrame,
        save_path: Path
    ):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor="#111111")

        def _scatter_obstacles(ax, frame, title):
            ax.set_facecolor("#181818")
            pts = frame.points
            lbls = frame.labels if frame.labels is not None else np.zeros(len(pts))

            # Plot ground in faint dark green
            ground_mask = (lbls == SuperClass.DRIVABLE_TERRAIN) | (lbls == SuperClass.NON_DRIVABLE_TERRAIN)
            if np.any(ground_mask):
                ax.scatter(pts[ground_mask, 1], pts[ground_mask, 0], c="#1b431b", s=0.2, alpha=0.3, label="Ground")

            # Static obstacles in cyan
            static_mask = (lbls == SuperClass.STATIC_OBSTACLE)
            if np.any(static_mask):
                ax.scatter(pts[static_mask, 1], pts[static_mask, 0], c="#00ffff", s=1.5, alpha=0.9, label="Static Obstacle")

            # Dynamic objects in neon red/orange
            dyn_mask = (lbls == SuperClass.DYNAMIC_OBJECT)
            if np.any(dyn_mask):
                ax.scatter(pts[dyn_mask, 1], pts[dyn_mask, 0], c="#ff3366", s=3.0, alpha=1.0, label="Dynamic Object")

            ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=10)
            ax.set_xlabel("+Y (Left / Right) [m]", color="#cccccc")
            ax.set_ylabel("+X (Forward) [m]", color="#cccccc")
            ax.set_xlim(-50, 50)
            ax.set_ylim(-5, 100)
            ax.tick_params(colors="#cccccc")
            ax.grid(True, color="#333333", linestyle="--", alpha=0.5)
            ax.legend(loc="upper right", facecolor="#222222", edgecolor="#555555", labelcolor="white", fontsize=10)

        _scatter_obstacles(axes[0], filt, f"Raw Obstacle Layout ({filt.num_points:,} pts)")
        _scatter_obstacles(axes[1], fov, f"Obstacle-Preserved Foveation ({fov.num_points:,} pts)")

        plt.suptitle("Obstacle & Dynamic Object Spatial Coverage (Obstacle-Preserving Policy)", color="white", fontsize=16, y=0.98, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
