"""
High-Resolution Diagnostic Figure & Visualization Generator.
Produces individual and complete multi-panel diagnostic PNG figures for every stage.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.types import SuperClass
from src.foveated_grid import DEFAULT_FROZEN_BANDS
from visualization.pipeline_visualizer import PipelineIntermediateState, PointTrace, CLASS_HEX_COLORS, CLASS_NAMES


class PipelinePlotGenerator:
    """Generates visual figures for each stage of the LiDAR processing pipeline."""

    def __init__(self, output_dir: Union[str, Path] = "debug_output/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_stage_plots(self, state: PipelineIntermediateState, trace: Optional[PointTrace] = None) -> Dict[str, Path]:
        """Renders and saves all individual diagnostic plots for each stage."""
        paths = {}
        paths["raw"] = self.plot_raw_lidar(state)
        paths["preprocess"] = self.plot_preprocessing(state)
        paths["foveation"] = self.plot_foveation(state)
        paths["voxelization"] = self.plot_voxelization(state)
        paths["spvcnn_prediction"] = self.plot_spvcnn_predictions(state)
        paths["confidence"] = self.plot_confidence(state)
        paths["gridmap"] = self.plot_gridmap25d(state)
        paths["pipeline_flow"] = self.plot_pipeline_flow(state)
        if trace is not None:
            paths["point_trace"] = self.plot_point_trace(trace)
        return paths

    def plot_raw_lidar(self, state: PipelineIntermediateState) -> Path:
        """Stage 1: Raw LiDAR 3D and BEV Distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="#0f111a")
        for ax in axes:
            ax.set_facecolor("#1a1c23")

        pts = state.raw_points
        sub = pts[::max(1, len(pts) // 15000)]

        # BEV Scatter
        sc = axes[0].scatter(sub[:, 0], sub[:, 1], c=sub[:, 2], cmap="viridis", s=1.5, alpha=0.8)
        axes[0].set_title(f"Raw LiDAR BEV Scan ({len(pts):,} points)", color="white", fontsize=14, pad=10)
        axes[0].set_xlabel("X (meters, forward)", color="#a0aec0")
        axes[0].set_ylabel("Y (meters, left/right)", color="#a0aec0")
        axes[0].tick_params(colors="#a0aec0")
        axes[0].axis("equal")
        axes[0].grid(True, linestyle="--", alpha=0.2, color="#718096")
        cb = plt.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.04)
        cb.set_label("Elevation Z (m)", color="#a0aec0")
        cb.ax.tick_params(colors="#a0aec0")

        # Elevation Profile
        axes[1].hist(pts[:, 2], bins=60, color="#4299e1", edgecolor="#2b6cb0", alpha=0.85)
        axes[1].set_title(f"Elevation Z Histogram (Z ∈ [{state.raw_bounds['z'][0]:.2f}, {state.raw_bounds['z'][1]:.2f}] m)", color="white", fontsize=14, pad=10)
        axes[1].set_xlabel("Elevation Z (meters)", color="#a0aec0")
        axes[1].set_ylabel("Point Count", color="#a0aec0")
        axes[1].tick_params(colors="#a0aec0")
        axes[1].grid(True, linestyle="--", alpha=0.2, color="#718096")

        out_p = self.output_dir / "stage1_raw_lidar.png"
        plt.tight_layout()
        plt.savefig(out_p, dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)
        return out_p

    def plot_preprocessing(self, state: PipelineIntermediateState) -> Path:
        """Stage 2: Preprocessing Range Filtering (Retained vs Removed)."""
        fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0f111a")
        ax.set_facecolor("#1a1c23")

        raw_p = state.raw_points[::max(1, len(state.raw_points) // 10000)]
        dists = np.sqrt(raw_p[:, 0]**2 + raw_p[:, 1]**2)
        valid = (dists >= 0.5) & (dists < 100.0)

        ax.scatter(raw_p[valid, 0], raw_p[valid, 1], c="#48bb78", s=2.0, alpha=0.7, label=f"Retained Points ({len(state.preprocessed_points):,})")
        if np.any(~valid):
            ax.scatter(raw_p[~valid, 0], raw_p[~valid, 1], c="#f56565", s=8.0, alpha=0.9, marker="x", label=f"Removed Out-of-Range ({state.removed_out_of_range_count:,})")

        # Range filter circle (100m)
        circle = plt.Circle((0, 0), 100.0, color="#ecc94b", fill=False, linestyle="--", linewidth=1.5, label="Max Range (100m)")
        ax.add_patch(circle)

        ax.set_title("Stage 2: Preprocessing Range Filter (0.5m ≤ r < 100m)", color="white", fontsize=14, pad=10)
        ax.set_xlabel("X (meters)", color="#a0aec0")
        ax.set_ylabel("Y (meters)", color="#a0aec0")
        ax.tick_params(colors="#a0aec0")
        ax.axis("equal")
        ax.legend(facecolor="#2d3748", edgecolor="#4a5568", labelcolor="white", loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.2, color="#718096")

        out_p = self.output_dir / "stage2_preprocessing.png"
        plt.tight_layout()
        plt.savefig(out_p, dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)
        return out_p

    def plot_foveation(self, state: PipelineIntermediateState) -> Path:
        """Stage 3: Multi-Band Foveation Spatial Distribution."""
        fig, ax = plt.subplots(figsize=(11, 9), facecolor="#0f111a")
        ax.set_facecolor("#1a1c23")

        pts = state.foveated_points
        sub = pts[::max(1, len(pts) // 15000)]
        dists = np.sqrt(sub[:, 0]**2 + sub[:, 1]**2)

        band_colors = {"near_field": "#38a169", "mid_field": "#3182ce", "far_field": "#dd6b20", "ultra_far": "#805ad5"}
        for band in DEFAULT_FROZEN_BANDS:
            mask = (dists >= band.min_range) & (dists < band.max_range)
            if np.any(mask):
                ax.scatter(sub[mask, 0], sub[mask, 1], c=band_colors.get(band.name, "#718096"), s=2.5, alpha=0.8,
                           label=f"{band.name} [{band.min_range}-{band.max_range}m, res={band.voxel_size}m]")
            ring = plt.Circle((0, 0), band.max_range, color="#e2e8f0", fill=False, linestyle=":", linewidth=1.0, alpha=0.6)
            ax.add_patch(ring)

        ax.set_title(f"Stage 3: Foveation Multi-Band Allocation ({len(pts):,} pts, {state.foveation_reduction_pct:.1f}% reduction)", color="white", fontsize=14, pad=10)
        ax.set_xlabel("X (meters)", color="#a0aec0")
        ax.set_ylabel("Y (meters)", color="#a0aec0")
        ax.tick_params(colors="#a0aec0")
        ax.axis("equal")
        ax.legend(facecolor="#2d3748", edgecolor="#4a5568", labelcolor="white", loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.2, color="#718096")

        out_p = self.output_dir / "stage3_foveation.png"
        plt.tight_layout()
        plt.savefig(out_p, dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)
        return out_p

    def plot_voxelization(self, state: PipelineIntermediateState) -> Path:
        """Stage 4: Voxelization & 64-Bit Packed Integer Hashing."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="#0f111a")
        for ax in axes:
            ax.set_facecolor("#1a1c23")

        # Voxel Sparsity Bar
        cats = ["Raw Points", "Foveated Points", "Unique 3D Voxels"]
        vals = [len(state.raw_points), len(state.foveated_points), state.unique_voxel_count]
        colors = ["#4299e1", "#48bb78", "#9f7aea"]
        bars = axes[0].bar(cats, vals, color=colors, width=0.5, edgecolor="#cbd5e0")
        axes[0].set_title("Voxel Compression & Sparsity", color="white", fontsize=14, pad=10)
        axes[0].set_ylabel("Element Count", color="#a0aec0")
        axes[0].tick_params(colors="#a0aec0")
        axes[0].grid(True, linestyle="--", alpha=0.2, color="#718096")
        for bar in bars:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., h + 1000, f"{h:,}", ha="center", va="bottom", color="white", fontweight="bold")

        # 64-bit packed integer explanation table
        axes[1].axis("off")
        axes[1].set_title("64-Bit Integer Spatial Hash Transformation", color="white", fontsize=14, pad=10)
        table_text = "3D Voxel Coordinate  -->  64-Bit Integer Key  -->  Decoded Voxel (100% Match)\n"
        table_text += "=" * 70 + "\n\n"
        for s in state.packed_keys_sample:
            orig = f"({s['original_voxel'][0]:4d}, {s['original_voxel'][1]:4d}, {s['original_voxel'][2]:4d})"
            key = f"{s['packed_64bit_key']:>18s}"
            rec = f"({s['recovered_voxel'][0]:4d}, {s['recovered_voxel'][1]:4d}, {s['recovered_voxel'][2]:4d})"
            status = "VERIFIED [OK]" if s["verified"] else "ERROR"
            table_text += f"{orig}  -->  {key}  -->  {rec}  [{status}]\n"
        axes[1].text(0.05, 0.85, table_text, color="#68d391", fontsize=11, family="monospace", va="top")

        out_p = self.output_dir / "stage4_voxelization.png"
        plt.tight_layout()
        plt.savefig(out_p, dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)
        return out_p

    def plot_spvcnn_predictions(self, state: PipelineIntermediateState) -> Path:
        """Stage 6: SPVCNN Semantic Predictions."""
        fig, ax = plt.subplots(figsize=(11, 9), facecolor="#0f111a")
        ax.set_facecolor("#1a1c23")

        pts = state.foveated_points
        preds = state.predicted_classes
        sub_idx = np.arange(0, len(pts), max(1, len(pts) // 20000))

        handles = []
        for c_id, c_name in CLASS_NAMES.items():
            if c_id == 255:
                continue
            mask = (preds[sub_idx] == c_id)
            if np.any(mask):
                c_hex = CLASS_HEX_COLORS.get(c_id, "#718096")
                ax.scatter(pts[sub_idx][mask, 0], pts[sub_idx][mask, 1], c=c_hex, s=3.0, alpha=0.85)
                handles.append(mpatches.Patch(color=c_hex, label=f"{c_name} ({state.class_distribution[c_name]['percentage']}%)"))

        ax.set_title("Stage 6: SPVCNN Semantic Predictions (4 Navigation Classes)", color="white", fontsize=14, pad=10)
        ax.set_xlabel("X (meters, forward)", color="#a0aec0")
        ax.set_ylabel("Y (meters, left/right)", color="#a0aec0")
        ax.tick_params(colors="#a0aec0")
        ax.axis("equal")
        ax.legend(handles=handles, facecolor="#2d3748", edgecolor="#4a5568", labelcolor="white", loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.2, color="#718096")

        out_p = self.output_dir / "stage6_spvcnn_prediction.png"
        plt.tight_layout()
        plt.savefig(out_p, dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)
        return out_p

    def plot_confidence(self, state: PipelineIntermediateState) -> Path:
        """Stage 7: Prediction Confidence Heatmap."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="#0f111a")
        for ax in axes:
            ax.set_facecolor("#1a1c23")

        pts = state.foveated_points
        conf = state.confidences
        sub_idx = np.arange(0, len(pts), max(1, len(pts) // 15000))

        sc = axes[0].scatter(pts[sub_idx, 0], pts[sub_idx, 1], c=conf[sub_idx], cmap="magma", s=2.5, vmin=0.5, vmax=1.0, alpha=0.85)
        axes[0].set_title(f"Prediction Confidence Heatmap (Mean: {state.confidence_stats['mean']:.3f})", color="white", fontsize=14, pad=10)
        axes[0].set_xlabel("X (meters)", color="#a0aec0")
        axes[0].set_ylabel("Y (meters)", color="#a0aec0")
        axes[0].tick_params(colors="#a0aec0")
        axes[0].axis("equal")
        axes[0].grid(True, linestyle="--", alpha=0.2, color="#718096")
        cb = plt.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.04)
        cb.set_label("Confidence Probability", color="#a0aec0")
        cb.ax.tick_params(colors="#a0aec0")

        # Distribution histogram
        axes[1].hist(conf, bins=40, color="#ed8936", edgecolor="#c05621", alpha=0.85)
        axes[1].set_title("Confidence Distribution Histogram", color="white", fontsize=14, pad=10)
        axes[1].set_xlabel("Confidence Score", color="#a0aec0")
        axes[1].set_ylabel("Point Count", color="#a0aec0")
        axes[1].tick_params(colors="#a0aec0")
        axes[1].grid(True, linestyle="--", alpha=0.2, color="#718096")

        out_p = self.output_dir / "stage7_confidence.png"
        plt.tight_layout()
        plt.savefig(out_p, dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)
        return out_p

    def plot_gridmap25d(self, state: PipelineIntermediateState) -> Path:
        """Stage 8: 2.5D Occupancy and Traversability GridMap."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="#0f111a")
        for ax in axes:
            ax.set_facecolor("#1a1c23")

        df = state.grid_map.to_dataframe()
        if len(df) > 0:
            cx = df["ix"].values * df["resolution"].values
            cy = df["iy"].values * df["resolution"].values
            elev = df["elevation_mean"].values
            c_ids = df["semantic_class"].values

            # Elevation view
            sc = axes[0].scatter(cx, cy, c=elev, cmap="terrain", s=3.0, alpha=0.85)
            axes[0].set_title(f"GridMap25D Elevation Layer ({len(df):,} occupied cells)", color="white", fontsize=14, pad=10)
            axes[0].set_xlabel("X (meters)", color="#a0aec0")
            axes[0].set_ylabel("Y (meters)", color="#a0aec0")
            axes[0].tick_params(colors="#a0aec0")
            axes[0].axis("equal")
            axes[0].grid(True, linestyle="--", alpha=0.2, color="#718096")
            cb = plt.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.04)
            cb.set_label("Cell Elevation (m)", color="#a0aec0")
            cb.ax.tick_params(colors="#a0aec0")

            # Traversability View
            trav = df["traversability"].values
            sc2 = axes[1].scatter(cx, cy, c=trav, cmap="RdYlGn", s=3.0, alpha=0.85, vmin=0.0, vmax=1.0)
            axes[1].set_title("GridMap25D Traversability Layer (1=Free, 0=Obstacle)", color="white", fontsize=14, pad=10)
            axes[1].set_xlabel("X (meters)", color="#a0aec0")
            axes[1].set_ylabel("Y (meters)", color="#a0aec0")
            axes[1].tick_params(colors="#a0aec0")
            axes[1].axis("equal")
            axes[1].grid(True, linestyle="--", alpha=0.2, color="#718096")
            cb2 = plt.colorbar(sc2, ax=axes[1], fraction=0.046, pad=0.04)
            cb2.set_label("Traversability Score", color="#a0aec0")
            cb2.ax.tick_params(colors="#a0aec0")

        out_p = self.output_dir / "stage8_gridmap25d.png"
        plt.tight_layout()
        plt.savefig(out_p, dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)
        return out_p

    def plot_pipeline_flow(self, state: PipelineIntermediateState) -> Path:
        """Stage 10: Complete Pipeline Flow & Timing Breakdown."""
        fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0f111a")
        ax.set_facecolor("#1a1c23")

        stages = ["1. Load", "2. Preprocess", "3. Foveation", "4. Voxelize", "5. SPVCNN", "6. GridMap"]
        t_vals = [
            state.stage_timings_ms["1_load_ms"],
            state.stage_timings_ms["2_preprocess_ms"],
            state.stage_timings_ms["3_foveation_ms"],
            state.stage_timings_ms["4_voxelization_ms"],
            state.stage_timings_ms["5_spvcnn_inference_ms"],
            state.stage_timings_ms["6_grid_generation_ms"]
        ]
        colors = ["#4299e1", "#48bb78", "#38b2ac", "#9f7aea", "#ed8936", "#e53e3e"]

        bars = ax.barh(stages, t_vals, color=colors, edgecolor="#cbd5e0", height=0.55)
        ax.set_title(f"Pipeline Stage Latency Breakdown (Total: {state.total_pipeline_ms:.2f} ms | {state.effective_fps:.2f} FPS)", color="white", fontsize=14, pad=10)
        ax.set_xlabel("Latency (milliseconds)", color="#a0aec0")
        ax.tick_params(colors="#a0aec0")
        ax.grid(True, linestyle="--", alpha=0.2, color="#718096")

        for bar in bars:
            w = bar.get_width()
            pct = (w / max(state.total_pipeline_ms, 1e-4)) * 100.0
            ax.text(w + 1.0, bar.get_y() + bar.get_height()/2., f"{w:.2f} ms ({pct:.1f}%)", ha="left", va="center", color="white", fontweight="bold")

        out_p = self.output_dir / "stage10_pipeline_performance.png"
        plt.tight_layout()
        plt.savefig(out_p, dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)
        return out_p

    def plot_point_trace(self, trace: PointTrace) -> Path:
        """Stage 9: Single Point Lifecycle Trace Card."""
        fig, ax = plt.subplots(figsize=(12, 7), facecolor="#0f111a")
        ax.axis("off")
        ax.set_facecolor("#1a1c23")

        card_text = f"""
========================================================================================
   LIDAR POINT LIFECYCLE TRACE (Point #{trace.point_index})
========================================================================================
1. RAW LiDAR SENSOR:
   Coordinates (X, Y, Z)     : ({trace.raw_xyz[0]:.3f}, {trace.raw_xyz[1]:.3f}, {trace.raw_xyz[2]:.3f}) meters
   Intensity                 : {trace.raw_intensity:.2f}
   Range Distance (r)        : {trace.distance_r:.3f} meters

2. PREPROCESSING:
   Passed 0.5m-100m Range    : {'YES [RETAINED]' if trace.is_preprocessed else 'NO [DISCARDED]'}

3. MULTI-BAND FOVEATION:
   Assigned Foveation Band   : {trace.band_name}
   Retained in Foveated Scan : {'YES' if trace.is_foveated_retained else 'NO (Downsampled)'}

4. 64-BIT PACKED VOXELIZATION:
   Voxel Grid Index (ix,iy,iz): {trace.voxel_coords}
   64-Bit Packed Integer Key : {hex(trace.packed_64bit_key) if trace.packed_64bit_key else 'N/A'}
   Decoded Coordinate Match  : {trace.recovered_voxel_coords == trace.voxel_coords}

5. SPVCNN NEURAL PREDICTION:
   Semantic Super-Class      : {trace.predicted_class_name} (Class ID: {trace.predicted_class_id})
   Confidence Score          : {trace.confidence:.4f}
   Class Probabilities       : [Drivable: {trace.class_probabilities[0]:.3f}, Non-Drivable: {trace.class_probabilities[1]:.3f}, Obstacle: {trace.class_probabilities[2]:.3f}, Dynamic: {trace.class_probabilities[3]:.3f}]

6. 2.5D GRIDMAP INTEGRATION:
   Grid Cell Index (ix, iy)  : ({trace.grid_cell_ix}, {trace.grid_cell_iy})
   Cell Mean Elevation       : {trace.grid_cell_elevation_mean} m
   Cell Traversability Score : {trace.grid_cell_traversability} (1=Free, 0=Blocked)
========================================================================================
"""
        ax.text(0.05, 0.95, card_text, color="#63b3ed", fontsize=11, family="monospace", va="top")

        out_p = self.output_dir / f"stage9_point_trace_{trace.point_index}.png"
        plt.tight_layout()
        plt.savefig(out_p, dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)
        return out_p
