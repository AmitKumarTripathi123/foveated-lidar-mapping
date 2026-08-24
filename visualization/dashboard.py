"""
Canonical Dashboard Visualizer (SIH PS 26130).
Single entry point generating live multi-panel diagnostic figures and interactive HTML HUD.
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.core.lidar_loader import load_lidar_points
from src.core.types import CLASS_HEX_COLORS, CLASS_NAMES
from src.inference.pipeline import FoveatedPipeline, PipelineResult


def render_canonical_dashboard(
    res: PipelineResult,
    raw_pts: np.ndarray,
    out_png: Path,
):
    """Render canonical 6-panel diagnostic visualization."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(20, 13), dpi=150)
    fig.suptitle(
        f"SIH PS 26130 — Canonical Foveated 2.5D LiDAR Perception & Mapping Architecture (Phase 18)\n"
        f"Total End-to-End Latency: {res.total_latency_ms:.1f} ms | Retained Points: {res.foveated_points_count:,} / {res.raw_points_count:,}",
        fontsize=15, fontweight="bold", y=0.98,
    )

    # 1. Raw LiDAR Scan
    ax1 = axes[0, 0]
    ax1.scatter(raw_pts[:, 0], raw_pts[:, 1], s=0.5, c=raw_pts[:, 2], cmap="viridis", alpha=0.6)
    ax1.set_title(f"1. Raw LiDAR Scan ({res.raw_points_count:,} pts)", fontsize=11, fontweight="bold")
    ax1.set_xlim([-55, 55])
    ax1.set_ylim([-55, 55])
    ax1.set_aspect("equal")
    ax1.set_xlabel("X (meters)")
    ax1.set_ylabel("Y (meters)")
    ax1.grid(True, linestyle="--", alpha=0.3)

    # 2. Canonical 3-Zone Foveation
    ax2 = axes[0, 1]
    dists = np.sqrt(raw_pts[:, 0]**2 + raw_pts[:, 1]**2 + raw_pts[:, 2]**2)
    near_m = dists < 10.0
    mid_m = (dists >= 10.0) & (dists < 40.0)
    far_m = (dists >= 40.0) & (dists <= 100.0)

    ax2.scatter(raw_pts[far_m, 0], raw_pts[far_m, 1], s=0.8, c="#1f77b4", label="Far (40-100m @ 50cm)", alpha=0.5)
    ax2.scatter(raw_pts[mid_m, 0], raw_pts[mid_m, 1], s=1.0, c="#ff7f0e", label="Mid (10-40m @ 15cm)", alpha=0.7)
    ax2.scatter(raw_pts[near_m, 0], raw_pts[near_m, 1], s=1.5, c="#2ca02c", label="Near (0-10m @ 5cm)", alpha=0.9)

    for r, col in [(10.0, "#2ca02c"), (40.0, "#ff7f0e"), (100.0, "#1f77b4")]:
        circle = plt.Circle((0, 0), r, color=col, fill=False, linestyle="--", linewidth=1.5)
        ax2.add_patch(circle)

    ax2.set_title(f"2. Canonical 3-Zone Foveation ({res.foveated_points_count:,} pts)", fontsize=11, fontweight="bold")
    ax2.set_xlim([-55, 55])
    ax2.set_ylim([-55, 55])
    ax2.set_aspect("equal")
    ax2.legend(loc="upper right", fontsize=7)
    ax2.grid(True, linestyle="--", alpha=0.3)

    # 3. SPVCNN Semantic Point Cloud
    ax3 = axes[0, 2]
    if res.foveated_xyz is not None and res.predicted_classes is not None:
        for c_id in [0, 1, 2, 3]:
            c_mask = res.predicted_classes == c_id
            if np.any(c_mask):
                ax3.scatter(
                    res.foveated_xyz[c_mask, 0], res.foveated_xyz[c_mask, 1],
                    s=1.2, c=CLASS_HEX_COLORS[c_id], label=CLASS_NAMES[c_id], alpha=0.8,
                )
    ax3.set_title("3. SPVCNN Semantic Predictions", fontsize=11, fontweight="bold")
    ax3.set_xlim([-55, 55])
    ax3.set_ylim([-55, 55])
    ax3.set_aspect("equal")
    ax3.legend(loc="upper right", fontsize=7)
    ax3.grid(True, linestyle="--", alpha=0.3)

    # 4. 2.5D Elevation Grid Map
    ax4 = axes[1, 0]
    elev_masked = np.ma.masked_invalid(res.grid_map.elevation_mean)
    im4 = ax4.imshow(elev_masked, origin="lower", extent=[-50, 50, -50, 50], cmap="terrain")
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, label="Elevation Mean (m)")
    ax4.set_title("4. 2.5D Mean Elevation Grid Map", fontsize=11, fontweight="bold")
    ax4.set_xlabel("X (meters)")
    ax4.set_ylabel("Y (meters)")
    ax4.grid(True, linestyle="--", alpha=0.3)

    # 5. Traversability Grid Map Layer
    ax5 = axes[1, 1]
    trav_masked = np.ma.masked_where(res.grid_map.traversability_layer < -0.5, res.grid_map.traversability_layer)
    im5 = ax5.imshow(trav_masked, origin="lower", extent=[-50, 50, -50, 50], cmap="RdYlGn", vmin=-1.0, vmax=1.0)
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04, label="Traversability (+1=Drivable, 0=Obstacle)")
    ax5.set_title("5. 2.5D Traversability Layer", fontsize=11, fontweight="bold")
    ax5.set_xlabel("X (meters)")
    ax5.set_ylabel("Y (meters)")
    ax5.grid(True, linestyle="--", alpha=0.3)

    # 6. Latency Breakdown HUD
    ax6 = axes[1, 2]
    stage_names = list(res.stage_latencies_ms.keys())
    lat_vals = list(res.stage_latencies_ms.values())

    bars = ax6.barh(stage_names, lat_vals, color="#3b82f6", alpha=0.85)
    ax6.set_xlabel("Latency (ms)")
    ax6.set_title(f"6. Pipeline Latency Telemetry ({res.total_latency_ms:.1f} ms Total)", fontsize=11, fontweight="bold")
    ax6.grid(True, linestyle="--", alpha=0.4, axis="x")

    for bar in bars:
        w = bar.get_width()
        ax6.text(w + 0.3, bar.get_y() + bar.get_height()/2, f"{w:.1f} ms", va="center", ha="left", fontsize=8, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_png, dpi=150)
    plt.close()


def generate_canonical_html(res: PipelineResult, out_html: Path):
    """Generate canonical standalone interactive HTML dashboard."""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SIH PS 26130 — Canonical Architecture Dashboard (Phase 18)</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
    .header {{ background: #1e293b; padding: 20px; border-radius: 8px; margin-bottom: 24px; border-left: 6px solid #3b82f6; }}
    .header h1 {{ margin: 0 0 6px 0; font-size: 22px; }}
    .header p {{ margin: 0; color: #94a3b8; font-size: 14px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
    .card {{ background: #1e293b; border-radius: 8px; padding: 20px; border: 1px solid #334155; }}
    .card h2 {{ margin: 0 0 16px 0; font-size: 16px; color: #60a5fa; border-bottom: 1px solid #334155; padding-bottom: 8px; }}
    .row {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #0f172a; }}
    .lbl {{ color: #94a3b8; }}
    .val {{ font-weight: bold; color: #38bdf8; }}
    .badge {{ background: #166534; color: #bbf7d0; padding: 3px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>SIH PS 26130 — Canonical Architecture Dashboard</h1>
    <p>Single Source of Truth (system_config.yaml) | Status: <span class="badge">CANONICAL ARCHITECTURE FROZEN</span></p>
  </div>
  <div class="grid">
    <div class="card">
      <h2>Pipeline Performance</h2>
      <div class="row"><span class="lbl">Total Latency</span><span class="val">{res.total_latency_ms:.1f} ms</span></div>
      <div class="row"><span class="lbl">Effective FPS</span><span class="val">{1000.0/res.total_latency_ms:.1f} FPS</span></div>
      <div class="row"><span class="lbl">Raw LiDAR Points</span><span class="val">{res.raw_points_count:,}</span></div>
      <div class="row"><span class="lbl">Foveated Retained</span><span class="val">{res.foveated_points_count:,}</span></div>
    </div>
    <div class="card">
      <h2>Canonical 3-Zone Geometry</h2>
      <div class="row"><span class="lbl">Near Zone (0-10m)</span><span class="val">0.05 m (5 cm)</span></div>
      <div class="row"><span class="lbl">Mid Zone (10-40m)</span><span class="val">0.15 m (15 cm)</span></div>
      <div class="row"><span class="lbl">Far Zone (40-100m)</span><span class="val">0.50 m (50 cm)</span></div>
      <div class="row"><span class="lbl">Grid Dimensions</span><span class="val">500 x 500 (4.77 MB)</span></div>
    </div>
  </div>
</body>
</html>
"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_doc)


def main():
    config_path = "configs/system_config.yaml"
    pipeline = FoveatedPipeline(config_path)
    seq_dir = Path("dataset/sequences/02/velodyne")
    bin_files = sorted(list(seq_dir.glob("*.bin")))
    if not bin_files:
        raise FileNotFoundError(f"No .bin files found in {seq_dir}")
    sample_bin = bin_files[0]
    pts = load_lidar_points(sample_bin)

    print("Executing canonical pipeline and generating dashboard...")
    res = pipeline.run(pts)

    out_png = Path("reports/phase18/figures/canonical_dashboard.png")
    out_html = Path("reports/phase18/canonical_dashboard.html")

    render_canonical_dashboard(res, pts, out_png)
    generate_canonical_html(res, out_html)

    print(f"Dashboard Figure saved: {out_png}")
    print(f"Interactive HTML saved: {out_html}")


if __name__ == "__main__":
    main()

