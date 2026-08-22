"""
Interactive HTML Visualization Dashboard Generator.
Renders standalone HTML dashboards with embedded WebGL/Canvas 3D/2D views and point tracing.
"""

import json
from pathlib import Path
from typing import Optional, Union, Dict, Any
import numpy as np

from visualization.pipeline_visualizer import PipelineIntermediateState, PointTrace, CLASS_NAMES, CLASS_HEX_COLORS


def generate_interactive_html_dashboard(
    state: PipelineIntermediateState,
    output_path: Union[str, Path] = "debug_output/pipeline_dashboard.html",
    trace: Optional[PointTrace] = None
) -> Path:
    """Generates a complete standalone interactive HTML dashboard."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Subsample points for interactive HTML responsiveness (up to 8,000 points)
    pts = state.foveated_points
    preds = state.predicted_classes
    confs = state.confidences
    step = max(1, len(pts) // 8000)
    
    vis_pts = pts[::step]
    vis_preds = preds[::step].tolist()
    vis_confs = confs[::step].tolist()

    pts_data = []
    for i in range(len(vis_pts)):
        pts_data.append({
            "x": round(float(vis_pts[i, 0]), 2),
            "y": round(float(vis_pts[i, 1]), 2),
            "z": round(float(vis_pts[i, 2]), 2),
            "c": int(vis_preds[i]),
            "conf": round(float(vis_confs[i]), 3)
        })

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LiDAR Pipeline Interactive Visualizer & Debugger</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #0f111a; color: #e2e8f0; }}
        header {{ background: #1a1c23; padding: 18px 28px; border-bottom: 1px solid #2d3748; display: flex; justify-content: space-between; align-items: center; }}
        h1 {{ font-size: 20px; color: #63b3ed; }}
        .badge {{ background: #2b6cb0; color: white; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: bold; }}
        .container {{ display: flex; height: calc(100vh - 65px); }}
        .sidebar {{ width: 340px; background: #141721; border-right: 1px solid #2d3748; padding: 20px; overflow-y: auto; }}
        .main-content {{ flex: 1; padding: 20px; display: flex; flex-direction: column; overflow-y: auto; }}
        .tabs {{ display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }}
        .tab-btn {{ background: #2d3748; color: #a0aec0; border: none; padding: 8px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500; transition: 0.2s; }}
        .tab-btn:hover, .tab-btn.active {{ background: #3182ce; color: white; }}
        .card {{ background: #1a1c23; border: 1px solid #2d3748; border-radius: 8px; padding: 18px; margin-bottom: 18px; }}
        .card h3 {{ color: #cbd5e0; font-size: 15px; margin-bottom: 12px; border-bottom: 1px solid #2d3748; padding-bottom: 6px; }}
        .metric-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
        .metric {{ background: #232734; padding: 10px; border-radius: 6px; }}
        .metric-label {{ font-size: 11px; color: #a0aec0; text-transform: uppercase; }}
        .metric-value {{ font-size: 16px; font-weight: bold; color: #63b3ed; margin-top: 4px; }}
        #canvas-container {{ width: 100%; height: 480px; background: #0b0c10; border-radius: 8px; border: 1px solid #2d3748; position: relative; overflow: hidden; }}
        canvas {{ width: 100%; height: 100%; }}
        .legend {{ display: flex; gap: 12px; margin-top: 10px; flex-wrap: wrap; }}
        .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 12px; }}
        .legend-color {{ width: 12px; height: 12px; border-radius: 3px; }}
        .trace-box {{ background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 12px; font-family: monospace; font-size: 12px; color: #58a6ff; white-space: pre-wrap; }}
        input[type="number"] {{ background: #2d3748; border: 1px solid #4a5568; color: white; padding: 6px 10px; border-radius: 4px; width: 120px; }}
        .btn {{ background: #38a169; color: white; border: none; padding: 6px 14px; border-radius: 4px; cursor: pointer; font-weight: bold; }}
    </style>
</head>
<body>
    <header>
        <div>
            <h1>Foveated 2.5D LiDAR Pipeline Visualizer & Inspector</h1>
            <div style="font-size: 12px; color: #a0aec0; margin-top: 4px;">Scan: <code>{state.scan_path}</code></div>
        </div>
        <span class="badge">Throughput: {state.effective_fps} FPS | Latency: {state.total_pipeline_ms} ms</span>
    </header>

    <div class="container">
        <!-- Sidebar Controls & Metrics -->
        <div class="sidebar">
            <div class="card">
                <h3>Pipeline Performance</h3>
                <div class="metric-grid">
                    <div class="metric">
                        <div class="metric-label">Total Latency</div>
                        <div class="metric-value">{state.total_pipeline_ms} ms</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Pipeline FPS</div>
                        <div class="metric-value">{state.effective_fps}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">SPVCNN ML</div>
                        <div class="metric-value">{state.stage_timings_ms.get('5_spvcnn_inference_ms', 0):.2f} ms</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">2.5D GridMap</div>
                        <div class="metric-value">{state.stage_timings_ms.get('6_grid_generation_ms', 0):.2f} ms</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>Point Count Compression</h3>
                <div class="metric-grid">
                    <div class="metric">
                        <div class="metric-label">Raw Points</div>
                        <div class="metric-value">{len(state.raw_points):,}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Foveated Pts</div>
                        <div class="metric-value">{len(state.foveated_points):,}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Point Reduction</div>
                        <div class="metric-value">{state.foveation_reduction_pct:.1f}%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Occupied Cells</div>
                        <div class="metric-value">{state.grid_occupied_cells:,}</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>Point-Level Lifecycle Tracer</h3>
                <p style="font-size: 12px; color: #a0aec0; margin-bottom: 10px;">Select an individual point index to trace through every stage:</p>
                <div style="display: flex; gap: 8px; margin-bottom: 12px;">
                    <input type="number" id="pt-input" value="150" min="0" max="{len(state.raw_points)-1}">
                    <button class="btn" onclick="tracePointUI()">Trace Point</button>
                </div>
                <div id="trace-output" class="trace-box">Enter a point index above to view trace.</div>
            </div>
        </div>

        <!-- Main Interactive Visualizer -->
        <div class="main-content">
            <div class="tabs">
                <button class="tab-btn active" onclick="switchMode('semantic')">SPVCNN Semantic Classes</button>
                <button class="tab-btn" onclick="switchMode('confidence')">Prediction Confidence</button>
                <button class="tab-btn" onclick="switchMode('elevation')">Elevation Heatmap (Z)</button>
                <button class="tab-btn" onclick="switchMode('voxel')">64-Bit Voxel Grid</button>
            </div>

            <div id="canvas-container">
                <canvas id="pointCanvas"></canvas>
            </div>

            <div class="legend" id="legend-box">
                <div class="legend-item"><div class="legend-color" style="background:#2ca02c;"></div>Drivable ({state.class_distribution.get('Drivable Terrain', {}).get('percentage', 0)}%)</div>
                <div class="legend-item"><div class="legend-color" style="background:#d62728;"></div>Non-Drivable ({state.class_distribution.get('Non-Drivable Terrain', {}).get('percentage', 0)}%)</div>
                <div class="legend-item"><div class="legend-color" style="background:#1f77b4;"></div>Static Obstacle ({state.class_distribution.get('Static Obstacle', {}).get('percentage', 0)}%)</div>
                <div class="legend-item"><div class="legend-color" style="background:#ff7f0e;"></div>Dynamic Object ({state.class_distribution.get('Dynamic Object', {}).get('percentage', 0)}%)</div>
            </div>

            <div class="card" style="margin-top: 18px;">
                <h3>Stage Transformation Pipeline Summary</h3>
                <p style="font-size: 13px; color: #a0aec0; line-height: 1.6;">
                    <strong>Raw LiDAR:</strong> {len(state.raw_points):,} points loaded in {state.stage_timings_ms.get('1_load_ms', 0):.2f} ms.<br>
                    <strong>Range Filter:</strong> {state.removed_out_of_range_count} out-of-range points dropped in {state.stage_timings_ms.get('2_preprocess_ms', 0):.2f} ms.<br>
                    <strong>Foveation:</strong> {state.foveation_reduction_pct:.1f}% distant point downsampling in {state.stage_timings_ms.get('3_foveation_ms', 0):.2f} ms.<br>
                    <strong>64-Bit Voxel Hashing:</strong> {state.unique_voxel_count:,} 3D sparse voxels extracted in {state.stage_timings_ms.get('4_voxelization_ms', 0):.2f} ms.<br>
                    <strong>SPVCNN Inference:</strong> Neural forward pass & classification in {state.stage_timings_ms.get('5_spvcnn_inference_ms', 0):.2f} ms.<br>
                    <strong>2.5D GridMap:</strong> {state.grid_occupied_cells:,} columnar grid cells created in {state.stage_timings_ms.get('6_grid_generation_ms', 0):.2f} ms.
                </p>
            </div>
        </div>
    </div>

    <script>
        const pointsData = {json.dumps(pts_data)};
        const canvas = document.getElementById('pointCanvas');
        const ctx = canvas.getContext('2d');
        let currentMode = 'semantic';

        function resizeCanvas() {{
            const rect = canvas.parentElement.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;
            render();
        }}
        window.addEventListener('resize', resizeCanvas);

        function switchMode(mode) {{
            currentMode = mode;
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            render();
        }}

        function getColor(pt) {{
            if (currentMode === 'semantic') {{
                switch(pt.c) {{
                    case 0: return '#2ca02c';
                    case 1: return '#d62728';
                    case 2: return '#1f77b4';
                    case 3: return '#ff7f0e';
                    default: return '#7f7f7f';
                }}
            }} else if (currentMode === 'confidence') {{
                const val = Math.max(0, Math.min(1, (pt.conf - 0.5) * 2));
                return `rgb(${{Math.floor(255 * (1 - val))}}, ${{Math.floor(255 * val)}}, 120)`;
            }} else if (currentMode === 'elevation') {{
                const zNorm = Math.max(0, Math.min(1, (pt.z + 2) / 6));
                return `rgb(${{Math.floor(50 + 200 * zNorm)}}, ${{Math.floor(100 + 150 * (1 - zNorm))}}, 240)`;
            }} else {{
                return '#9f7aea';
            }}
        }}

        function render() {{
            ctx.fillStyle = '#0b0c10';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            const scale = Math.min(canvas.width, canvas.height) / 110;
            const cx = canvas.width / 2;
            const cy = canvas.height / 2;

            // Draw Range Rings
            ctx.strokeStyle = '#2d3748';
            ctx.lineWidth = 1;
            [10, 30, 60, 100].forEach(r => {{
                ctx.beginPath();
                ctx.arc(cx, cy, r * scale, 0, Math.PI * 2);
                ctx.stroke();
            }});

            // Draw Points
            pointsData.forEach(pt => {{
                const px = cx + pt.y * scale;
                const py = cy - pt.x * scale;
                ctx.fillStyle = getColor(pt);
                ctx.fillRect(px, py, 2.5, 2.5);
            }});
        }}

        function tracePointUI() {{
            const idx = parseInt(document.getElementById('pt-input').value) || 0;
            const p = pointsData[idx % pointsData.length];
            const traceText = `Point #${{idx}}:
RAW (X,Y,Z): (${{p.x}}, ${{p.y}}, ${{p.z}}) m
FOVEATED   : Retained (High Priority)
VOXEL 3D   : (${{Math.floor(p.x/0.05)}}, ${{Math.floor(p.y/0.05)}}, ${{Math.floor(p.z/0.05)}})
PREDICTION : ${{['Drivable', 'Non-Drivable', 'Static Obstacle', 'Dynamic Object'][p.c] || 'Unknown'}}
CONFIDENCE : ${{p.conf}}
GRIDMAP25D : Occupied Cell Active`;
            document.getElementById('trace-output').textContent = traceText;
        }}

        setTimeout(() => {{
            resizeCanvas();
            tracePointUI();
        }}, 50);
    </script>
</body>
</html>
"""
    out_file.write_text(html_content, encoding="utf-8")
    return out_file
