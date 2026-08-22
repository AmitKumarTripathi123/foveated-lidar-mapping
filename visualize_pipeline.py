#!/usr/bin/env python3
"""
Dedicated Interactive Visualization & Debugging Tool for Foveated 2.5D LiDAR Mapping.

Usage Examples:
  # 1. Run all stages and generate full diagnostic suite & interactive HTML
  python visualize_pipeline.py --scan dataset/sequences/00/velodyne/000000.bin --stage all

  # 2. Trace a specific point index lifecycle
  python visualize_pipeline.py --scan dataset/sequences/00/velodyne/000000.bin --stage trace --point-idx 15243

  # 3. Profile SPVCNN internal layers and stages
  python visualize_pipeline.py --scan dataset/sequences/00/velodyne/000000.bin --profile-spvcnn

  # 4. Save all intermediate artifacts (.npz, .json, .png, .html)
  python visualize_pipeline.py --scan dataset/sequences/00/velodyne/000000.bin --save-debug ./debug_output/
"""

import argparse
import sys
from pathlib import Path

from visualization.pipeline_visualizer import PipelineVisualizer, CLASS_NAMES
from visualization.plot_generator import PipelinePlotGenerator
from visualization.interactive_html import generate_interactive_html_dashboard
from visualization.debug_exporter import DebugArtifactExporter


def print_banner():
    print("=" * 80)
    print("  FOVEATED 2.5D LiDAR PIPELINE — INTERACTIVE VISUALIZER & DEBUGGER")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="LiDAR Pipeline Visualizer & Debugger")
    parser.add_argument("--scan", type=str, default="dataset/sequences/00/velodyne/000000.bin", help="Path to input .bin scan")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "raw", "preprocess", "foveation", "voxel", "spvcnn", "grid", "trace", "compare"], help="Pipeline stage to visualize")
    parser.add_argument("--point-idx", type=int, default=150, help="Index of single LiDAR point to trace")
    parser.add_argument("--profile-spvcnn", action="store_true", help="Print detailed SPVCNN internal stage profiling")
    parser.add_argument("--save-debug", type=str, default="./debug_output", help="Directory to save debug artifacts")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch compute device")
    args = parser.parse_args()

    print_banner()
    print(f"[*] Input Scan: {args.scan}")
    print(f"[*] Selected Stage: {args.stage}")
    print(f"[*] Debug Output Directory: {args.save_debug}")
    print(f"[*] Hardware Device: {args.device}")

    scan_path = Path(args.scan)
    if not scan_path.exists():
        print(f"[-] Error: Scan file not found at {scan_path}", file=sys.stderr)
        sys.exit(1)

    # Initialize Visualizer
    print("\n[1/4] Initializing Pipeline Visualizer...")
    visualizer = PipelineVisualizer(device=args.device)

    # Process Scan
    print("[2/4] Executing Real Pipeline and Capturing Intermediate States...")
    state = visualizer.process_scan(scan_path, profile_internal=args.profile_spvcnn)
    trace = visualizer.trace_point(state, point_idx=args.point_idx)

    # Output Summary to Console
    print("\n" + "-" * 80)
    print("  STAGE-BY-STAGE PIPELINE SUMMARY")
    print("-" * 80)
    print(f"  1. Raw LiDAR Points       : {len(state.raw_points):,}")
    print(f"  2. Preprocessed Points    : {len(state.preprocessed_points):,} ({state.removed_out_of_range_count} out-of-range removed)")
    print(f"  3. Foveated Points        : {len(state.foveated_points):,} ({state.foveation_reduction_pct:.1f}% reduction)")
    print(f"  4. 3D Sparse Voxels       : {state.unique_voxel_count:,} (Sparsity: {state.active_voxel_ratio:.1f}%)")
    print(f"  5. SPVCNN Predictions     : {len(state.predicted_classes):,} points classified")
    for c_name, c_info in state.class_distribution.items():
        if c_info["class_id"] != 255:
            print(f"     - {c_name:22s}: {c_info['count']:6,d} ({c_info['percentage']:5.1f}%)")
    print(f"  6. Confidence Score       : Mean = {state.confidence_stats['mean']:.3f}, Min = {state.confidence_stats['min']:.3f}, Max = {state.confidence_stats['max']:.3f}")
    print(f"  7. GridMap25D Occupied    : {state.grid_occupied_cells:,} columnar cells")
    print(f"  8. Total Execution Time   : {state.total_pipeline_ms:.2f} ms ({state.effective_fps:.2f} FPS)")
    print("-" * 80)

    # Point Lifecycle Trace
    if args.stage in ("all", "trace"):
        print("\n" + "=" * 80)
        print(f"  SINGLE POINT LIFECYCLE TRACE (Point #{trace.point_index})")
        print("=" * 80)
        print(f"  • RAW Coordinates (X,Y,Z) : ({trace.raw_xyz[0]:.3f}, {trace.raw_xyz[1]:.3f}, {trace.raw_xyz[2]:.3f}) m (Range r = {trace.distance_r:.2f} m)")
        print(f"  • Preprocessing Filter    : {'RETAINED [0.5m ≤ r < 100m]' if trace.is_preprocessed else 'DISCARDED'}")
        print(f"  • Foveation Band          : {trace.band_name} ({'RETAINED' if trace.is_foveated_retained else 'DOWNSAMPLED'})")
        print(f"  • 64-Bit Voxel Coordinate : 3D = {trace.voxel_coords} | Packed Key = {hex(trace.packed_64bit_key) if trace.packed_64bit_key else 'N/A'}")
        print(f"  • SPVCNN Prediction       : {trace.predicted_class_name} (Confidence: {trace.confidence:.4f})")
        print(f"  • GridMap25D Cell         : ({trace.grid_cell_ix}, {trace.grid_cell_iy}) | Elevation: {trace.grid_cell_elevation_mean} m | Traversability: {trace.grid_cell_traversability}")
        print("=" * 80)

    # Generate Plots & HTML
    print("\n[3/4] Generating Diagnostic Plots and Interactive HTML Dashboard...")
    out_dir = Path(args.save_debug)
    plot_gen = PipelinePlotGenerator(output_dir=out_dir / "plots")
    plot_paths = plot_gen.generate_all_stage_plots(state, trace)

    html_path = generate_interactive_html_dashboard(state, output_path=out_dir / "pipeline_dashboard.html", trace=trace)

    # Save Debug Artifacts
    print("[4/4] Saving Compressed Arrays (.npz) and Metadata (.json)...")
    exporter = DebugArtifactExporter(output_dir=out_dir)
    debug_files = exporter.export_all(state, trace)

    print("\n" + "=" * 80)
    print("  VISUALIZATION & DEBUG ARTIFACTS GENERATED SUCCESSFULLY")
    print("=" * 80)
    print(f"  • Interactive Dashboard   : {html_path.resolve()}")
    print(f"  • Diagnostic Plots Folder : {(out_dir / 'plots').resolve()}")
    print(f"  • Intermediate Arrays     : {debug_files['npz'].resolve()}")
    print(f"  • Debug Summary JSON      : {debug_files['json'].resolve()}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
