"""
Main CLI Pipeline Driver for Phase 1 LiDAR Data Validation & Foveation.
Executes the full pipeline:
Raw LiDAR -> Validation -> Standardization -> Range Filtering -> Foveation -> Preservation Metrics -> Benchmarking -> Visualization -> Reporting
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import yaml
from tabulate import tabulate

from src.types import PointCloudFrame, AggregationPolicy, ValidationPolicy
from src.data_loader import LiDARDataLoader
from src.validator import PointCloudValidator
from src.label_mapper import LabelMapper
from src.range_filter import RangeFilter
from src.foveation import FoveatedVoxelizer
from src.metrics.elevation_preservation import ElevationPreservationValidator
from src.metrics.obstacle_preservation import ObstaclePreservationValidator
from src.metrics.semantic_preservation import SemanticPreservationValidator
from src.benchmark import PerformanceBenchmark
from src.visualization import VisualizationExporter
from src.report_generator import Phase1ReportGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: LiDAR Data Validation & Foveated Pipeline (Smart India Hackathon)"
    )
    parser.add_argument("--dataset-path", type=str, default="data/synthetic_sequence", help="Path to LiDAR dataset sequence")
    parser.add_argument("--sequence", type=str, default="00", help="Sequence ID")
    parser.add_argument("--config", type=str, default="configs/foveation_default.yaml", help="Default foveation configuration YAML")
    parser.add_argument("--candidates-config", type=str, default="configs/foveation_candidates.yaml", help="Candidate configurations YAML")
    parser.add_argument("--mapping-config", type=str, default="configs/semantickitti_mapping.yaml", help="Label mapping YAML")
    parser.add_argument("--policy", type=str, default="obstacle_preserving", choices=["nearest", "centroid", "majority", "confidence_weighted", "obstacle_preserving"])
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory for reports")
    parser.add_argument("--vis-dir", type=str, default="visualizations", help="Output directory for visualizations")
    parser.add_argument("--max-frames", type=int, default=10, help="Maximum frames to process")
    parser.add_argument("--repeats", type=int, default=5, help="Benchmark repeat count per frame")
    parser.add_argument("--export-vis", action="store_true", default=True, help="Export visualization plots")
    parser.add_argument("--export-reports", action="store_true", default=True, help="Generate JSON and Markdown reports")

    args = parser.parse_args()

    print("=" * 80)
    print("  FOVEATED 2.5D LiDAR: Phase 1 Data Validation & Foveated Pipeline")
    print("  Autonomous Navigation - Smart India Hackathon")
    print("=" * 80)

    # 1. Initialize Modules
    print(f"\n[1/8] Initializing Modules...")
    loader = LiDARDataLoader(
        dataset_path=args.dataset_path,
        sequence_id=args.sequence,
        validation_policy=ValidationPolicy.SKIP_AND_WARN
    )

    with open(args.config, "r") as f:
        default_cfg = yaml.safe_load(f)

    max_range = float(default_cfg.get("max_range", 100.0))
    validator = PointCloudValidator(max_allowed_range=max_range)
    label_mapper = LabelMapper(mapping_config_path=args.mapping_config)
    range_filter = RangeFilter(min_range=0.0, max_range=max_range)
    foveator = FoveatedVoxelizer(config_path=args.config, default_policy=AggregationPolicy(args.policy))

    elevation_val = ElevationPreservationValidator(grid_resolution=0.20, max_range=max_range)
    obstacle_val = ObstaclePreservationValidator(grid_resolution=0.25, max_range=max_range)
    semantic_val = SemanticPreservationValidator()
    benchmark_runner = PerformanceBenchmark(max_range=max_range)
    vis_exporter = VisualizationExporter(output_dir=args.vis_dir)
    report_gen = Phase1ReportGenerator(output_dir=args.output_dir)

    print(f"  -> Loader: Dataset {args.dataset_path}, Seq {args.sequence}")
    print(f"  -> Range Filter Max: {max_range} m")
    print(f"  -> Foveation Bands: {[b.name for b in foveator.bands]}")
    print(f"  -> Label Mapping: {label_mapper.dataset_name}")
    if label_mapper.has_mapping_warning:
        print(f"  -> [WARNING]: {label_mapper.mapping_warning_msg}")

    # 2. Discover and Load Frames
    print(f"\n[2/8] Loading and Validating Frames...")
    frames = list(loader.iterate_frames())[: args.max_frames]
    if not frames:
        print(f"ERROR: No frames discovered in {args.dataset_path}/sequences/{args.sequence}")
        sys.exit(1)

    print(f"  -> Successfully loaded {len(frames)} frames.")
    invalid_count = len(loader.invalid_frames)
    if invalid_count > 0:
        print(f"  -> Handled {invalid_count} invalid frames according to validation policy.")

    # 3. Comprehensive Diagnostics on Primary Frame (Frame 0)
    print(f"\n[3/8] Running Point Cloud & Semantic Diagnostics on Primary Frame...")
    primary_frame = frames[0]
    val_summary = validator.validate_frame(primary_frame)
    label_report = label_mapper.analyze_and_validate(primary_frame.labels)

    print(f"  -> Total Points: {val_summary.coordinate_validation.total_points:,}")
    print(f"  -> Invalid Coordinates (NaN/Inf): {val_summary.coordinate_validation.invalid_point_count} ({val_summary.coordinate_validation.invalid_point_percentage}%)")
    print(f"  -> Radial Range: Min={val_summary.range_validation.min_range}m, Max={val_summary.range_validation.max_range}m, Mean={val_summary.range_validation.mean_range}m, p95={val_summary.range_validation.p95_range}m")
    print(f"  -> Intensity Format: {val_summary.intensity_validation.detected_format} (Range [{val_summary.intensity_validation.min_intensity}, {val_summary.intensity_validation.max_intensity}])")
    print(f"  -> Coordinate Orientation: {val_summary.coordinate_distribution.coordinate_convention_status}")
    print(f"     Forward X > 0: {val_summary.coordinate_distribution.forward_x_percentage}%, Lateral |Y| Mean: {val_summary.coordinate_distribution.lateral_y_symmetry_error}m")

    # 4. Range Filtering & Normalization
    print(f"\n[4/8] Executing Range Filtering (0 <= r <= 100m)...")
    mapped_primary = label_mapper.map_frame(primary_frame)
    filtered_primary, filter_rep = range_filter.filter_frame(mapped_primary)
    print(f"  -> Input Points: {filter_rep.input_points:,}")
    print(f"  -> Filtered Output Points: {filter_rep.output_points:,} (Retention: {filter_rep.retention_percentage}%)")
    print(f"  -> Removed Out-of-Range (>100m): {filter_rep.removed_out_of_range_points:,}")

    # 5. Distance-Aware Foveation
    print(f"\n[5/8] Performing Distance-Aware Foveation (Policy: {args.policy})...")
    fov_res = foveator.voxelize(filtered_primary, policy=AggregationPolicy(args.policy))
    foveated_primary = fov_res.foveated_frame
    print(f"  -> Foveated Output Points: {fov_res.foveated_points:,}")
    print(f"  -> Point Reduction: {fov_res.point_reduction_percentage}% (Compression Ratio: {fov_res.compression_ratio}x)")
    print(f"  -> Processing Latency: {fov_res.processing_time_ms} ms ({fov_res.fps} FPS)")
    for b_stat in fov_res.band_stats:
        print(f"     - {b_stat.band_name} [{b_stat.min_range}-{b_stat.max_range}m @ {b_stat.voxel_size}m]: {b_stat.raw_points:,} -> {b_stat.foveated_points:,} pts ({b_stat.point_reduction_percentage}% red.)")

    # 6. Preservation Metrics
    print(f"\n[6/8] Evaluating Information-Preservation Metrics...")
    elev_rep = elevation_val.evaluate(filtered_primary, foveated_primary)
    obs_rep = obstacle_val.evaluate(filtered_primary, foveated_primary)
    sem_rep = semantic_val.evaluate(filtered_primary, foveated_primary, policy_name=args.policy)

    print(f"  -> 2.5D Elevation Overall RMSE: {elev_rep.overall_rmse} m (MAE: {elev_rep.overall_mae} m, p95: {elev_rep.overall_p95_error} m)")
    print(f"     Near (0-10m): RMSE={elev_rep.near_field_error.rmse}m, Mid (10-40m): RMSE={elev_rep.mid_field_error.rmse}m, Far (40-100m): RMSE={elev_rep.far_field_error.rmse}m")
    print(f"  -> Obstacle Occupancy Recall: {obs_rep.obstacle_grid_recall}% (Loss: {obs_rep.obstacle_loss_percentage}%, IoU: {obs_rep.obstacle_grid_iou})")
    print(f"  -> Far-Field Dynamic Object Survival: {obs_rep.far_field_dynamic_survival_rate}%")
    print(f"  -> Semantic Ground Preservation: {sem_rep.ground_preservation_score:.3f}, Static Obstacle: {sem_rep.static_obstacle_preservation_score:.3f}, Dynamic: {sem_rep.dynamic_object_preservation_score:.3f}")

    # 7. Benchmarking All Candidate Configurations
    print(f"\n[7/8] Benchmarking Candidate Configurations across {len(frames)} frames...")
    with open(args.candidates_config, "r") as f:
        cand_data = yaml.safe_load(f)

    benchmark_results: Dict[str, Any] = {}
    for c_name, c_dict in cand_data.get("configurations", {}).items():
        print(f"  -> Profiling {c_name} ({c_dict.get('description', '')})...")
        b_res = benchmark_runner.benchmark_pipeline_on_frames(
            frames=frames,
            config_name=c_name,
            config_dict=c_dict,
            repeats_per_frame=args.repeats,
            policy=AggregationPolicy(args.policy)
        )
        benchmark_results[c_name] = b_res

    # Print candidate comparison table
    comp_rows = []
    for name, res in benchmark_results.items():
        comp_rows.append([
            name,
            res.config_type,
            f"{res.point_reduction_percent:.1f}%",
            f"{res.compression_ratio:.1f}x",
            f"{res.total_latency_mean_ms:.1f} ms",
            f"{res.total_latency_p95_ms:.1f} ms",
            f"{res.fps_mean:.1f}",
            f"{res.elevation_rmse_m:.3f} m",
            f"{res.obstacle_recall_percent:.1f}%",
            f"{res.dynamic_object_survival_percent:.1f}%"
        ])
    headers = ["Configuration", "Type", "Pt. Reduc.", "Ratio", "Mean Lat.", "p95 Lat.", "FPS", "Elev. RMSE", "Obs. Recall", "Dyn. Survival"]
    print("\n" + tabulate(comp_rows, headers=headers, tablefmt="grid"))

    # 8. Visualizations and Reporting
    print(f"\n[8/8] Exporting Visualizations and Automated Reports...")
    if args.export_vis:
        vis_paths = vis_exporter.export_comparison_suite(
            raw_frame=primary_frame,
            filtered_frame=filtered_primary,
            foveated_frame=foveated_primary,
            bands=foveator.bands,
            prefix=f"seq{args.sequence}_frame{primary_frame.frame_id}"
        )
        for k, p in vis_paths.items():
            print(f"  -> Generated Visualization: {p}")

    reproducibility = {
        "dataset": args.dataset_path,
        "sequence": args.sequence,
        "frame_ids": [f.frame_id for f in frames],
        "primary_configuration": args.config,
        "aggregation_policy": args.policy,
        "maximum_range_m": max_range,
        "bands": [{"name": b.name, "min_range": b.min_range, "max_range": b.max_range, "voxel_size": b.voxel_size} for b in foveator.bands],
        "software_version": "1.0.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": 42
    }

    if args.export_reports:
        json_report = report_gen.generate_json_report(
            dataset_name=f"SemanticKITTI_Seq{args.sequence}",
            frames_tested=len(frames),
            invalid_frames=invalid_count,
            benchmark_results=benchmark_results,
            primary_config_name="config_A",
            elevation_report=elev_rep,
            obstacle_report=obs_rep,
            semantic_report=sem_rep,
            label_report=label_report,
            validation_summary=val_summary,
            reproducibility_meta=reproducibility,
            output_filename="phase1_validation_report.json"
        )
        md_text = report_gen.generate_markdown_report(
            report_data=json_report,
            output_filename="phase1_validation_report.md"
        )
        print(f"  -> Exported JSON Report: reports/phase1_validation_report.json")
        print(f"  -> Exported Markdown Report: reports/phase1_validation_report.md")

    print("\n" + "=" * 80)
    print("  PHASE 1 VALIDATION & BENCHMARKING COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
