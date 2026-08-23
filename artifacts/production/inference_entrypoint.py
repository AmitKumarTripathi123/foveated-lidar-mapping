#!/usr/bin/env python3
import sys, argparse
from pathlib import Path
import numpy as np
from ml.pipeline.production_pipeline import ProductionPipeline
from ml.data.dataset import load_point_cloud

def main():
    parser = argparse.ArgumentParser(description="Production Real-Time 3D LiDAR Perception & 2.5D Mapping Entrypoint")
    parser.add_argument("--config", type=str, default="configs/production.yaml", help="Path to production YAML config")
    parser.add_argument("--input-scan", type=str, required=True, help="Path to raw .bin LiDAR scan")
    args = parser.parse_args()

    pipeline = ProductionPipeline(args.config)
    pts = load_point_cloud(args.input_scan)
    res = pipeline.process_frame(pts)
    if res.success:
        print(f"Processed frame in {res.latency_ms:.2f} ms ({1000.0/res.latency_ms:.1f} FPS)")
        print(f"Points: {res.num_input_points} -> Foveated: {res.num_foveated_points}")
        print(f"2.5D GridMap Shape: {res.grid_map.grid_shape}")
    else:
        print(f"Processing failed: {res.error_message}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
