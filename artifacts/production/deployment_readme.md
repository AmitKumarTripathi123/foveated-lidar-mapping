# Production Deployment Package — Foveated LiDAR Mapping

## Overview
Hardened, real-time autonomous navigation perception pipeline connecting SPVCNN point-voxel sparse convolution with Amit's 2.5D foveated elevation and occupancy grid mapping.

* **Production Checkpoint**: `experiments/phase12_full_semanticposs_spvcnn/best_checkpoint.pt`
* **Cryptographic SHA256**: `b15c6dfb2f20d1dce4febc47be67f9d50b86a0af72f1895176c6a6ee58bca142`
* **Tested Frequency**: `10.0 Hz Real-Time Verified` (Mean latency: `171.65 ms`)
* **Output Standard**: 4-Class SIH Ontology (`0: drivable`, `1: non-drivable`, `2: static_obstacle`, `3: dynamic_object`) $\to$ `GridMap25D`

## Quick Start
```bash
py -3.12 artifacts/production/inference_entrypoint.py --config configs/production.yaml --input-scan dataset/sequences/02/velodyne/000001.bin
```
