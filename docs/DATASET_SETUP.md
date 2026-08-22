# LiDAR Dataset Acquisition & Configuration Guide (Phase 8)

This guide provides instructions for connecting and configuring external multi-sequence SemanticPOSS or SemanticKITTI point-cloud datasets for the **Foveated 2.5D LiDAR Mapping** perception system.

---

## 1. Directory Structure Requirements

The dataset discovery engine (`ml/data/frame_discovery.py`) expects the standard SemanticKITTI / SemanticPOSS folder hierarchy:

```text
DATASET_ROOT/
├── sequences/
│   ├── 00/
│   │   ├── velodyne/
│   │   │   ├── 000000.bin
│   │   │   ├── 000001.bin
│   │   │   └── ...
│   │   └── labels/
│   │       ├── 000000.label
│   │       ├── 000001.label
│   │       └── ...
│   ├── 01/
│   │   ├── velodyne/*.bin
│   │   └── labels/*.label
│   ├── 02/
│   └── ...
```

---

## 2. File Format Specifications

* **Point Clouds (`.bin`)**: Little-endian `float32` binary array of shape $(N, 4)$ representing $[x, y, z, \text{intensity}]$ in the sensor coordinate frame. File size must be a multiple of 16 bytes ($4 \times 4$ bytes).
* **Labels (`.label`)**: Little-endian `uint32` binary array of shape $(N,)$ where the lower 16 bits encode the raw semantic class ID (`raw_label & 0xFFFF`).
* **Alignment Invariant**: $N_{\text{points}} == N_{\text{labels}}$ for every physical frame scan.

---

## 3. Environment Variable & CLI Configuration

You can specify the location of the dataset without modifying source code:

### Option A: Environment Variable
```bash
# Windows PowerShell:
$env:DATASET_ROOT = "D:\Datasets\SemanticKITTI\dataset"

# Linux / macOS:
export DATASET_ROOT="/data/SemanticKITTI/dataset"
```

### Option B: CLI Parameter
```bash
python scripts/generate_manifest.py --dataset-root "D:\Datasets\SemanticKITTI\dataset"
```

---

## 4. Discovery, Preprocessing & Training Workflow

```bash
# 1. Discover all sequences and frames
python scripts/generate_manifest.py

# 2. Preprocess and generate foveated downsampled cache
python scripts/preprocess_foveated.py

# 3. Train PointNet++ Baseline
python scripts/train_phase7.py --config configs/phase8_training.yaml --experiment phase8_baseline_ce

# 4. Measure End-to-End Latency
python scripts/benchmark_latency.py --iterations 10
```
