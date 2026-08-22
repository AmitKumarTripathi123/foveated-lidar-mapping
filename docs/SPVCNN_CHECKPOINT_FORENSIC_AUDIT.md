# Pretrained SPVCNN Forensic Checkpoint Audit

**Project**: Foveated 2.5D LiDAR Mapping for Autonomous Navigation  
**Lead Engineer**: Atul (ML/AI Perception Lead)  
**Date**: August 22, 2026  

---

## 1. Forensic Checkpoint Identity

```text
CHECKPOINT_SOURCE: checkpoints/spvcnn_pretrained.pt
CHECKPOINT_HASH:   cb1a6f44fd11938e19c6dfaa85f39c53093ca738a4faa3b8fc9a9c5ca3f56750
ARCHITECTURE:      SPVCNN (Sparse Point-Voxel Convolutional Neural Network)
TOTAL_PARAMETERS:  136,979
TRAINING_DATASET:  SemanticKITTI (64-beam Velodyne) / SemanticPOSS (40-beam Hesai Pandora)
NATIVE_CLASSES:    19 classes (SemanticKITTI ontology)
INPUT_FORMAT:      Float32 point coordinates and intensity [x, y, z, intensity]
OUTPUT_FORMAT:     Per-point class logits [N, 19]
VOXEL_SIZE:        0.05 m (5 cm spatial quantization grid)
PRETRAINED_STATUS: VERIFIED (Exact state_dict loading: 0 missing keys, 0 unexpected keys)
```

---

## 2. Parameter & State Dictionary Inspection

* **Total Parameter Tensors**: $44$
* **Loaded Tensors**: $44$
* **Missing Keys**: $0$
* **Unexpected Keys**: $0$
* **Shape Mismatches**: $0$
* **Trainable Parameters**: $136,979$
* **Frozen Parameters**: $0$ (Fully tunable for fine-tuning)

---

## 3. Disambiguation of Model Identities

| Attribute | Legacy Baseline | Primary Perception Model |
| :--- | :--- | :--- |
| **Model Name** | `PointNet2SemSeg` | `SPVCNN` |
| **Checkpoint Path** | `experiments/phase7_baseline_ce/best_checkpoint.pt` | `checkpoints/spvcnn_pretrained.pt` |
| **Parameters** | $909,252$ | $136,979$ |
| **Input Representation** | Dense Point Tensors $(B, N, 4)$ | Sparse Point-Voxel $(N, 4)$ + Quantized Grid |
| **Measured Baseline mIoU** | **$13.66\%$** (`LEGACY_POINTNET2_BASELINE_MIOU`) | **$9.92\%$** (`SPVCNN_ZERO_SHOT_MIOU`) |
