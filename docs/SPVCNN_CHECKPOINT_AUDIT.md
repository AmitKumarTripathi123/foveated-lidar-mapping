# Pretrained SPVCNN Checkpoint & Architecture Audit

**Project**: Foveated 2.5D LiDAR Mapping for Autonomous Navigation  
**Perception Lead**: Atul  
**Date**: August 22, 2026  

---

## 1. Model & Source Provenance

* **Model Family**: Sparse Point-Voxel Convolutional Neural Network (SPVCNN)
* **Citation**: *Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution*, Tang et al., ECCV 2020.
* **Canonical Repository**: MIT HAN Lab / TorchSparse (`https://github.com/mit-han-lab/spvnas`)
* **Reference Architecture**: `SPVCNN` with multi-scale Point-Voxel Sparse Convolution (SPVConv) blocks.
* **Total Parameters**: ~2,180,000 parameters.

---

## 2. Input / Output Specifications

| Attribute | Specification | Notes |
| :--- | :--- | :--- |
| **Input Channels** | $4$ | `[x, y, z, intensity]` (float32) |
| **Coordinate Frame** | Vehicle LiDAR frame | $+X$ forward, $+Y$ left, $+Z$ up (meters) |
| **Recommended Voxel Size** | $0.05\text{m}$ ($5\text{ cm}$) | Quantization grid for 3D sparse convolutions |
| **Input Structure** | Sparse point-voxel representation | Points $(N, 4)$ with index mapping table |
| **Output Structure** | Per-point class logits $(N, C)$ | Direct projection to each input point |
| **Inference Hardware** | CPU & CUDA | Full PyTorch native execution |

---

## 3. Label Ontology & SIH Compatibility Matrix

### SemanticKITTI 19-Class $\to$ SIH 4-Class Mapping:

| Native ID | Native Semantic Class | SIH Class ID | SIH Super-Class |
| :---: | :--- | :---: | :--- |
| `0` | `car` | `3` | `dynamic_object` |
| `1` | `bicycle` | `3` | `dynamic_object` |
| `2` | `motorcycle` | `3` | `dynamic_object` |
| `3` | `truck` | `3` | `dynamic_object` |
| `4` | `other-vehicle` | `3` | `dynamic_object` |
| `5` | `person` | `3` | `dynamic_object` |
| `6` | `bicyclist` | `3` | `dynamic_object` |
| `7` | `motorcyclist` | `3` | `dynamic_object` |
| `8` | `road` | `0` | `drivable_terrain` |
| `9` | `parking` | `0` | `drivable_terrain` |
| `10` | `sidewalk` | `1` | `non_drivable_terrain` |
| `11` | `other-ground` | `1` | `non_drivable_terrain` |
| `12` | `building` | `2` | `static_obstacle` |
| `13` | `fence` | `2` | `static_obstacle` |
| `14` | `vegetation` | `2` | `static_obstacle` |
| `15` | `trunk` | `2` | `static_obstacle` |
| `16` | `terrain` | `1` | `non_drivable_terrain` |
| `17` | `pole` | `2` | `static_obstacle` |
| `18` | `traffic-sign` | `2` | `static_obstacle` |
| `255` | `unlabeled / outlier` | `255` | `ignore` |

---

## 4. Zero-Shot vs Fine-Tuning Decision

* **Status**: **SPVCNN PRETRAINED WEIGHTS LOADED — SIH FINE-TUNING / ADAPTER VERIFIED**
* **Rationale**: The official SemanticKITTI pretrained SPVCNN weights provide rich 3D geometric feature representations. The `SPVCNNLabelAdapter` deterministically translates the 19 native classes into the frozen SIH 4-class ontology for zero-shot inference, and provides a fine-tuning head interface for full multi-sequence training when the complete 2,988-frame SemanticPOSS dataset is extracted.
