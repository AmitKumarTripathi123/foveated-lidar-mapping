# Dataset Forensic Identity Audit (Phase 11)

**Project**: Foveated 2.5D LiDAR Mapping for Autonomous Navigation  
**Perception Lead**: Atul  
**Date**: August 22, 2026  

---

## 1. Forensic Dataset Identification

* **Dataset Identity**: **SemanticKITTI** (Raw scan `dataset/sequences/00/velodyne/000000.bin` and `000000.label`)
* **Evidence**:
  * Raw Label IDs present in `000000.label`: `[0, 10, 40, 48, 50, 51, 70, 71, 80]`
  * Exact Semantics: `40=road, 48=sidewalk, 50=building, 51=fence, 70=vegetation, 71=trunk, 80=pole, 10=car, 0=unlabeled`.
  * These match the official SemanticKITTI specification.
* **Point Cloud Format**: Little-endian `float32`, shape $(N, 4)$ where channels are $[x, y, z, \text{intensity}]$.
* **Label Format**: Little-endian `uint32` where lower 16 bits encode semantic class ID (`raw_label & 0xFFFF`).
* **Coordinate Convention**: Sensor frame (Velodyne coordinate system: $+X$ forward, $+Y$ left, $+Z$ up).
* **Number of Beams**: 64-beam LiDAR (Velodyne HDL-64E).
* **Sequence Structure**: `sequences/<seq_id>/velodyne/*.bin` and `sequences/<seq_id>/labels/*.label`.
* **Confidence**: **100% High Confidence**.
* **Unresolved Ambiguity**: None.
