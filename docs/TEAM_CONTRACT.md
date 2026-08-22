# Team Collaboration & Interface Contract

**Project**: Foveated 2.5D LiDAR Mapping for Autonomous Navigation  
**Team Members**:
- **Amit**: Team Lead & Lead for Foveated Preprocessing, Point Cloud Filtering, 2.5D Grid Mapping Architecture
- **Atul**: ML/AI Perception Lead (Phases 1–5 Deep Learning Architecture, PointNet++, Label Remapping, Training System)

---

## 1. Domain Ownership Boundaries

### AMIT''S DOMAIN: Foveated Data Processing & Mapping
- **Foveated Voxelization**: Owns the 3-Zone variable resolution voxelizer ($0\text{--}10\text{m} \to 0.05\text{m}$, $10\text{--}40\text{m} \to 0.15\text{m}$, $40\text{--}100\text{m} \to 0.50\text{m}$).
- **Range & Spatial Filtering**: Owns maximum sensor radius ($100\text{m}$ Euclidean distance) and spatial cropping.
- **Offline Voxel Cache**: Owns the caching structure under `processed/` for fast IO during training and inference.
- **Future Mapping Module**: Owns consuming the ML output contract to construct the variable-resolution 2.5D elevation and semantic grid.

### ATUL''S DOMAIN: Machine Learning Perception & Training
- **Semantic Label Remapping**: Owns the SIH 4-Class ontology ($0=\text{drivable}$, $1=\text{non-drivable}$, $2=\text{static}$, $3=\text{dynamic}$, $255=\text{ignore}$).
- **PyTorch Dataset Adapter**: Owns `FoveatedLidarDataset`, `lidar_collate_fn`, and point-count normalization ($N = 1024$ / $16384$).
- **PointNet++ Architecture**: Owns `PointNet2SemSeg` and all sub-modules (Set Abstraction, Feature Propagation).
- **Training Engine**: Owns loss formulations (Plain & Weighted CE), optimizers, schedulers, and `PointNet2Trainer`.
- **Evaluation & Diagnostics**: Owns the $4 \times 4$ confusion matrix, per-class IoU, mIoU, precision, recall, and model collapse detection.
- **Prediction Contract**: Owns `PointNet2Predictor` implementing the frozen ML $\to$ Mapping contract.

---

## 2. Shared Interface Agreements

1. **Point Cloud Schema**:
   All intermediate stages represent LiDAR points as $[N, 4]$ NumPy/PyTorch arrays of type `float32` containing $[x, y, z, \text{intensity}]$.
2. **Label Schema**:
   All labels represent ground-truth annotations as $[N]$ arrays of type `uint8`/`int64` with values $\in \{0, 1, 2, 3, 255\}$.
3. **1-to-1 Point-Label Alignment**:
   Every point transformation (filtering, voxelization, sampling) must strictly preserve corresponding semantic labels ($N_{\text{points}} == N_{\text{labels}}$).
4. **Frozen ML $\to$ Mapping Contract**:
   `PointNet2Predictor.predict(pts)` returns:
   - `xyz`: `(N, 3)` `float32` in exact original point order ($\text{in\_xyz}[i] == \text{out\_xyz}[i]$)
   - `predicted_class`: `(N,)` `int64` strictly in $\{0, 1, 2, 3\}$
   - `confidence`: `(N,)` `float32` strictly in $[0.0, 1.0]$.
