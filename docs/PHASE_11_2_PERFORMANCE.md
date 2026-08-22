# Phase 11.2 End-to-End Latency & Performance Benchmark

**Project**: Foveated 2.5D LiDAR Mapping for Autonomous Navigation  
**Lead Engineer**: Atul (ML/AI Perception Lead)  
**Date**: August 22, 2026  

---

## 1. Stage-by-Stage Latency Benchmark (CPU)

Measured on representative LiDAR frame ($66,658$ raw points downsampled to $50,571$ points and normalized to $N=1024$ points) using [`scripts/benchmark_latency.py`](file:///C:/Users/atuls/OneDrive/Desktop/Lidar/scripts/benchmark_latency.py):

* **Total Measured Latency**: **$205.81\text{ ms} / \text{frame}$**
* **System Throughput**: **$4.86\text{ FPS}$**
* **Stage Breakdown**:
  * Raw Loading: $5.35\text{ ms}$
  * Amit Foveated Voxelizer: $24.45\text{ ms}$
  * Point Normalization: $2.21\text{ ms}$
  * PointNet++ Inference: $156.96\text{ ms}$
  * 2.5D Mapping Grid: $16.82\text{ ms}$
* **GPU Latency**: **UNAVAILABLE** (CUDA GPU not present in execution environment).
