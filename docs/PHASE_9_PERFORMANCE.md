# Phase 9 Perception & Mapping Latency Performance Report

**Project**: Foveated 2.5D LiDAR Mapping for Autonomous Navigation  
**Lead Engineer**: Atul (ML/AI Perception Lead)  
**Date**: August 22, 2026  

---

## 1. Measured System Latency Breakdown (CPU)

Measured on representative LiDAR frame ($66,658$ raw points downsampled to $50,571$ foveated points and normalized to $N=1024$ points) using [`scripts/benchmark_latency.py`](file:///C:/Users/atuls/OneDrive/Desktop/Lidar/scripts/benchmark_latency.py):

| Pipeline Subsystem | Latency (ms) | Percentage (%) | Classification |
| :--- | :---: | :---: | :---: |
| **1. Raw LiDAR Loading (`.bin` / `.label`)** | $5.35\text{ ms}$ | $2.60\%$ | Measured |
| **2. Amit 3-Zone Foveated Downsampler** | $24.45\text{ ms}$ | $11.88\%$ | Measured |
| **3. Point Normalization ($N=1024$)** | $2.21\text{ ms}$ | $1.07\%$ | Measured |
| **4. PointNet++ Segmentation Model** | $156.96\text{ ms}$ | $76.26\%$ | Measured |
| **5. 2.5D Mapping GridMap25D Projection** | $16.82\text{ ms}$ | $8.17\%$ | Measured |
| **Total End-to-End Latency** | **$205.81\text{ ms}$** | **$100.0\%$** | **Measured ($4.86\text{ FPS}$)** |

---

## 2. Hardware Classification & Target Hardware Estimates

* **Local CPU**: Intel / AMD Multi-Core (Measured: $205.81\text{ms} / \text{frame}$)
* **CUDA GPU (NVIDIA RTX / Orin)**: **UNAVAILABLE IN CURRENT ENVIRONMENT**
  * *Estimated TensorRT / PyTorch GPU Forward Pass*: $\sim 10\text{--}15\text{ms}$
  * *Estimated End-to-End GPU Throughput*: $\sim 30\text{--}40\text{ FPS}$
