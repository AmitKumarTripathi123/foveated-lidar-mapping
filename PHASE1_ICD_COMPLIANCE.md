# Phase 1 — ICD Compliance & Interface Verification Report

**Interface Contract**: Smart India Hackathon — Foveated 2.5D LiDAR Mapping  
**Compliance Status**: **PASS (with documented SemanticPOSS Human Decision checkpoint)**  

---

## 1. Interface Checklist

| ICD Requirement | Contract Specification | Implementation Status | Verified Evidence |
| :--- | :--- | :--- | :--- |
| **XYZ Datatype** | `float32[N, 3]` | **PASS** | Enforced in `PointCloudFrame.points[:, :3]` as `np.float32` |
| **Intensity Datatype** | `float32[N]` | **PASS** | Enforced in `PointCloudFrame.points[:, 3]` as `np.float32` |
| **Intensity Range** | Internal frozen range `float32 [0.0, 1.0]` | **PASS** | Raw format detected; normalized via `PointCloudValidator.normalize_intensity()` without modifying raw data |
| **Class ID Datatype** | `uint8 / uint32[N]` | **PASS** | Enforced in `PointCloudFrame.labels` as `np.uint32` matching `uint8_t` in C struct |
| **Super-Class Mappings** | `0=drivable, 1=non_drivable, 2=static_obstacle, 3=dynamic_object, 255=ignore` | **PASS** | Enforced in `src/types.py:SuperClass` and `src/label_mapper.py` |
| **Class-1 Mapping Resolution** | SemanticKITTI: 48, 49, 72 -> 1<br>SemanticPOSS: Ambiguous | **PASS / HUMAN CONFIRMATION** | SemanticKITTI fully resolved; SemanticPOSS flags explicit `WARNING: non_drivable_terrain mapping is undefined/incomplete. Human confirmation required.` |
| **Confidence Datatype** | `float32[N]` with values $\\in [0.0, 1.0]$ | **PASS** | `PointCloudFrame.confidences` initialized to 1.0 for ground truth, bounded $[0, 1]$ |
| **Coordinate Convention** | `+X = forward, +Y = left, +Z = upward` | **PASS** | Evaluated in `PointCloudValidator.validate_coordinate_distribution()`; visual diagnostics confirmed |
| **Radial Range Formula** | $r = \\sqrt{x^2 + y^2}$ (2D horizontal) | **PASS** | Strictly uses horizontal plane $r = \\sqrt{x^2+y^2}$, NOT spherical 3D $\\sqrt{x^2+y^2+z^2}$ |
| **Maximum Range** | $0 \\le r \\le 100.0\\text{m}$ | **PASS** | Configurable in YAML (default 100.0m); points $>100\\text{m}$ filtered non-destructively |
| **Foveation Bands** | Band 1: 0–10m @ 0.05m<br>Band 2: 10–40m @ 0.15m<br>Band 3: 40–100m @ 0.50m | **PASS** | Exact band inequalities: $[0, 10), [10, 40), [40, 100]$; tested on exact boundary epsilons |
| **C++ Struct Interoperability** | `struct ClassifiedPoint` | **PASS** | `CClassifiedPoint` and `CClassifiedPointPacked` ctypes structs provided in `src/types.py` |
| **Timestamp / Frame ID** | `frame_id: str, timestamp: float` | **PASS** | Captured and propagated throughout pipeline metadata |
