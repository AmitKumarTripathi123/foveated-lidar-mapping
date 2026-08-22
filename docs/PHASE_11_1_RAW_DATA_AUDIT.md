# Phase 11.1 Raw Data Inventory & Forensic Audit Report

**Project**: Foveated 2.5D LiDAR Mapping for Autonomous Navigation  
**Lead Engineer**: Atul (ML/AI Perception Lead)  
**Date**: August 22, 2026  

---

## 1. Expected vs Actual Inventory Comparison

* **Amit Reported Inventory**:
  * Sequence 00: 488 frames
  * Sequence 01: 500 frames
  * Sequence 02: 500 frames
  * Sequence 03: 500 frames
  * Sequence 04: 500 frames
  * Sequence 05: 500 frames
  * **Expected Total**: 2,988 .bin and 2,988 .label pairs
* **Physical Local Inventory Discovered**:
  * Sequence 00: 1 frame (`000000.bin` / `000000.label`)
  * Sequences 01–05: Not yet extracted in local workspace directory
  * **Actual Total**: 1 physical frame pair (66,658 points)
* **Discrepancy Status**: **DISCREPANCY DETECTED & DOCUMENTED** (Local workspace contains 1 representative scan; full 2,988-frame multi-sequence archive is pending physical extraction into `dataset/sequences/`).

---

## 2. Physical File Integrity Verification

* **Point Cloud File**: `dataset/sequences/00/velodyne/000000.bin` ($1,066,528$ bytes, $66,658$ float32 $(x, y, z, i)$ points).
* **Label File**: `dataset/sequences/00/labels/000000.label` ($266,632$ bytes, $66,658$ uint32 raw labels).
* **Point-Label Alignment**: $100\%$ aligned ($N_{\text{points}} == N_{\text{labels}} == 66,658$).
* **Data Quality**: Zero NaNs, zero Infs, $100\%$ finite values.
