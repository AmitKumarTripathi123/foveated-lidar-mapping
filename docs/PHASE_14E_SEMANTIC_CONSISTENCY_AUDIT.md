# Phase 14E — Semantic Quality Consistency & CPU/GPU Model Validation Audit Report

## 1. Executive Summary
This document provides the definitive forensic explanation for the semantic metric discrepancy between untrained channel-width latency profiling and the authoritative trained checkpoint evaluation.

---

## 2. Forensic Discrepancy Root-Cause Analysis

| Benchmark Scope | Model Weights Status | Evaluated Checkpoint | mIoU (%) | Accuracy (%) | Dynamic Obj IoU (%) | Static Obs IoU (%) | Explanation |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Phase 14D Channel Sweep** | **Untrained (Random Initializations)** | `SPVCNN(base_channels=...)` (New instance) | 6.7% - 17.6% | 13.4% - 35.3% | 0.0% - 26.3% | 0.8% - 35.1% | Profiled purely for hardware latency scaling across channel widths; weights were randomly initialized. |
| **Phase 14E Authoritative Audit** | **Trained & Distilled** | `checkpoints/spvcnn_student_16ch.pt` (16-ch Student) | **70.01%** | **89.62%** | **78.86%** | **87.36%** | Fully trained with Knowledge Distillation on SemanticPOSS sequence 00. |

---

## 3. Authoritative Performance & Quality Baseline

### Primary GPU Production Target (16-Channel Distilled Student)
* **Checkpoint**: `checkpoints/spvcnn_student_16ch.pt` (SHA256: `3130b01a3badf31a...`)
* **Parameters**: **34,724 parameters** ($74.5\%$ compression)
* **GPU Pipeline Latency**: **$\approx 23.59\text{ ms}$** ($42.4\text{ FPS}$)
* **mIoU**: **$70.01\%$**
* **Overall Accuracy**: **$89.62\%$**
* **Dynamic Object IoU**: **$78.86\%$**
* **Static Obstacle IoU**: **$87.36\%$**
* **Non-Drivable Terrain IoU**: **$78.06\%$**
* **Drivable Terrain IoU**: **$35.76\%$**

---

## 4. Final Verdict
* **Discrepancy Status**: **100% EXPLAINED AND RESOLVED**.
* **Phase 14D Decision**: **PHASE 14D FROZEN**.
* **Primary Deployment Architecture**: **16-Channel Distilled SPVCNN on CUDA GPU** ($23.59\text{ ms}$, $70.01\%\text{ mIoU}$, $89.62\%\text{ Accuracy}$).
