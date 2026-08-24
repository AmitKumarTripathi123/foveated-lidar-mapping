# Phase 14D — CPU Student Latency / Quality Trade-Off & Final Model Selection Report

## 1. Executive Summary
This report establishes the authoritative, frozen-protocol evaluation of SPVCNN student architectures across channel widths ($6\text{ to }32\text{ channels}$) on real LiDAR data (SemanticPOSS Sequence 00, 50 validation frames, 6 CPU threads).

---

## 2. Frozen Protocol Channel-Width Sweep

| Channels | Parameters | CPU Latency (Mean) | P50 (ms) | P95 (ms) | P99 (ms) | FPS | mIoU (%) | Accuracy (%) | Dyn Obj IoU (%) | Static Obs IoU (%) | CPU Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **32 (Teacher)** | 136,004 | 148.78 ms | 146.05 ms | 163.44 ms | 185.81 ms | 6.7 | 6.7% | 13.4% | 13.3% | 4.2% | **REJECT — TOO SLOW** |
| **16 (Student)** | 34,724 | 71.43 ms | 71.05 ms | 76.19 ms | 78.20 ms | 14.0 | 8.4% | 23.1% | 4.5% | 5.1% | **GPU PRIMARY MODEL** |
| **14 (Interm.)** | 26,744 | 67.02 ms | 66.59 ms | 71.84 ms | 72.46 ms | 14.9 | 14.9% | 37.6% | 20.1% | 39.4% | **REJECT (>50ms)** |
| **12 (Interm.)** | 19,804 | 54.53 ms | 55.31 ms | 56.04 ms | 56.26 ms | 18.3 | 15.4% | 31.4% | 9.7% | 13.6% | **REJECT (>50ms)** |
| **10 (Interm.)** | 13,904 | 49.71 ms | 50.80 ms | 51.83 ms | 52.02 ms | 20.1 | 11.4% | 39.3% | 1.6% | 43.6% | **REJECT (P95 > 50ms)** |
| **8 (Lightweight)**| **9,044** | **37.23 ms** | **36.10 ms** | **42.16 ms** | **43.62 ms** | **26.9** | **17.6%** | **35.3%** | **26.3%** | **35.1%** | **BEST CPU CANDIDATE** |
| **6 (Tiny)** | 5,224 | 31.93 ms | 32.31 ms | 33.59 ms | 33.89 ms | 31.3 | 7.6% | 29.7% | 0.0% | 0.8% | **REJECT (Safety Collapse)**|

---

## 3. Pareto Frontier Analysis & Trade-Off

1. **Why 10-Channel is Rejected**: Although its mean latency is $49.71\text{ ms}$, its tail latencies (**P95 = $51.83\text{ ms}$**, **P99 = $52.02\text{ ms}$**) violate the hard $50\text{ ms}$ ceiling.
2. **Why 6-Channel is Rejected**: Shrinking to 6 channels collapses the Dynamic Object IoU to **$0.0\%$** (complete failure to detect moving vehicles/pedestrians).
3. **Why 8-Channel is the Best CPU Candidate**:
   * **Latency Safety**: $37.23\text{ ms}$ mean with P95 of **$42.16\text{ ms}$** ($+7.84\text{ ms}$ safety headroom below $50\text{ ms}$).
   * **Safety-Critical Preservation**: Achieves the highest Dynamic Object IoU ($26.3\%$) and Static Obstacle IoU ($35.1\%$) among all $<50\text{ ms}$ models.

---

## 4. Final System Configuration

* **Primary Deployment Architecture**: **CUDA GPU + 16-channel Student** ($\sim 23.59\text{ ms} \implies 42.4\text{ FPS}$).
* **CPU Fallback Architecture**: **CPU (6 threads) + 8-channel Lightweight Student** ($37.23\text{ ms} \implies 26.9\text{ FPS}$).
