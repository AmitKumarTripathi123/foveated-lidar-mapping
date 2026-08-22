# Dataset Integrity Audit Report

**Total Discovered Scans**: 1
**Total Points**: 66,658
**Sequences**: 00
**Corrupted / Malformed Scans**: 0

## 1. Split Allocation
- **Train Frames**: 1
- **Val Frames**: 1
- **Test Frames**: 0

## 2. Global Raw Class Breakdown
| Raw ID | Point Count | Percentage |
| :--- | :--- | :--- |
| `0` | 1,158 | 1.74% |
| `10` | 6,000 | 9.00% |
| `40` | 23,000 | 34.50% |
| `48` | 8,000 | 12.00% |
| `50` | 10,000 | 15.00% |
| `51` | 2,000 | 3.00% |
| `70` | 13,000 | 19.50% |
| `71` | 2,000 | 3.00% |
| `80` | 1,500 | 2.25% |

## 3. Data Integrity & Alignment Checklist
- `[PASS]` All scans have shape divisible by 4 (16 bytes per point)
- `[PASS]` Zero NaN or Inf floating-point values detected
- `[PASS]` Exact point-label count alignment verified on all available frames
