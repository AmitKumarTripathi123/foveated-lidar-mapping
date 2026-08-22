# Dataset Integrity Audit Report

**Total Discovered Scans**: 2988
**Total Points**: 202,504,402
**Sequences**: 00, 01, 02, 03, 04, 05
**Corrupted / Malformed Scans**: 0

## 1. Split Allocation
- **Train Frames**: 2488
- **Val Frames**: 500
- **Test Frames**: 0

## 2. Global Raw Class Breakdown
| Raw ID | Point Count | Percentage |
| :--- | :--- | :--- |
| `0` | 4,223,840 | 2.09% |
| `1` | 5 | 0.00% |
| `4` | 3,118,730 | 1.54% |
| `5` | 626,702 | 0.31% |
| `6` | 905,865 | 0.45% |
| `7` | 16,726,113 | 8.26% |
| `8` | 2,565,775 | 1.27% |
| `9` | 72,701,026 | 35.90% |
| `10` | 655,751 | 0.32% |
| `11` | 262,611 | 0.13% |
| `12` | 331,384 | 0.16% |
| `13` | 984,778 | 0.49% |
| `14` | 175,463 | 0.09% |
| `15` | 43,267,526 | 21.37% |
| `16` | 194,468 | 0.10% |
| `17` | 3,082,198 | 1.52% |
| `18` | 149,643 | 0.07% |
| `19` | 7,491 | 0.00% |
| `20` | 110,847 | 0.05% |
| `21` | 10,970,574 | 5.42% |
| `22` | 41,384,112 | 20.44% |
| `40` | 23,000 | 0.01% |
| `48` | 8,000 | 0.00% |
| `50` | 10,000 | 0.00% |
| `51` | 2,000 | 0.00% |
| `70` | 13,000 | 0.01% |
| `71` | 2,000 | 0.00% |
| `80` | 1,500 | 0.00% |

## 3. Data Integrity & Alignment Checklist
- `[PASS]` All scans have shape divisible by 4 (16 bytes per point)
- `[PASS]` Zero NaN or Inf floating-point values detected
- `[PASS]` Exact point-label count alignment verified on all available frames
