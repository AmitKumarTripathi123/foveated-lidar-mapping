# Forensic Storage Audit: SemanticPOSS Dataset Discovery

## Executive Summary
A comprehensive forensic data audit was executed across the local filesystem and mounted storage to locate the physical **SemanticPOSS** dataset (6 sequences, 2,988 LiDAR scan pairs). 

The audit confirmed that:
1. The **raw dataset archive** (`SemanticPOSS_dataset.zip`, **2,299.05 MB / 2.3 GB**) is present in `/Users/amitkumartripathi/Downloads/SemanticPOSS_dataset.zip`.
2. The **extracted raw dataset** containing all **6 sequences (`00` through `05`)** and **2,988 matched raw scan pairs** is physically present and fully extracted at `dataset/` (`/Users/amitkumartripathi/Desktop/3d lidar foveated mapping/dataset`).

---

## 1. Search Locations Audited

The following directories and mounted storage paths were scanned for `.bin`, `.label`, `.zip`, `.tar`, `.gz`, and sequence folders:

| Search Path | Status | Archives Found | Extracted Sequences Found |
| :--- | :---: | :---: | :---: |
| `/Users/amitkumartripathi/Desktop` | Audited | None | `dataset/sequences/00` to `05` |
| `/Users/amitkumartripathi/Downloads` | Audited | `SemanticPOSS_dataset.zip` (2,299.05 MB) | None |
| `/Users/amitkumartripathi/Documents` | Audited | None | None |
| `/Users/amitkumartripathi` | Audited | None | None |
| `/Volumes` | Audited | None | None |

---

## 2. Archives Found

| Archive File Path | File Size | Zip Contents | Status |
| :--- | :---: | :--- | :---: |
| `/Users/amitkumartripathi/Downloads/SemanticPOSS_dataset.zip` | **2,299.05 MB** | 2,988 `.bin` files, 2,988 `.label` files across sequences `00`–`05` | **Complete Raw Archive** |
| `/Users/amitkumartripathi/Downloads/caludelidar.zip` | 0.00 MB | Empty scratch archive | Ignored |

---

## 3. Extracted Dataset Locations & Sequence Frame Audit

| Sequence ID | `.bin` Scan Files | `.label` Label Files | Stem Matching Status | Sequence Split Role |
| :---: | :---: | :---: | :---: | :---: |
| **`00`** | 488 | 488 | **100% Matched** (`000000.bin` $\leftrightarrow$ `000000.label`) | **Training** |
| **`01`** | 500 | 500 | **100% Matched** (`000001.bin` $\leftrightarrow$ `000001.label`) | **Training** |
| **`02`** | 500 | 500 | **100% Matched** (`000001.bin` $\leftrightarrow$ `000001.label`) | **Validation** |
| **`03`** | 500 | 500 | **100% Matched** (`000001.bin` $\leftrightarrow$ `000001.label`) | **Training** |
| **`04`** | 500 | 500 | **100% Matched** (`000001.bin` $\leftrightarrow$ `000001.label`) | **Training** |
| **`05`** | 500 | 500 | **100% Matched** (`000001.bin` $\leftrightarrow$ `000001.label`) | **Training** |
| **TOTAL** | **2,988** | **2,988** | **100% Matched (2,988 Pairs)** | **Full SemanticPOSS** |

---

## 4. Data Format & Integrity Verification

- **Raw Point Cloud Format**: Binary `float32` arrays representing `(x, y, z, intensity)` per point.
- **Raw Label Format**: Binary `uint32` arrays where the lower 16 bits (`& 0xFFFF`) store raw semantic class IDs.
- **Derivative Check**: Confirmed raw sensor data (uncached, uncompressed `.bin` and `.label` files).
- **All 6 Sequences Present**: **YES** (Sequences `00`, `01`, `02`, `03`, `04`, `05` are all present).

---

## 5. Recommendation for Canonical `DATASET_ROOT`

- **Recommended Canonical `DATASET_ROOT`**: `dataset/` (Absolute path: `/Users/amitkumartripathi/Desktop/3d lidar foveated mapping/dataset`)
- **Fallback Archive Backup**: `/Users/amitkumartripathi/Downloads/SemanticPOSS_dataset.zip`
- **Resolution Priority Order**:
  1. CLI Argument: `--dataset-root`
  2. Environment Variable: `DATASET_ROOT`
  3. Repository Default: `dataset/`
