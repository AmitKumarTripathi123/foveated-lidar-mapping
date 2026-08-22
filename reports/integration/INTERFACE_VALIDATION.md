# Phase 1 -> Phase 2 Interface Contract Validation

## 1. Frozen Interface Specifications

### Phase 1 Output Contract: `PointCloudFrame`
- `points`: `np.ndarray` of shape `(N, 4)`, dtype `float32` representing `(x, y, z, intensity)`
- `labels`: `np.ndarray` of shape `(N,)`, dtype `uint32` (or `int64`) with values in `{0, 1, 2, 3, 255}`
- `frame_id`: `str`
- `timestamp`: `float`
- `sequence_id`: `str`
- **Coordinate System**: $+X = 	ext{forward}$, $+Y = 	ext{left}$, $+Z = 	ext{upward}$ (Right-handed, ISO 8855)
- **Units**: Meters for XYZ, normalized $[0, 1]$ float32 for Intensity.

### Phase 2 Output Contract: `SemanticPrediction`
- `points`: `np.ndarray` of shape `(N, 4)`, dtype `float32`
- `predicted_class`: `np.ndarray` of shape `(N,)`, dtype `int64` with values in `{0, 1, 2, 3}`
- `class_probabilities`: `np.ndarray` of shape `(N, 4)`, dtype `float32` in range $[0, 1]$ summing to $1.0$
- `confidence`: `np.ndarray` of shape `(N,)`, dtype `float32` where $	ext{confidence}[i] = \max(P[i])$
- `frame_id`: `str`
- `timestamp`: `float`

---

## 2. Contract Compliance Verification

| Contract Property | Phase 1 Output | Phase 2 Input | Phase 2 Output | Compliance Status |
| :--- | :--- | :--- | :--- | :--- |
| **Spatial Array Shape** | `(N, 4)` | `(N, 4)` | `(N, 4)` | **PASS (Exact Match)** |
| **Data Type** | `float32` | `float32` | `float32` | **PASS (Exact Match)** |
| **Intensity Range** | $[0, 1]$ float32 | $[0, 1]$ float32 | Preserved | **PASS (No re-scaling)** |
| **Coordinate System** | $+X=	ext{fwd}, +Y=	ext{left}, +Z=	ext{up}$ | Preserved | Preserved | **PASS (No axis swaps)** |
| **Label Numbering** | `0, 1, 2, 3, 255` | `0, 1, 2, 3, 255` | `0, 1, 2, 3` | **PASS (Consistent)** |
| **Probability Bounds** | N/A | N/A | $\sum P pprox 1, P \in [0, 1]$ | **PASS (Softmax verified)**|
