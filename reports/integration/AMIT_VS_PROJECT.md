# Architectural Comparison: Amit Reference vs Integrated Project

| Component | Amit Reference Implementation | Integrated Project | Classification | Impact on Phase 2 |
| :--- | :--- | :--- | :--- | :--- |
| **LiDAR Sensor** | Hesai Pandar40 (40-beam, $1800 	imes 40$) | Hesai Pandar40 (40-beam, $1800 	imes 40$) | **MATCH** | Identical sensor characteristics |
| **Foveation Bands** | 0-10m @ 0.05m, 10-40m @ 0.15m, 40-100m @ 0.50m | 0-10m @ 0.05m, 10-40m @ 0.15m, 40-100m @ 0.50m | **MATCH** | Identical distance bands |
| **Range Filter** | $r = \sqrt{x^2+y^2} < 100.0	ext{m}$ | $r = \sqrt{x^2+y^2} \le 100.0	ext{m}$ | **MATCH** | Identical 2D horizontal boundary |
| **Voxel Aggregation**| First point in hash cell | Modular (`obstacle_preserving` + `amit_first_point`) | **INTENTIONAL DIFFERENCE** | Enhances obstacle retention in near/mid field |
| **Label Mapper** | `class_map.py` (`remap_labels`) | `SEMANTICPOSS_TO_PROJECT` single adapter | **MATCH** | Same mapping logic, single conversion point |
| **Data Split** | Train: 00,01,03,04,05 / Val: 02 | Sequence-based non-leaking loader | **MATCH** | Zero frame overlap |
| **AI Model Input** | NumPy arrays | PyTorch FloatTensor `(N, 4)` | **MATCH** | Seamless zero-copy tensor ingestion |
| **Interface Contract**| `.npy` disk caching | In-memory `SemanticPrediction` dataclass | **INTENTIONAL DIFFERENCE** | Optimized for real-time Phase 3 costmap engine |
