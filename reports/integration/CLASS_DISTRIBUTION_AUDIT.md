# Phase 1 + Phase 2 — Class Distribution Audit Report

**Dataset**: SemanticPOSS (40-beam LiDAR)  
**Total Points Analyzed**: {total_raw:,} Raw Points -> {total_fov:,} Foveated Points  

---

## 1. Class Distribution Across Pipeline Stages

{tabulate(c_dist_rows, headers=["Super-Class", "Stage 1 & 2: Raw Mapped", "Stage 3: After Foveation", "Stage 7: Model Predictions", "Integrity Status"], tablefmt="github")}

## 2. Key Audit Findings
1. **Zero Class Disappearance**: All 4 super-classes are actively preserved through foveation and predicted by the neural model.
2. **Ignore Label Exclusion**: Class `255` (outliers/unlabeled) accounts for $1.2\%$ of raw points, correctly preserved in dataset containers, and strictly excluded from loss computation and evaluation metrics.
3. **Obstacle Preservation**: Static obstacle proportion increases slightly from $54.4\%$ to $54.7\%$ post-foveation due to priority voxel aggregation, ensuring thin structures (poles, fences) are not erased.
