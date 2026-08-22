# Phase 11.2 Raw Label Distribution & Ontology Verification

**Project**: Foveated 2.5D LiDAR Mapping for Autonomous Navigation  
**Lead Engineer**: Atul (ML/AI Perception Lead)  
**Date**: August 22, 2026  

---

## 1. Raw Label Distribution

* **Total Scanned Points**: $66,658$ points
* **Supervised Points**: $65,500$ ($98.26\%$)
* **Ignored Points**: $1,158$ ($1.74\%$)

| Raw ID | Class Name | Supervised Count | Percentage | SIH Class ID | SIH Super-Class |
| :---: | :--- | :---: | :---: | :---: | :--- |
| `40` | `road` | $23,000$ | $34.50\%$ | `0` | `drivable_terrain` |
| `48` | `sidewalk` | $8,000$ | $12.00\%$ | `1` | `non_drivable_terrain` |
| `50` | `building` | $10,000$ | $15.00\%$ | `2` | `static_obstacle` |
| `51` | `fence` | $2,000$ | $3.00\%$ | `2` | `static_obstacle` |
| `70` | `vegetation` | $13,000$ | $19.50\%$ | `2` | `static_obstacle` |
| `71` | `trunk` | $2,000$ | $3.00\%$ | `2` | `static_obstacle` |
| `80` | `pole` | $1,500$ | $2.25\%$ | `2` | `static_obstacle` |
| `10` | `car` | $6,000$ | $9.00\%$ | `3` | `dynamic_object` |
| `0` | `unlabeled` | $1,158$ | $1.74\%$ | `255` | `ignore` |
