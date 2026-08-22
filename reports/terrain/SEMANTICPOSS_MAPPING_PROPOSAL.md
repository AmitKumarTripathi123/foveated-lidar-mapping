# SemanticPOSS Super-Class Mapping Proposal

This document establishes the technical rationale and navigation-safety risk assessment for each SemanticPOSS raw class.

---

### Raw ID: 21 (`16-ground/road`)
- **Official Meaning**: Paved asphalt and concrete road surface.
- **Observed Context**: Central vehicle travel corridor with smooth elevation ($z \approx -1.73\\text{m}$) and clear boundaries at $|y| \le 3.5\\text{m}$.
- **Current Mapping**: `0` (`drivable_terrain`)
- **AI Recommendation**: **DRIVABLE (SuperClass 0)**
- **Evidence**: `vis_B_isolated_labels.png`, `vis_D_bev_context.png`
- **Navigation Safety Risk**: **LOW**. Standard vehicle traversal surface.
- **Human Confirmation**: **REQUIRED**

---

### Raw ID: 20 (`15-other-ground`)
- **Official Meaning**: Sidewalks, curbs, pedestrian walkways, manhole covers.
- **Observed Context**: Parallel strips running alongside the road at $3.5 < |y| \le 5.5\\text{m}$ with a $+15\\text{cm}$ vertical curb step.
- **Current Mapping**: `255` (`IGNORE_LABEL`) in legacy config -> **Proposed: `1` (`non_drivable_terrain`)**
- **AI Recommendation**: **NON_DRIVABLE (SuperClass 1)**
- **Evidence**: `vis_D_bev_context.png` (Elevation profile shows $+15\\text{cm}$ curb step)
- **Navigation Safety Risk**: **HIGH if mapped to 0 (curb mounting & pedestrian collision)**; **MEDIUM if mapped to 255 (loss of curb elevation geometry)**; **LOW if mapped to 1 (safely preserves elevation grid while forbidding vehicle planning)**.
- **Human Confirmation**: **REQUIRED**

---

### Raw ID: 19 (`14-terrain`)
- **Official Meaning**: Grass, lawn, flowerbeds, unpaved soil/dirt.
- **Observed Context**: Outer areas at $|y| > 5.5\\text{m}$ beyond sidewalks.
- **Current Mapping**: `1` (`non_drivable_terrain`)
- **AI Recommendation**: **NON_DRIVABLE (SuperClass 1)**
- **Evidence**: `vis_B_isolated_labels.png`, `vis_C_semantic_overlay.png`
- **Navigation Safety Risk**: **HIGH if mapped to 0 (off-road entrapment & vegetation destruction)**; **LOW if mapped to 1**.
- **Human Confirmation**: **REQUIRED**

---

### Raw ID: 22 (`17-outlier`)
- **Official Meaning**: Sensor noise, dust, beam reflections.
- **Observed Context**: Isolated floating points above building roofs and ground.
- **Current Mapping**: `255` (`IGNORE_LABEL`)
- **AI Recommendation**: **IGNORE (SuperClass 255)**
- **Evidence**: `vis_E_superclass_overlay.png`
- **Navigation Safety Risk**: **LOW**. Prevents ghost obstacle false braking.
- **Human Confirmation**: **REQUIRED**

---

### Comparison with SemanticKITTI Mapping

| Navigation Concept | SemanticKITTI Raw Labels | SemanticPOSS Raw Labels | Target Super-Class |
| :--- | :--- | :--- | :--- |
| **Drivable Road** | 40 (road), 44 (parking), 60 (lane-marking) | 21 (ground/road) | **0: drivable_terrain** |
| **Non-Drivable Terrain** | 48 (sidewalk), 49 (other-ground), 72 (terrain) | 20 (other-ground), 19 (terrain) | **1: non_drivable_terrain** |
| **Static Obstacles** | 50 (building), 51 (fence), 70 (vegetation), 80 (pole), 81 (traffic-sign) | 9 (building), 10 (fence), 11 (other-structure), 13 (pole), 14 (sign), 15 (cone), 16 (trashcan), 17 (vegetation), 18 (trunk) | **2: static_obstacle** |
| **Dynamic Objects** | 10 (car), 11 (bicycle), 13 (bus), 15 (motorcycle), 30 (person), 31 (bicyclist), 32 (motorcyclist) | 4 (person), 5 (two-wheelers), 6 (rider), 7 (car), 8 (other-vehicle) | **3: dynamic_object** |
| **Unknown / Outliers** | 0 (unlabeled), 1 (outlier), 99 (other-object) | 0 (unlabeled), 22 (outlier) | **255: IGNORE_LABEL** |
