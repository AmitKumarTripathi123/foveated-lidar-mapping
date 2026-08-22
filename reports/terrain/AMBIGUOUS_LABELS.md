# Analysis of Ambiguous SemanticPOSS Labels

This report investigates raw labels with potential ambiguity in autonomous navigation classification.

## 1. Summary of Investigated Labels

| Raw ID | Dataset Label | Navigational Ambiguity | Risk Level |
| :--- | :--- | :--- | :--- |
| **19** | `14-terrain` | Grass / lawn / unpaved dirt. Often flat like road, but not vehicle-drivable. | **HIGH (False Drivable if mapped to 0)** |
| **20** | `15-other-ground` | Sidewalks, curbs, manhole covers, pedestrian plaza. Physically traversable in emergency but strictly non-drivable under standard road rules. | **HIGH (Boundary Hazard)** |
| **21** | `16-ground/road` | Asphalt / concrete road. Standard vehicle travel path. | **LOW (Clear Drivable)** |
| **22** | `17-outlier` | Sensor artifacts, optical noise, reflections in air/ground. | **LOW (Clear Ignore)** |

## 2. In-Depth Label Analysis

### Label 19 (`terrain`)
- **Official Definition**: Grass, flowerbeds, dirt lawns, unpaved natural surfaces.
- **LiDAR Characteristics**: Elevation is approximately at ground level ($z \approx -1.73\\text{m}$), lateral position $|y| > 5.5\\text{m}$.
- **AI Assessment**: **LIKELY NON_DRIVABLE (Super-Class 1)**.
- **Rationale**: If mapped to `0 (drivable_terrain)`, the vehicle path planner could consider off-road grass/dirt as safe drivable space, risking vehicle immobilization or lawn damage. Mapping to `1 (non_drivable_terrain)` correctly marks terrain elevation while forbidding vehicle traversal.

### Label 20 (`other-ground`)
- **Official Definition**: Sidewalks, curbs, pedestrian walkways, paved surfaces adjacent to roads.
- **LiDAR Characteristics**: Elevated $10-15\\text{cm}$ above the main road plane with a clear vertical curb step ($z \approx -1.58\\text{m}$). Lateral position $3.5 < |y| \le 5.5\\text{m}$.
- **AI Assessment**: **LIKELY NON_DRIVABLE (Super-Class 1)**.
- **Rationale**: Mapping to `255 (ignore)` removes critical sidewalk curb geometry from the 2.5D elevation map. Mapping to `0 (drivable)` risks mounting sidewalks and endangering pedestrians. Mapping to `1 (non_drivable_terrain)` preserves the curb height step in 2.5D mapping while designating it non-drivable for navigation.

### Label 21 (`ground/road`)
- **Official Definition**: Asphalt, concrete, drivable vehicle road.
- **AI Assessment**: **LIKELY DRIVABLE (Super-Class 0)**.
- **Rationale**: Primary vehicle operating corridor ($|y| \le 3.5\\text{m}$).

### Label 22 (`outlier`)
- **Official Definition**: Sensor noise, flare, multipath reflections.
- **AI Assessment**: **LIKELY IGNORE (Super-Class 255)**.
- **Rationale**: Outliers floating above the scene should be ignored to prevent false obstacle phantom braking.
