# SemanticPOSS Human Terrain Review & Decision Table

**Target Application**: Autonomous Navigation Path Planning & 2.5D Elevation Grid Mapping

Allowed Decisions: `DRIVABLE`, `NON_DRIVABLE`, `STATIC_OBSTACLE`, `DYNAMIC_OBJECT`, `IGNORE`

|   Raw ID | Official Meaning                | AI Assessment          |   Current Mapping | Visual Evidence     | Human Decision   |
|----------|---------------------------------|------------------------|-------------------|---------------------|------------------|
|       21 | 16-ground/road                  | Likely Drivable        |                 0 | vis_B, vis_C, vis_D | [ ]              |
|       20 | 15-other-ground (sidewalk/curb) | Likely Non-Drivable    |               255 | vis_B, vis_C, vis_D | [ ]              |
|       19 | 14-terrain (grass/lawn/dirt)    | Likely Non-Drivable    |                 1 | vis_B, vis_C, vis_D | [ ]              |
|       22 | 17-outlier (sensor noise)       | Ignore                 |               255 | vis_E               | [ ]              |
|        4 | 1-person (pedestrian)           | Likely Dynamic Object  |                 3 | vis_C, vis_E        | [ ]              |
|        5 | 2-rider (cyclist/motorcyclist)  | Likely Dynamic Object  |                 3 | vis_C, vis_E        | [ ]              |
|        6 | 3-car (sedan/SUV)               | Likely Dynamic Object  |                 3 | vis_C, vis_E        | [ ]              |
|        7 | 4-trunk (heavy vehicle)         | Likely Dynamic Object  |                 3 | vis_C, vis_E        | [ ]              |
|        8 | 5-plants (bushes/shrubs)        | Likely Static Obstacle |                 3 | vis_C               | [ ]              |
|        9 | 6-traffic-sign                  | Likely Static Obstacle |                 2 | vis_A               | [ ]              |
|       10 | 7-pole                          | Likely Static Obstacle |                 2 | vis_A               | [ ]              |
|       11 | 8-trashcan                      | Likely Static Obstacle |                 2 | vis_A               | [ ]              |
|       12 | 9-building                      | Likely Static Obstacle |                 2 | vis_A               | [ ]              |
|       13 | 10-cone/stone                   | Likely Static Obstacle |                 2 | vis_A               | [ ]              |
|       14 | 11-fence                        | Likely Static Obstacle |                 2 | vis_A               | [ ]              |
|       17 | 12-vegetation (trees)           | Likely Static Obstacle |                 2 | vis_A               | [ ]              |
|       18 | 13-trunk                        | Likely Static Obstacle |                 2 | vis_A               | [ ]              |

## Human Confirmation Sign-Off

- **Reviewer Name**: __________________________
- **Date**: __________________________
- **Decision Summary**: __________________________
- **Operational Approval**: [ ] APPROVED   [ ] REJECTED
