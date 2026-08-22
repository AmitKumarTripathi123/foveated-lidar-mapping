# Current SemanticPOSS Label Mapping in Repository

**Configuration File**: `configs/semanticposs_mapping.yaml`

|   Raw ID |   Project Class ID | Super-Class Name     | Official SemanticPOSS Name   | Official Description                                        |
|----------|--------------------|----------------------|------------------------------|-------------------------------------------------------------|
|        0 |                255 | ignore_label         | unlabeled                    | Unlabeled or unclassified noise                             |
|        4 |                  3 | dynamic_object       | 1-person                     | Pedestrians, standing/walking persons                       |
|        5 |                  3 | dynamic_object       | 2-rider                      | Riders on bicycles/motorcycles                              |
|        6 |                  3 | dynamic_object       | 3-car                        | Automobiles, sedans, SUVs                                   |
|        7 |                  3 | dynamic_object       | 4-trunk                      | Trucks, heavy transport vehicles                            |
|        8 |                  3 | dynamic_object       | 5-plants                     | Shrubs, bushes, lower plants                                |
|        9 |                  2 | static_obstacle      | 6-traffic-sign               | Traffic signs, boards, billboards                           |
|       10 |                  2 | static_obstacle      | 7-pole                       | Poles, lamp posts, vertical narrow fixtures                 |
|       11 |                  2 | static_obstacle      | 8-trashcan                   | Trash bins, recycling containers                            |
|       12 |                255 | ignore_label         | 9-building                   | Buildings, architectural facades, walls                     |
|       13 |                  2 | static_obstacle      | 10-cone/stone                | Traffic cones, warning markers, bollards                    |
|       14 |                  2 | static_obstacle      | 11-fence                     | Fences, guardrails, barriers                                |
|       17 |                  2 | static_obstacle      | 12-vegetation                | Foliage, tree canopies, large vegetation                    |
|       18 |                  2 | static_obstacle      | 13-trunk                     | Tree trunks, vertical wood structures                       |
|       19 |                  1 | non_drivable_terrain | 14-terrain                   | Grass, lawn, dirt, gravel, unpaved surfaces                 |
|       20 |                255 | ignore_label         | 15-other-ground              | Sidewalks, curbs, manholes, paved non-road pedestrian areas |
|       21 |                  0 | drivable_terrain     | 16-ground/road               | Main paved asphalt/concrete road, drivable road surface     |
|       22 |                255 | ignore_label         | 17-outlier                   | Sensor artifacts, optical reflections, dust                 |