export interface SemanticColorDef {
  id: number;
  name: string;
  hex: string;
  rgb: [number, number, number];
  traversability: number;
  description: string;
}

// Authoritative Semantic Ontology matching project specifications
export const SEMANTIC_CLASSES: Record<number, SemanticColorDef> = {
  0: {
    id: 0,
    name: 'Drivable Terrain',
    hex: '#22C55E', // Green
    rgb: [0.13, 0.77, 0.37],
    traversability: 1.0,
    description: 'Drivable asphalt, road surface, and intersection lanes',
  },
  1: {
    id: 1,
    name: 'Non-Drivable Terrain',
    hex: '#F59E0B', // Amber / Orange
    rgb: [0.96, 0.62, 0.04],
    traversability: 0.35,
    description: 'Sidewalks, road curbs, paved shoulders, and crosswalks',
  },
  2: {
    id: 2,
    name: 'Static Obstacle',
    hex: '#A855F7', // Purple
    rgb: [0.66, 0.33, 0.97],
    traversability: 0.0,
    description: 'Buildings, utility poles, guard rails, walls, fences',
  },
  3: {
    id: 3,
    name: 'Dynamic Object',
    hex: '#EF4444', // Red
    rgb: [0.94, 0.27, 0.27],
    traversability: 0.0,
    description: 'Moving/Parked vehicles, pedestrians, cyclists, buses',
  },
  4: {
    id: 4,
    name: 'Vegetation',
    hex: '#15803D', // Dark Green
    rgb: [0.08, 0.50, 0.24],
    traversability: 0.1,
    description: 'Roadside trees, bushes, lawn, and urban greenery',
  },
  255: {
    id: 255,
    name: 'Unknown / Background',
    hex: '#64748B', // Slate Gray
    rgb: [0.39, 0.45, 0.54],
    traversability: 0.0,
    description: 'Outliers, occluded regions, and unclassified returns',
  },
};

// 3-Zone Distance-Adaptive Spatial Foveation Scheme
export const FOVEATED_ZONE_COLORS: Record<number, { name: string; hex: string; resolution: string; radius: string; cellSize: number }> = {
  0: { name: 'ZONE 0 — FOVEAL / NEAR', hex: '#38BDF8', resolution: '5 cm (0.05m)', radius: '0–10m', cellSize: 0.05 },
  1: { name: 'ZONE 1 — INTERMEDIATE', hex: '#F59E0B', resolution: '25 cm (0.25m)', radius: '10–50m', cellSize: 0.25 },
  2: { name: 'ZONE 2 — PERIPHERAL / FAR', hex: '#A855F7', resolution: '50 cm (0.50m)', radius: '50–100m', cellSize: 0.50 },
};

export function getSemanticColor(classId: number): [number, number, number] {
  return SEMANTIC_CLASSES[classId]?.rgb || SEMANTIC_CLASSES[0].rgb;
}

export function getSemanticHex(classId: number): string {
  return SEMANTIC_CLASSES[classId]?.hex || SEMANTIC_CLASSES[0].hex;
}

// Scientific Elevation Colormap (Turbo / Terrain height mapping from -1.65m to +4.0m)
export function getElevationColor(elevation: number, minEl: number = -1.65, maxEl: number = 3.5): [number, number, number] {
  const norm = Math.max(0, Math.min(1, (elevation - minEl) / (maxEl - minEl)));
  if (norm < 0.2) {
    // Deep blue to cyan (ground datum -1.65m to -1.2m)
    return [0.05, 0.35 + norm * 3.0, 0.95];
  } else if (norm < 0.45) {
    // Cyan to green (low curbs / ground slope -1.2m to 0m)
    const t = (norm - 0.2) / 0.25;
    return [0.1, 0.9, 0.95 - t * 0.8];
  } else if (norm < 0.75) {
    // Green to amber/yellow (vehicles / low obstacles 0m to 1.8m)
    const t = (norm - 0.45) / 0.3;
    return [0.2 + t * 0.75, 0.85 + t * 0.1, 0.1];
  } else {
    // Yellow to vibrant red/crimson (tall structures / buildings > 2.0m)
    const t = (norm - 0.75) / 0.25;
    return [0.95, 0.95 - t * 0.75, 0.1];
  }
}

export function getTraversabilityColor(traversability: number): [number, number, number] {
  if (traversability >= 0.8) {
    return [0.13, 0.77, 0.37]; // Green - Safe Drivable
  } else if (traversability >= 0.3) {
    return [0.96, 0.62, 0.04]; // Amber - Caution / Off-road
  } else {
    return [0.94, 0.27, 0.27]; // Red - Hazard / Impassable Obstacle
  }
}
