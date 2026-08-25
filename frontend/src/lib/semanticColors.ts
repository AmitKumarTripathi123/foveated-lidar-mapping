export interface SemanticColorDef {
  id: number;
  name: string;
  hex: string;
  rgb: [number, number, number];
  traversability: number;
  description: string;
}

// Authoritative 4-Super-Class Ontology from class_map.py
export const SEMANTIC_CLASSES: Record<number, SemanticColorDef> = {
  0: {
    id: 0,
    name: 'drivable_terrain',
    hex: '#38BDF8', // Steel Blue / Sky Cyan
    rgb: [0.22, 0.74, 0.97],
    traversability: 1.0,
    description: 'Drivable asphalt, concrete, and road surfaces',
  },
  1: {
    id: 1,
    name: 'non_drivable_terrain',
    hex: '#EAB308', // Goldenrod
    rgb: [0.92, 0.70, 0.03],
    traversability: 0.35,
    description: 'Sidewalks, curbs, off-road terrain, gravel',
  },
  2: {
    id: 2,
    name: 'static_obstacle',
    hex: '#EF4444', // Firebrick Red / Coral
    rgb: [0.94, 0.27, 0.27],
    traversability: 0.0,
    description: 'Buildings, poles, trees, fences, vegetation',
  },
  3: {
    id: 3,
    name: 'dynamic_object',
    hex: '#22C55E', // Lime Green
    rgb: [0.13, 0.77, 0.37],
    traversability: 0.0,
    description: 'Moving vehicles, pedestrians, cyclists, riders',
  },
  255: {
    id: 255,
    name: 'ignore',
    hex: '#64748B',
    rgb: [0.39, 0.45, 0.54],
    traversability: 0.0,
    description: 'Outliers and unclassified points',
  },
};

// 3-Zone Distance-Adaptive Foveation Scheme from README & Phase 17 Freeze
export const FOVEATED_ZONE_COLORS: Record<number, { name: string; hex: string; resolution: string; radius: string }> = {
  0: { name: 'Zone 0 — Near-Field (Foveal)', hex: '#38BDF8', resolution: '5 cm (0.05m)', radius: '0–10m' },
  1: { name: 'Zone 1 — Mid-Field (Intermediate)', hex: '#F59E0B', resolution: '15 cm (0.15m)', radius: '10–40m' },
  2: { name: 'Zone 2 — Far-Field (Peripheral)', hex: '#A855F7', resolution: '50 cm (0.50m)', radius: '40–100m' },
};

export function getSemanticColor(classId: number): [number, number, number] {
  return SEMANTIC_CLASSES[classId]?.rgb || SEMANTIC_CLASSES[0].rgb;
}

export function getElevationColor(elevation: number, minEl: number = -2.5, maxEl: number = 2.5): [number, number, number] {
  const norm = Math.max(0, Math.min(1, (elevation - minEl) / (maxEl - minEl)));
  if (norm < 0.25) {
    return [0.0, norm * 4.0, 1.0];
  } else if (norm < 0.5) {
    return [0.0, 1.0, 1.0 - (norm - 0.25) * 4.0];
  } else if (norm < 0.75) {
    return [(norm - 0.5) * 4.0, 1.0, 0.0];
  } else {
    return [1.0, 1.0 - (norm - 0.75) * 4.0, 0.0];
  }
}

export function getTraversabilityColor(traversability: number): [number, number, number] {
  if (traversability >= 0.8) {
    return [0.22, 0.74, 0.97]; // Sky Cyan - Drivable
  } else if (traversability >= 0.3) {
    return [0.92, 0.70, 0.03]; // Goldenrod - Caution
  } else {
    return [0.94, 0.27, 0.27]; // Red - Hazard
  }
}
