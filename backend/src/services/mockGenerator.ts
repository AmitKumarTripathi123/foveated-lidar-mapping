import {
  RawPoint,
  SemanticPoint,
  BoundingBox3D,
  FoveatedCell,
} from '../types/lidar';

export const CLASS_NAMES: Record<number, string> = {
  0: 'drivable_terrain',
  1: 'non_drivable_terrain',
  2: 'static_obstacle',
  3: 'dynamic_object',
};

export const TRAVERSABILITY_MAP: Record<number, number> = {
  0: 1.0,  // Drivable Road
  1: 0.35, // Sidewalk / Curb
  2: 0.0,  // Static Obstacle
  3: 0.0,  // Dynamic Obstacle
  255: 0.0,
};

export class MockLidarGenerator {
  // Generates 360° surround synthetic 3D LiDAR frames
  public generateRawPointCloud(frameIdx: number, pointCount: number = 18000): RawPoint[] {
    const points: RawPoint[] = [];
    const timeOffset = frameIdx * 0.1;

    // Helper to push
    const addPt = (x: number, y: number, z: number, intensity: number = 0.8) => {
      const distSq = x * x + y * y;
      if (distSq > 10000) return;
      points.push({
        x: Number(x.toFixed(3)),
        y: Number(y.toFixed(3)),
        z: Number(z.toFixed(3)),
        intensity: Number(intensity.toFixed(2)),
      });
    };

    // 1. Drivable Road (North-South: X in [-4.8m, 4.8m], Y in [-50m, 80m])
    for (let i = 0; i < pointCount * 0.28; i++) {
      const y = -48.0 + Math.random() * 128.0;
      const x = -4.8 + Math.random() * 9.6;
      const z = -1.6 + 0.02 * Math.sin(y * 0.15 + timeOffset) + (Math.random() - 0.5) * 0.03;
      addPt(x, y, z, 0.85);
    }

    // East-West Cross Road (Y in [-6m, 6m], X in [-50m, 50m])
    for (let i = 0; i < pointCount * 0.12; i++) {
      const x = -48.0 + Math.random() * 96.0;
      const y = -6.0 + Math.random() * 12.0;
      const z = -1.6 + (Math.random() - 0.5) * 0.03;
      addPt(x, y, z, 0.82);
    }

    // 2. Sidewalks & Curbs (All 4 Quadrants)
    for (let i = 0; i < pointCount * 0.16; i++) {
      const y = -48.0 + Math.random() * 128.0;
      const isEast = Math.random() > 0.5;
      const x = isEast ? 4.8 + Math.random() * 3.2 : -8.0 + Math.random() * 3.2;
      const z = -1.45 + (Math.random() - 0.5) * 0.03;
      addPt(x, y, z, 0.55);
    }

    for (let i = 0; i < pointCount * 0.08; i++) {
      const x = -48.0 + Math.random() * 96.0;
      const isNorth = Math.random() > 0.5;
      const y = isNorth ? 6.0 + Math.random() * 3.0 : -9.0 + Math.random() * 3.0;
      const z = -1.45 + (Math.random() - 0.5) * 0.03;
      addPt(x, y, z, 0.55);
    }

    // 3. Static Obstacles (All 4 Quadrants: Buildings, Trees, Walls)
    for (let i = 0; i < pointCount * 0.08; i++) {
      const x = 9.0 + Math.random() * 38.0;
      const y = 9.0 + Math.random() * 65.0;
      const z = -1.4 + Math.random() * 6.5;
      addPt(x, y, z, 0.72);
    }

    for (let i = 0; i < pointCount * 0.08; i++) {
      const x = -47.0 + Math.random() * 38.0;
      const y = 9.0 + Math.random() * 65.0;
      const z = -1.4 + Math.random() * 6.5;
      addPt(x, y, z, 0.72);
    }

    for (let i = 0; i < pointCount * 0.07; i++) {
      const x = -47.0 + Math.random() * 38.0;
      const y = -48.0 + Math.random() * 38.0;
      const z = -1.4 + Math.random() * 5.5;
      addPt(x, y, z, 0.7);
    }

    for (let i = 0; i < pointCount * 0.07; i++) {
      const x = 9.0 + Math.random() * 38.0;
      const y = -48.0 + Math.random() * 38.0;
      const z = -1.4 + Math.random() * 5.5;
      addPt(x, y, z, 0.7);
    }

    // 4. Dynamic Objects (Vehicles & Pedestrians)
    const v1Y = 12.0 + ((frameIdx * 0.45) % 50.0);
    for (let i = 0; i < pointCount * 0.015; i++) {
      const x = 2.4 + (Math.random() - 0.5) * 1.8;
      const y = v1Y + (Math.random() - 0.5) * 4.2;
      const z = -1.5 + Math.random() * 1.6;
      addPt(x, y, z, 0.95);
    }

    const v2Y = 45.0 - ((frameIdx * 0.5) % 40.0);
    for (let i = 0; i < pointCount * 0.015; i++) {
      const x = -2.4 + (Math.random() - 0.5) * 1.8;
      const y = v2Y + (Math.random() - 0.5) * 4.0;
      const z = -1.5 + Math.random() * 1.5;
      addPt(x, y, z, 0.95);
    }

    const v3Y = -14.0 - ((frameIdx * 0.35) % 25.0);
    for (let i = 0; i < pointCount * 0.012; i++) {
      const x = -2.4 + (Math.random() - 0.5) * 1.8;
      const y = v3Y + (Math.random() - 0.5) * 4.0;
      const z = -1.5 + Math.random() * 1.5;
      addPt(x, y, z, 0.95);
    }

    return points;
  }
}
