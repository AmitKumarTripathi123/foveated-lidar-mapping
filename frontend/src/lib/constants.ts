import { LayerVisibility, CameraViewPreset } from '@/types/lidar';

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
export const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';

export const CAMERA_POSITIONS: Record<CameraViewPreset, { pos: [number, number, number]; target: [number, number, number] }> = {
  perspective: {
    pos: [0, -38, 28],
    target: [0, 10, -1],
  },
  top: {
    pos: [0, 15, 95],
    target: [0, 15, 0],
  },
  front: {
    pos: [0, -32, 2.5],
    target: [0, 25, -0.5],
  },
  side: {
    pos: [-55, 15, 12],
    target: [0, 15, 0],
  },
};

export const DEFAULT_CAMERA_POSITION = CAMERA_POSITIONS.perspective.pos;
export const TOP_DOWN_CAMERA_POSITION = CAMERA_POSITIONS.top.pos;
export const CAMERA_TARGET = CAMERA_POSITIONS.perspective.target;

export const INITIAL_LAYERS: LayerVisibility = {
  rawPoints: false,
  semanticPoints: true,
  foveatedGrid: true,
  adaptiveGridWireframe: true,
  traversabilityMap: false,
  boundingBoxes: true,
  zoneRings: true,
  elevationMesh: false,
  egoVehicle: true,
  coordinateAxes: true,
};
