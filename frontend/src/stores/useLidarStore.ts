import { create } from 'zustand';
import {
  SemanticPoint,
  FoveatedCell,
  BoundingBox3D,
  FrameMetadata,
  SystemMetrics,
  BenchmarkComparison,
  DatasetInfo,
  LayerVisibility,
  ColorMode,
  ViewMode3D,
  CameraViewPreset,
  PipelineStageId,
  GridDisplayMode,
  GridRenderStyle,
  GridValidationResult,
} from '@/types/lidar';
import { INITIAL_LAYERS } from '@/lib/constants';
import { projectPointsToFoveatedGrid } from '@/lib/gridEngine';

// Deterministic Pseudo-Random Number Generator (LCG)
let seed = 42;
function prng() {
  seed = (seed * 9301 + 49297) % 233280;
  return seed / 233280;
}

// Generates believable, structured LiDAR point clouds for frame N
export function generateStructuredFramePoints(frameIdx: number = 0): {
  points: SemanticPoint[];
  boundingBoxes: BoundingBox3D[];
} {
  seed = 1337 + frameIdx * 37;
  const points: SemanticPoint[] = [];

  // Helper to add classified point
  const addPt = (x: number, y: number, z: number, cls: number, conf: number = 0.95) => {
    const distSq = x * x + y * y;
    if (distSq > 10000) return; // 100m cutoff

    const className =
      cls === 0
        ? 'Drivable Terrain'
        : cls === 1
        ? 'Non-Drivable Terrain'
        : cls === 2
        ? 'Static Obstacle'
        : cls === 3
        ? 'Dynamic Object'
        : cls === 4
        ? 'Vegetation'
        : 'Unknown';

    points.push({
      x: Number(x.toFixed(3)),
      y: Number(y.toFixed(3)),
      z: Number(z.toFixed(3)),
      intensity: 0.85,
      semantic_class: cls,
      class_name: className,
      confidence: conf,
    });
  };

  // 1. Zone 0 (0–10m @ 5cm): Near-field contiguous high-resolution road corridor
  for (let y = -1.5; y <= 1.5; y += 0.05) {
    for (let x = -1.5; x <= 1.5; x += 0.05) {
      const z = -1.60 + (prng() - 0.5) * 0.02;
      addPt(x, y, z, 0, 0.98);
    }
  }
  // Zone 0 Curbs & Sidewalks (Yellow cls: 1)
  for (let y = -1.5; y <= 1.5; y += 0.05) {
    for (let x of [-1.65, -1.55, 1.55, 1.65]) {
      const z = -1.45 + (prng() - 0.5) * 0.03;
      addPt(x, y, z, 1, 0.95);
    }
  }

  // 2. Zone 1 (10–50m @ 25cm): Mid-field contiguous road corridor (Green cls: 0)
  for (let y = 1.75; y <= 24.5; y += 0.25) {
    for (let x = -2.25; x <= 2.25; x += 0.25) {
      const z = -1.60 + (prng() - 0.5) * 0.02;
      addPt(x, y, z, 0, 0.98);
    }
  }
  for (let y = -24.5; y <= -1.75; y += 0.25) {
    for (let x = -2.25; x <= 2.25; x += 0.25) {
      const z = -1.60 + (prng() - 0.5) * 0.02;
      addPt(x, y, z, 0, 0.98);
    }
  }
  // Zone 1 Crossing Intersection
  for (let x = -18.0; x <= 18.0; x += 0.25) {
    for (let y = -1.5; y <= 1.5; y += 0.25) {
      const dist = Math.sqrt(x * x + y * y);
      if (dist > 1.8 && dist <= 24.5) {
        const z = -1.60 + (prng() - 0.5) * 0.02;
        addPt(x, y, z, 0, 0.97);
      }
    }
  }

  // 3. Zone 2 (50–100m @ 50cm): Far-field road corridor (Orange cls: 0)
  for (let y = 25.0; y <= 65.0; y += 0.50) {
    for (let x = -2.5; x <= 2.5; x += 0.50) {
      const z = -1.60 + (prng() - 0.5) * 0.02;
      addPt(x, y, z, 0, 0.96);
    }
  }
  for (let y = -65.0; y <= -25.0; y += 0.50) {
    for (let x = -2.5; x <= 2.5; x += 0.50) {
      const z = -1.60 + (prng() - 0.5) * 0.02;
      addPt(x, y, z, 0, 0.96);
    }
  }

  // 4. Urban Vegetation & Trees (cls: 4, Z = -1.4m to +3.5m)
  const trees = [
    { x: 5.0, y: 12.0 },
    { x: 5.0, y: 22.0 },
    { x: -5.0, y: 12.0 },
    { x: -5.0, y: 22.0 },
  ];
  for (const tree of trees) {
    for (let i = 0; i < 40; i++) {
      const rx = tree.x + (prng() - 0.5) * 1.5;
      const ry = tree.y + (prng() - 0.5) * 1.5;
      const rz = -1.4 + prng() * 3.0;
      addPt(rx, ry, rz, 4, 0.93);
    }
  }

  // 5. Static Obstacles & Building Envelopes (cls: 2, Z = -1.5m to +3.5m)
  const buildings = [
    { x1: 6, x2: 18, y1: 8, y2: 24, h: 3.5 },
    { x1: -18, x2: -6, y1: 8, y2: 24, h: 3.5 },
  ];
  for (const b of buildings) {
    for (let x = b.x1; x <= b.x2; x += 1.0) {
      for (let y = b.y1; y <= b.y2; y += 1.0) {
        const z = -1.5 + prng() * b.h;
        addPt(x, y, z, 2, 0.96);
      }
    }
  }

  // 6. Dynamic Objects with frame-wise kinematics (cls: 3)
  // Ahead Vehicle moving forward
  const v1Y = 10.0 + ((frameIdx * 0.45) % 35.0);
  for (let dy = -2.0; dy <= 2.0; dy += 0.3) {
    for (let dx = -0.8; dx <= 0.8; dx += 0.3) {
      addPt(1.8 + dx, v1Y + dy, -0.75 + prng() * 0.5, 3, 0.97);
    }
  }

  // Oncoming Vehicle moving south
  const v2Y = 32.0 - ((frameIdx * 0.5) % 28.0);
  for (let dy = -2.0; dy <= 2.0; dy += 0.3) {
    for (let dx = -0.8; dx <= 0.8; dx += 0.3) {
      addPt(-1.8 + dx, v2Y + dy, -0.8 + prng() * 0.5, 3, 0.95);
    }
  }

  // Rear Trailing Vehicle
  const v3Y = -8.0 - ((frameIdx * 0.35) % 18.0);
  for (let dy = -2.0; dy <= 2.0; dy += 0.3) {
    for (let dx = -0.8; dx <= 0.8; dx += 0.3) {
      addPt(-1.8 + dx, v3Y + dy, -0.8 + prng() * 0.5, 3, 0.94);
    }
  }

  const boundingBoxes: BoundingBox3D[] = [
    {
      id: 'dyn_veh_01',
      class_name: 'Dynamic Object (Ahead Vehicle)',
      confidence: 0.97,
      center: [2.2, Number(v1Y.toFixed(2)), -0.75],
      size: [1.8, 4.4, 1.6],
      rotation_yaw: 0.0,
    },
    {
      id: 'dyn_veh_02',
      class_name: 'Dynamic Object (Oncoming Car)',
      confidence: 0.95,
      center: [-2.2, Number(v2Y.toFixed(2)), -0.8],
      size: [1.8, 4.2, 1.5],
      rotation_yaw: Math.PI,
    },
    {
      id: 'dyn_veh_03',
      class_name: 'Dynamic Object (Rear Vehicle)',
      confidence: 0.94,
      center: [-2.2, Number(v3Y.toFixed(2)), -0.8],
      size: [1.8, 4.2, 1.5],
      rotation_yaw: 0.0,
    },
  ];

  return { points, boundingBoxes };
}

// Compute Initial Scene Frame 0
const initialData = generateStructuredFramePoints(0);
const initialGridResult = projectPointsToFoveatedGrid(initialData.points, 0);

interface LidarState {
  isConnected: boolean;
  connectionState: 'connected' | 'connecting' | 'reconnecting' | 'simulated' | 'disconnected' | 'dataset_replay';
  datasets: DatasetInfo[];
  activeDatasetId: string;
  totalFrames: number;
  currentFrameIdx: number;
  playbackState: 'idle' | 'running' | 'paused';
  targetFps: number;

  metadata: FrameMetadata | null;
  points: SemanticPoint[];
  cells: FoveatedCell[];
  boundingBoxes: BoundingBox3D[];
  metrics: SystemMetrics | null;
  benchmark: BenchmarkComparison | null;
  validation: GridValidationResult;

  activePipelineStage: PipelineStageId;
  viewMode3D: ViewMode3D;
  cameraPreset: CameraViewPreset;
  colorMode: ColorMode;
  gridDisplayMode: GridDisplayMode;
  gridRenderStyle: GridRenderStyle;
  layers: LayerVisibility;
  pointSize: number;
  gridOpacity: number;
  elevationExaggeration: number;

  isPresentationMode: boolean;
  presentationStep: number;

  selectedCell: FoveatedCell | null;
  selectedBox: BoundingBox3D | null;
  hoveredCell: FoveatedCell | null;
  isComparisonOpen: boolean;
  isSettingsOpen: boolean;

  setIsConnected: (connected: boolean) => void;
  setConnectionState: (state: 'connected' | 'connecting' | 'reconnecting' | 'simulated' | 'disconnected' | 'dataset_replay') => void;
  setDatasets: (datasets: DatasetInfo[]) => void;
  setActiveDatasetId: (id: string) => void;
  setPlaybackState: (state: 'idle' | 'running' | 'paused') => void;
  setTargetFps: (fps: number) => void;
  setCurrentFrameIdx: (idx: number) => void;
  setFrameData: (data: {
    metadata: FrameMetadata;
    points: SemanticPoint[];
    map: { cells: FoveatedCell[]; total_cells: number };
    metrics: SystemMetrics;
    benchmark?: BenchmarkComparison;
  }) => void;
  setActivePipelineStage: (stage: PipelineStageId) => void;
  setViewMode3D: (mode: ViewMode3D) => void;
  setGridDisplayMode: (mode: GridDisplayMode) => void;
  setGridRenderStyle: (style: GridRenderStyle) => void;
  setCameraPreset: (preset: CameraViewPreset) => void;
  setColorMode: (mode: ColorMode) => void;
  toggleLayer: (layer: keyof LayerVisibility) => void;
  setLayer: (layer: keyof LayerVisibility, value: boolean) => void;
  setPointSize: (size: number) => void;
  setGridOpacity: (opacity: number) => void;
  measuredFps: number;
  setMeasuredFps: (fps: number) => void;
  setSelectedCell: (cell: FoveatedCell | null) => void;
  setSelectedBox: (box: BoundingBox3D | null) => void;
  setHoveredCell: (cell: FoveatedCell | null) => void;
  setIsComparisonOpen: (open: boolean) => void;
  setIsSettingsOpen: (open: boolean) => void;
  setIsPresentationMode: (mode: boolean) => void;
  setPresentationStep: (step: number) => void;
  applyPipelinePreset: (stage: PipelineStageId) => void;
  applyPresentationStep: (step: number) => void;
  stepFrame: (delta: number) => void;
}

export const useLidarStore = create<LidarState>((set, get) => ({
  isConnected: false,
  connectionState: 'dataset_replay',
  measuredFps: 60,
  setMeasuredFps: (fps) => set({ measuredFps: fps }),
  datasets: [],
  activeDatasetId: 'urban_driving_demo_01',
  totalFrames: 100,
  currentFrameIdx: 0,
  playbackState: 'running',
  targetFps: 10,

  metadata: {
    frame_id: 0,
    timestamp_ms: 1724544000000,
    total_points: initialData.points.length,
    sequence_id: 'urban_driving_demo_01',
    bounding_boxes: initialData.boundingBoxes,
  },
  points: initialData.points,
  cells: initialGridResult.cells,
  boundingBoxes: initialData.boundingBoxes,
  validation: initialGridResult.validation,
  metrics: {
    fps: 10.0,
    total_latency_ms: 30.3,
    ai_latency_ms: 18.2,
    grid_latency_ms: initialGridResult.gridLatencyMs || 12.1,
    memory_ram_mb: 134.8,
    memory_vram_mb: 215.5,
    cpu_percent: 18.5,
    raw_point_count: initialData.points.length,
    cell_count: initialGridResult.cells.length,
    compression_ratio_percent: 82.8,
  },
  benchmark: initialGridResult.benchmark,

  // Primary Default: 2.5D Elevation Map + Top-Down BEV view + Pure Grid Map
  activePipelineStage: 'elevation_25d',
  viewMode3D: 'foveated_elevation',
  cameraPreset: 'top',
  colorMode: 'semantic',
  gridDisplayMode: 'grid',
  gridRenderStyle: 'top_down_2d',
  layers: {
    rawPoints: false,
    semanticPoints: false,
    foveatedGrid: true,
    adaptiveGridWireframe: true,
    traversabilityMap: false,
    boundingBoxes: true,
    zoneRings: true,
    elevationMesh: false,
    egoVehicle: true,
    coordinateAxes: true,
  },
  pointSize: 3.5,
  gridOpacity: 0.9,
  elevationExaggeration: 1.0,

  isPresentationMode: false,
  presentationStep: 1,

  selectedCell: null,
  selectedBox: null,
  hoveredCell: null,
  isComparisonOpen: false,
  isSettingsOpen: false,

  setIsConnected: (connected) => set({ isConnected: connected }),
  setConnectionState: (state) => set({ connectionState: state }),
  setDatasets: (datasets) => set({ datasets }),
  setActiveDatasetId: (id) => set({ activeDatasetId: id }),
  setPlaybackState: (state) => set({ playbackState: state }),
  setTargetFps: (fps) => set({ targetFps: fps }),

  setCurrentFrameIdx: (idx: number) => {
    const frameData = generateStructuredFramePoints(idx);
    const gridRes = projectPointsToFoveatedGrid(frameData.points, idx);

    set({
      currentFrameIdx: idx,
      points: frameData.points,
      cells: gridRes.cells,
      boundingBoxes: frameData.boundingBoxes,
      validation: gridRes.validation,
      benchmark: gridRes.benchmark,
      metadata: {
        frame_id: idx,
        timestamp_ms: 1724544000000 + idx * 100,
        total_points: frameData.points.length,
        sequence_id: get().activeDatasetId,
        bounding_boxes: frameData.boundingBoxes,
      },
      metrics: {
        fps: get().targetFps,
        total_latency_ms: Number((18.2 + gridRes.gridLatencyMs).toFixed(1)),
        ai_latency_ms: 18.2,
        grid_latency_ms: gridRes.gridLatencyMs,
        memory_ram_mb: 134.8,
        memory_vram_mb: 215.5,
        cpu_percent: 18.5,
        raw_point_count: frameData.points.length,
        cell_count: gridRes.cells.length,
        compression_ratio_percent: 82.8,
      },
    });
  },

  stepFrame: (delta: number) => {
    const total = get().totalFrames;
    const current = get().currentFrameIdx;
    const nextIdx = (current + delta + total) % total;
    get().setCurrentFrameIdx(nextIdx);
  },

  setFrameData: (data) =>
    set({
      metadata: data.metadata,
      points: data.points,
      cells: data.map.cells,
      boundingBoxes: data.metadata.bounding_boxes || [],
      metrics: data.metrics,
      benchmark: data.benchmark || get().benchmark,
      currentFrameIdx: data.metadata.frame_id,
    }),

  setActivePipelineStage: (stage) => set({ activePipelineStage: stage }),

  setGridDisplayMode: (mode: GridDisplayMode) => {
    set({ gridDisplayMode: mode });
    const currentLayers = get().layers;
    if (mode === 'grid') {
      set({
        layers: {
          ...currentLayers,
          rawPoints: false,
          semanticPoints: false,
          foveatedGrid: true,
          adaptiveGridWireframe: true,
        },
      });
    } else if (mode === 'points') {
      set({
        layers: {
          ...currentLayers,
          rawPoints: false,
          semanticPoints: true,
          foveatedGrid: false,
          adaptiveGridWireframe: false,
        },
      });
    } else if (mode === 'both') {
      set({
        layers: {
          ...currentLayers,
          rawPoints: false,
          semanticPoints: true,
          foveatedGrid: true,
          adaptiveGridWireframe: true,
        },
      });
    }
  },

  setGridRenderStyle: (style: GridRenderStyle) => set({ gridRenderStyle: style }),

  setViewMode3D: (mode: ViewMode3D) => {
    const currentLayers = get().layers;
    switch (mode) {
      case 'raw':
        set({
          viewMode3D: mode,
          colorMode: 'intensity',
          activePipelineStage: 'raw',
          gridDisplayMode: 'points',
          layers: {
            ...currentLayers,
            rawPoints: true,
            semanticPoints: false,
            foveatedGrid: false,
            adaptiveGridWireframe: false,
            zoneRings: false,
            boundingBoxes: false,
            traversabilityMap: false,
          },
        });
        break;
      case 'semantic':
        set({
          viewMode3D: mode,
          colorMode: 'semantic',
          activePipelineStage: 'semantic',
          gridDisplayMode: 'points',
          layers: {
            ...currentLayers,
            rawPoints: false,
            semanticPoints: true,
            foveatedGrid: false,
            adaptiveGridWireframe: false,
            zoneRings: false,
            boundingBoxes: true,
            traversabilityMap: false,
          },
        });
        break;
      case 'foveated':
        set({
          viewMode3D: mode,
          colorMode: 'semantic',
          activePipelineStage: 'foveation',
          gridDisplayMode: 'grid',
          layers: {
            ...currentLayers,
            rawPoints: false,
            semanticPoints: false,
            foveatedGrid: true,
            adaptiveGridWireframe: true,
            zoneRings: true,
            boundingBoxes: true,
            traversabilityMap: false,
          },
        });
        break;
      case 'elevation':
        set({
          viewMode3D: mode,
          colorMode: 'elevation',
          activePipelineStage: 'data',
          gridDisplayMode: 'points',
          layers: {
            ...currentLayers,
            rawPoints: true,
            semanticPoints: false,
            foveatedGrid: false,
            adaptiveGridWireframe: false,
            zoneRings: false,
            boundingBoxes: false,
            traversabilityMap: false,
          },
        });
        break;
      case 'foveated_semantic':
        set({
          viewMode3D: mode,
          colorMode: 'semantic',
          activePipelineStage: 'variable_grid',
          gridDisplayMode: 'both',
          layers: {
            ...currentLayers,
            rawPoints: false,
            semanticPoints: true,
            foveatedGrid: true,
            adaptiveGridWireframe: true,
            zoneRings: true,
            boundingBoxes: true,
            traversabilityMap: false,
          },
        });
        break;
      case 'foveated_elevation':
        set({
          viewMode3D: mode,
          colorMode: 'semantic',
          activePipelineStage: 'elevation_25d',
          gridDisplayMode: 'grid',
          layers: {
            ...currentLayers,
            rawPoints: false,
            semanticPoints: false,
            foveatedGrid: true,
            adaptiveGridWireframe: true,
            zoneRings: true,
            boundingBoxes: true,
            traversabilityMap: false,
          },
        });
        break;
    }
  },

  setCameraPreset: (preset) => set({ cameraPreset: preset }),
  setColorMode: (mode) => set({ colorMode: mode }),

  toggleLayer: (layer) =>
    set((state) => ({
      layers: { ...state.layers, [layer]: !state.layers[layer] },
    })),

  setLayer: (layer, value) =>
    set((state) => ({
      layers: { ...state.layers, [layer]: value },
    })),

  setPointSize: (size) => set({ pointSize: size }),
  setGridOpacity: (opacity) => set({ gridOpacity: opacity }),
  setSelectedCell: (cell) => set({ selectedCell: cell }),
  setSelectedBox: (box) => set({ selectedBox: box }),
  setHoveredCell: (cell) => set({ hoveredCell: cell }),
  setIsComparisonOpen: (open) => set({ isComparisonOpen: open }),
  setIsSettingsOpen: (open) => set({ isSettingsOpen: open }),
  setIsPresentationMode: (mode) => set({ isPresentationMode: mode }),
  setPresentationStep: (step) => set({ presentationStep: step }),

  applyPipelinePreset: (stage: PipelineStageId) => {
    set({ activePipelineStage: stage });
    switch (stage) {
      case 'data':
      case 'raw':
        get().setViewMode3D('raw');
        break;
      case 'ai':
      case 'semantic':
        get().setViewMode3D('semantic');
        break;
      case 'foveation':
        get().setViewMode3D('foveated');
        break;
      case 'variable_grid':
        get().setViewMode3D('foveated_semantic');
        break;
      case 'elevation_25d':
        get().setViewMode3D('foveated_elevation');
        break;
      case 'benchmark':
        set({ isComparisonOpen: true });
        break;
    }
  },

  applyPresentationStep: (step: number) => {
    set({ isPresentationMode: true, presentationStep: step });
    switch (step) {
      case 1:
        get().setViewMode3D('raw');
        break;
      case 2:
        get().setViewMode3D('semantic');
        break;
      case 3:
        get().setViewMode3D('foveated');
        break;
      case 4:
        get().setViewMode3D('foveated_semantic');
        break;
      case 5:
        get().setViewMode3D('foveated_elevation');
        break;
      case 6:
        set({ isComparisonOpen: true });
        break;
    }
  },
}));
