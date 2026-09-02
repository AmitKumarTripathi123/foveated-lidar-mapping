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

  // 1. Zone 0 (0–10m @ 5cm): Near-field continuous high-resolution road network
  // Continuous Main Road (-2.2m to +2.2m, full 0-10m radius)
  for (let y = -9.8; y <= 9.8; y += 0.12) {
    for (let x = -2.2; x <= 2.2; x += 0.12) {
      const dist = Math.hypot(x, y);
      if (dist <= 10.0) {
        const z = -1.60 + (prng() - 0.5) * 0.02;
        addPt(x, y, z, 0, 0.98);
      }
    }
  }

  // Zone 0 Continuous Curbs & Sidewalks (bordering road at x in [-3.4, -2.25] and [2.25, 3.4])
  for (let y = -9.8; y <= 9.8; y += 0.12) {
    for (let x of [-3.2, -2.8, -2.4, 2.4, 2.8, 3.2]) {
      const dist = Math.hypot(x, y);
      if (dist <= 10.0) {
        const z = -1.45 + (prng() - 0.5) * 0.02;
        addPt(x, y, z, 1, 0.95);
      }
    }
  }

  // Zone 0 Continuous Crossing Street (y in [-2.0, 2.0], x in [-9.8, 9.8])
  for (let x = -9.8; x <= 9.8; x += 0.12) {
    for (let y = -2.0; y <= 2.0; y += 0.12) {
      const dist = Math.hypot(x, y);
      if (dist <= 10.0) {
        const z = -1.60 + (prng() - 0.5) * 0.02;
        addPt(x, y, z, 0, 0.97);
      }
    }
  }

  // 2. Zone 1 (10–50m @ 25cm): Mid-field continuous road corridor & environment
  // Main Road continues seamlessly into Zone 1 (North y in [9.8, 48.5] and South y in [-48.5, -9.8])
  for (let y = -48.0; y <= 48.0; y += 0.25) {
    if (Math.abs(y) < 9.75) continue; // handoff smoothly to Zone 0
    for (let x = -2.25; x <= 2.25; x += 0.25) {
      const dist = Math.hypot(x, y);
      if (dist > 10.0 && dist <= 50.0) {
        const z = -1.60 + (prng() - 0.5) * 0.02;
        addPt(x, y, z, 0, 0.98);
      }
    }
    // Curbs & Sidewalks continue seamlessly
    for (let x of [-3.25, -2.75, 2.75, 3.25]) {
      const dist = Math.hypot(x, y);
      if (dist > 10.0 && dist <= 50.0) {
        const z = -1.45 + (prng() - 0.5) * 0.02;
        addPt(x, y, z, 1, 0.95);
      }
    }
  }

  // Zone 1 Crossing Street continues seamlessly (y in [-2.0, 2.0], x in [-38.0, 38.0])
  for (let x = -38.0; x <= 38.0; x += 0.25) {
    if (Math.abs(x) < 9.75) continue; // handoff smoothly to Zone 0
    for (let y = -2.0; y <= 2.0; y += 0.25) {
      const dist = Math.hypot(x, y);
      if (dist > 10.0 && dist <= 50.0) {
        const z = -1.60 + (prng() - 0.5) * 0.02;
        addPt(x, y, z, 0, 0.97);
      }
    }
  }

  // Roadside Buildings & Envelopes in Zone 1 (directly adjoining the sidewalks with zero gap!)
  for (let side of [-1, 1]) {
    for (let y of [14, 26, -14, -26]) {
      for (let bx = 3.5; bx <= 7.5; bx += 0.5) {
        for (let by = -3.5; by <= 3.5; by += 0.5) {
          const px = side * bx;
          const py = y + by;
          const dist = Math.hypot(px, py);
          if (dist > 10.0 && dist <= 50.0) {
            const z = -1.50 + prng() * 3.5; // building elevation profile
            addPt(px, py, z, 2, 0.96);
          }
        }
      }
    }
  }

  // Urban Vegetation & Trees in Zone 1 (along sidewalk edge)
  const trees = [
    { x: 3.2, y: 12.0 },
    { x: 3.2, y: 22.0 },
    { x: -3.2, y: 12.0 },
    { x: -3.2, y: 22.0 },
  ];
  for (const tree of trees) {
    for (let i = 0; i < 35; i++) {
      const rx = tree.x + (prng() - 0.5) * 0.8;
      const ry = tree.y + (prng() - 0.5) * 0.8;
      const rz = -1.3 + prng() * 2.8;
      addPt(rx, ry, rz, 4, 0.93);
    }
  }

  // 3. Zone 2 (50–100m @ 50cm): Far-field continuous road corridor
  // Far Main Road continues seamlessly into Zone 2 (North y in [48.0, 75.0] and South y in [-75.0, -48.0])
  for (let y = -75.0; y <= 75.0; y += 0.50) {
    if (Math.abs(y) < 48.0) continue; // handoff smoothly to Zone 1
    for (let x = -2.5; x <= 2.5; x += 0.50) {
      const dist = Math.hypot(x, y);
      if (dist > 50.0 && dist <= 100.0) {
        const z = -1.60 + (prng() - 0.5) * 0.02;
        addPt(x, y, z, 0, 0.96);
      }
    }
    // Road boundaries
    for (let x of [-3.5, 3.5]) {
      const dist = Math.hypot(x, y);
      if (dist > 50.0 && dist <= 100.0) {
        const z = -1.45 + (prng() - 0.5) * 0.02;
        addPt(x, y, z, 1, 0.94);
      }
    }
  }

  // Far Crossing Street in Zone 2
  for (let x = -60.0; x <= 60.0; x += 0.50) {
    if (Math.abs(x) < 38.0) continue;
    for (let y = -2.0; y <= 2.0; y += 0.50) {
      const dist = Math.hypot(x, y);
      if (dist > 50.0 && dist <= 100.0) {
        const z = -1.60 + (prng() - 0.5) * 0.02;
        addPt(x, y, z, 0, 0.95);
      }
    }
  }

  // 4. Dynamic Objects with frame-wise kinematics (cls: 3, Red)
  // Ahead Vehicle moving forward in northbound lane
  const v1Y = 10.0 + ((frameIdx * 0.45) % 30.0);
  for (let dy = -1.8; dy <= 1.8; dy += 0.25) {
    for (let dx = -0.75; dx <= 0.75; dx += 0.25) {
      addPt(1.2 + dx, v1Y + dy, -0.75 + prng() * 0.4, 3, 0.97);
    }
  }

  // Oncoming Vehicle moving south in southbound lane
  const v2Y = 32.0 - ((frameIdx * 0.5) % 26.0);
  for (let dy = -1.8; dy <= 1.8; dy += 0.25) {
    for (let dx = -0.75; dx <= 0.75; dx += 0.25) {
      addPt(-1.2 + dx, v2Y + dy, -0.8 + prng() * 0.4, 3, 0.95);
    }
  }

  // Rear Trailing Vehicle
  const v3Y = -8.0 - ((frameIdx * 0.35) % 15.0);
  for (let dy = -1.8; dy <= 1.8; dy += 0.25) {
    for (let dx = -0.75; dx <= 0.75; dx += 0.25) {
      addPt(1.2 + dx, v3Y + dy, -0.8 + prng() * 0.4, 3, 0.94);
    }
  }

  // Pedestrian on sidewalk
  for (let dy = -0.25; dy <= 0.25; dy += 0.15) {
    for (let dx = -0.25; dx <= 0.25; dx += 0.15) {
      addPt(3.0 + dx, 5.0 + dy, -0.6 + prng() * 0.6, 3, 0.94);
    }
  }

  const boundingBoxes: BoundingBox3D[] = [
    {
      id: 'dyn_veh_01',
      class_name: 'Dynamic Object (Ahead Vehicle)',
      confidence: 0.97,
      center: [1.2, Number(v1Y.toFixed(2)), -0.75],
      size: [1.8, 4.2, 1.5],
      rotation_yaw: 0.0,
    },
    {
      id: 'dyn_veh_02',
      class_name: 'Dynamic Object (Oncoming Car)',
      confidence: 0.95,
      center: [-1.2, Number(v2Y.toFixed(2)), -0.8],
      size: [1.8, 4.2, 1.5],
      rotation_yaw: Math.PI,
    },
    {
      id: 'dyn_veh_03',
      class_name: 'Dynamic Object (Trailing Car)',
      confidence: 0.94,
      center: [1.2, Number(v3Y.toFixed(2)), -0.8],
      size: [1.8, 4.0, 1.5],
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
