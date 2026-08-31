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
} from '@/types/lidar';
import { INITIAL_LAYERS } from '@/lib/constants';

// Deterministic Pseudo-Random Number Generator (LCG) to eliminate React Hydration Mismatches
let seed = 42;
function prng() {
  seed = (seed * 9301 + 49297) % 233280;
  return seed / 233280;
}

// Deterministic pre-populated 360° surround scene
function createInitialScene() {
  seed = 1337; // Fixed seed for 100% identical SSR & Client rendering
  const points: SemanticPoint[] = [];
  const cells: FoveatedCell[] = [];
  const pointCount = 10000;

  // Add 360° road points
  for (let i = 0; i < pointCount * 0.35; i++) {
    const y = -40.0 + prng() * 110.0;
    const x = -4.5 + prng() * 9.0;
    const z = -1.6 + (prng() - 0.5) * 0.03;
    points.push({
      x: Number(x.toFixed(3)),
      y: Number(y.toFixed(3)),
      z: Number(z.toFixed(3)),
      intensity: 0.85,
      semantic_class: 0,
      class_name: 'drivable_terrain',
      confidence: 0.98,
    });
  }

  // Cross intersection
  for (let i = 0; i < pointCount * 0.15; i++) {
    const x = -40.0 + prng() * 80.0;
    const y = -5.0 + prng() * 10.0;
    const z = -1.6 + (prng() - 0.5) * 0.03;
    points.push({
      x: Number(x.toFixed(3)),
      y: Number(y.toFixed(3)),
      z: Number(z.toFixed(3)),
      intensity: 0.8,
      semantic_class: 0,
      class_name: 'drivable_terrain',
      confidence: 0.95,
    });
  }

  // Sidewalks
  for (let i = 0; i < pointCount * 0.2; i++) {
    const y = -40.0 + prng() * 110.0;
    const isEast = prng() > 0.5;
    const x = isEast ? 4.5 + prng() * 3.0 : -7.5 + prng() * 3.0;
    const z = -1.45 + (prng() - 0.5) * 0.03;
    points.push({
      x: Number(x.toFixed(3)),
      y: Number(y.toFixed(3)),
      z: Number(z.toFixed(3)),
      intensity: 0.55,
      semantic_class: 1,
      class_name: 'non_drivable_terrain',
      confidence: 0.94,
    });
  }

  // Static obstacles
  for (let i = 0; i < pointCount * 0.25; i++) {
    const isNorth = prng() > 0.4;
    const y = isNorth ? 8.0 + prng() * 60.0 : -38.0 + prng() * 30.0;
    const isEast = prng() > 0.5;
    const x = isEast ? 8.0 + prng() * 30.0 : -38.0 + prng() * 30.0;
    const z = -1.4 + prng() * 5.0;
    points.push({
      x: Number(x.toFixed(3)),
      y: Number(y.toFixed(3)),
      z: Number(z.toFixed(3)),
      intensity: 0.7,
      semantic_class: 2,
      class_name: 'static_obstacle',
      confidence: 0.96,
    });
  }

  // Moving Vehicles
  for (let i = 0; i < pointCount * 0.05; i++) {
    const x = 2.4 + (prng() - 0.5) * 1.8;
    const y = 14.0 + (prng() - 0.5) * 4.0;
    const z = -1.5 + prng() * 1.5;
    points.push({
      x: Number(x.toFixed(3)),
      y: Number(y.toFixed(3)),
      z: Number(z.toFixed(3)),
      intensity: 0.95,
      semantic_class: 3,
      class_name: 'dynamic_object',
      confidence: 0.95,
    });
  }

  // Create initial 2.5D cells
  const cellMap = new Map<string, { x: number; y: number; z: number; cls: number; res: number; zone: number }>();
  for (const p of points) {
    const d = Math.sqrt(p.x * p.x + p.y * p.y);
    let res = 0.05;
    let zone = 0;
    if (d <= 10) {
      res = 0.05;
      zone = 0;
    } else if (d <= 40) {
      res = 0.15;
      zone = 1;
    } else {
      res = 0.5;
      zone = 2;
    }
    const cx = Number((Math.floor(p.x / res) * res + res / 2).toFixed(3));
    const cy = Number((Math.floor(p.y / res) * res + res / 2).toFixed(3));
    const key = `${cx}_${cy}_${zone}`;
    if (!cellMap.has(key)) {
      cellMap.set(key, { x: cx, y: cy, z: p.z, cls: p.semantic_class, res, zone });
    }
  }

  cellMap.forEach((item) => {
    cells.push({
      x: item.x,
      y: item.y,
      elevation: item.z,
      resolution: item.res,
      zone_id: item.zone,
      semantic_class: item.cls,
      class_name:
        item.cls === 0
          ? 'drivable_terrain'
          : item.cls === 1
          ? 'non_drivable_terrain'
          : item.cls === 2
          ? 'static_obstacle'
          : 'dynamic_object',
      confidence: 0.95,
      point_count: 5,
      traversability: item.cls === 0 ? 1.0 : item.cls === 1 ? 0.35 : 0.0,
      roughness: 0.04,
    });
  });

  const boundingBoxes: BoundingBox3D[] = [
    {
      id: 'dyn_veh_01',
      class_name: 'dynamic_object (Ahead Vehicle)',
      confidence: 0.96,
      center: [2.4, 14.0, -0.7],
      size: [1.8, 4.4, 1.6],
      rotation_yaw: 0.0,
    },
    {
      id: 'dyn_veh_02',
      class_name: 'dynamic_object (Oncoming Car)',
      confidence: 0.94,
      center: [-2.4, 38.0, -0.75],
      size: [1.8, 4.2, 1.5],
      rotation_yaw: Math.PI,
    },
  ];

  return { points, cells, boundingBoxes };
}

const initialScene = createInitialScene();

interface LidarState {
  isConnected: boolean;
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

  activePipelineStage: PipelineStageId;
  viewMode3D: ViewMode3D;
  cameraPreset: CameraViewPreset;
  colorMode: ColorMode;
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
  setCameraPreset: (preset: CameraViewPreset) => void;
  setColorMode: (mode: ColorMode) => void;
  toggleLayer: (layer: keyof LayerVisibility) => void;
  setLayer: (layer: keyof LayerVisibility, value: boolean) => void;
  setPointSize: (size: number) => void;
  setGridOpacity: (opacity: number) => void;
  setSelectedCell: (cell: FoveatedCell | null) => void;
  setSelectedBox: (box: BoundingBox3D | null) => void;
  setHoveredCell: (cell: FoveatedCell | null) => void;
  setIsComparisonOpen: (open: boolean) => void;
  setIsSettingsOpen: (open: boolean) => void;
  setIsPresentationMode: (mode: boolean) => void;
  setPresentationStep: (step: number) => void;
  applyPipelinePreset: (stage: PipelineStageId) => void;
  applyPresentationStep: (step: number) => void;
}

export const useLidarStore = create<LidarState>((set, get) => ({
  isConnected: false,
  datasets: [],
  activeDatasetId: 'urban_driving_demo_01',
  totalFrames: 100,
  currentFrameIdx: 0,
  playbackState: 'running',
  targetFps: 10,

  metadata: {
    frame_id: 0,
    timestamp_ms: 1724544000000,
    total_points: initialScene.points.length,
    sequence_id: 'urban_driving_demo_01',
    bounding_boxes: initialScene.boundingBoxes,
  },
  points: initialScene.points,
  cells: initialScene.cells,
  boundingBoxes: initialScene.boundingBoxes,
  metrics: {
    fps: 10.0,
    total_latency_ms: 30.3,
    ai_latency_ms: 18.2,
    grid_latency_ms: 12.1,
    memory_ram_mb: 135,
    memory_vram_mb: 215.5,
    cpu_percent: 18.5,
    raw_point_count: initialScene.points.length,
    cell_count: initialScene.cells.length,
    compression_ratio_percent: 82.8,
  },
  benchmark: null,

  activePipelineStage: 'foveation',
  viewMode3D: 'foveated_semantic',
  cameraPreset: 'perspective',
  colorMode: 'semantic',
  layers: INITIAL_LAYERS,
  pointSize: 3.5,
  gridOpacity: 0.85,
  elevationExaggeration: 1.0,

  isPresentationMode: false,
  presentationStep: 1,

  selectedCell: null,
  selectedBox: null,
  hoveredCell: null,
  isComparisonOpen: false,
  isSettingsOpen: false,

  setIsConnected: (connected) => set({ isConnected: connected }),
  setDatasets: (datasets) => set({ datasets }),
  setActiveDatasetId: (id) => set({ activeDatasetId: id }),
  setPlaybackState: (state) => set({ playbackState: state }),
  setTargetFps: (fps) => set({ targetFps: fps }),
  setCurrentFrameIdx: (idx) => set({ currentFrameIdx: idx }),

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

  setViewMode3D: (mode: ViewMode3D) => {
    const currentLayers = get().layers;
    switch (mode) {
      case 'raw':
        set({
          viewMode3D: mode,
          colorMode: 'intensity',
          activePipelineStage: 'raw',
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
          colorMode: 'elevation',
          activePipelineStage: 'elevation_25d',
          layers: {
            ...currentLayers,
            rawPoints: false,
            semanticPoints: false,
            foveatedGrid: true,
            adaptiveGridWireframe: true,
            zoneRings: true,
            boundingBoxes: false,
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
