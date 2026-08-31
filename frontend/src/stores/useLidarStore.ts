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
} from '@/types/lidar';
import { INITIAL_LAYERS } from '@/lib/constants';

// Deterministic Pseudo-Random Number Generator (LCG) to eliminate React Hydration Mismatches
let seed = 42;
function prng() {
  seed = (seed * 9301 + 49297) % 233280;
  return seed / 233280;
}

// Generates a true, structured 3-Zone Variable-Resolution Grid Map + underlying classified points
function createStructuredVariableGridScene() {
  seed = 1337; // Deterministic seed
  const points: SemanticPoint[] = [];
  const cells: FoveatedCell[] = [];
  const cellMap = new Map<string, FoveatedCell>();

  let cellCounter = 0;

  // Helper to add classified point and aggregate into the corresponding variable-resolution grid cell
  function addPoint(x: number, y: number, z: number, semanticClass: number, confidence: number = 0.95) {
    const dist = Math.sqrt(x * x + y * y);
    if (dist > 100.0) return; // 100m sensor range cut-off

    let cellSize = 0.05;
    let zoneId = 0;
    let zoneName = 'ZONE 0 — FOVEAL (0–10m @ 5cm)';

    if (dist <= 10.0) {
      cellSize = 0.05; // 5 cm near-field
      zoneId = 0;
      zoneName = 'ZONE 0 — FOVEAL (0–10m @ 5cm)';
    } else if (dist <= 50.0) {
      cellSize = 0.25; // 25 cm intermediate
      zoneId = 1;
      zoneName = 'ZONE 1 — INTERMEDIATE (10–50m @ ~25cm)';
    } else {
      cellSize = 0.50; // 50 cm peripheral
      zoneId = 2;
      zoneName = 'ZONE 2 — PERIPHERAL (50–100m @ 50cm)';
    }

    const className =
      semanticClass === 0
        ? 'Drivable Terrain'
        : semanticClass === 1
        ? 'Non-Drivable Terrain'
        : semanticClass === 2
        ? 'Static Obstacle'
        : semanticClass === 3
        ? 'Dynamic Object'
        : semanticClass === 4
        ? 'Vegetation'
        : 'Unknown / Background';

    // Push point
    points.push({
      x: Number(x.toFixed(3)),
      y: Number(y.toFixed(3)),
      z: Number(z.toFixed(3)),
      intensity: 0.85,
      semantic_class: semanticClass,
      class_name: className,
      confidence,
    });

    // Compute cell center quantized to cellSize grid
    const cx = Number((Math.floor(x / cellSize) * cellSize + cellSize / 2.0).toFixed(3));
    const cy = Number((Math.floor(y / cellSize) * cellSize + cellSize / 2.0).toFixed(3));
    const key = `${cx}_${cy}_${zoneId}`;

    if (!cellMap.has(key)) {
      cellCounter++;
      const cellId = `G${zoneId}_${String(cellCounter).padStart(5, '0')}`;
      cellMap.set(key, {
        id: cellId,
        x: cx,
        y: cy,
        elevation: z,
        resolution: cellSize,
        cellSize: cellSize,
        zone_id: zoneId,
        zone_name: zoneName,
        semantic_class: semanticClass,
        class_name: className,
        confidence,
        point_count: 1,
        sourcePointCount: 1,
        traversability: semanticClass === 0 ? 1.0 : semanticClass === 1 ? 0.35 : semanticClass === 4 ? 0.1 : 0.0,
        roughness: 0.03,
        occupied: true,
      });
    } else {
      const existing = cellMap.get(key)!;
      existing.point_count += 1;
      existing.sourcePointCount += 1;
      // Running average of elevation
      existing.elevation = Number(((existing.elevation * (existing.point_count - 1) + z) / existing.point_count).toFixed(3));
      // Dominant class priority
      if (semanticClass > existing.semantic_class) {
        existing.semantic_class = semanticClass;
        existing.class_name = className;
        existing.traversability = semanticClass === 0 ? 1.0 : semanticClass === 1 ? 0.35 : 0.0;
      }
    }
  }

  // 1. Urban Arterial Highway: North-South Main Road (Drivable, cls: 0, Z = -1.60m)
  for (let y = -95; y <= 95; y += 0.04) {
    for (let x = -4.5; x <= 4.5; x += 0.04) {
      if (prng() > 0.03) continue; // Sample density
      const z = -1.60 + (prng() - 0.5) * 0.02;
      addPoint(x, y, z, 0, 0.98);
    }
  }

  // 2. East-West Crossing Intersection (Drivable, cls: 0, Z = -1.60m)
  for (let x = -85; x <= 85; x += 0.06) {
    for (let y = -4.5; y <= 4.5; y += 0.06) {
      if (prng() > 0.04) continue;
      const z = -1.60 + (prng() - 0.5) * 0.02;
      addPoint(x, y, z, 0, 0.97);
    }
  }

  // 3. Sidewalks, Curbs & Crosswalks (Non-Drivable, cls: 1, Z = -1.45m elevated curb)
  for (let y = -90; y <= 90; y += 0.06) {
    // East Sidewalk (X: 4.8m to 8.0m)
    for (let x = 4.8; x <= 8.0; x += 0.08) {
      if (Math.abs(y) < 5.0 || prng() > 0.05) continue; // Skip intersection center
      const z = -1.45 + (prng() - 0.5) * 0.03;
      addPoint(x, y, z, 1, 0.95);
    }
    // West Sidewalk (X: -8.0m to -4.8m)
    for (let x = -8.0; x <= -4.8; x += 0.08) {
      if (Math.abs(y) < 5.0 || prng() > 0.05) continue;
      const z = -1.45 + (prng() - 0.5) * 0.03;
      addPoint(x, y, z, 1, 0.95);
    }
  }

  // 4. Urban Vegetation & Roadside Trees (cls: 4, Z = -1.4m to +2.5m)
  const treePositions = [
    { x: 9.0, y: 15.0 },
    { x: 9.0, y: 32.0 },
    { x: 9.0, y: -20.0 },
    { x: -9.0, y: 18.0 },
    { x: -9.0, y: -25.0 },
    { x: 25.0, y: 8.0 },
    { x: -28.0, y: 8.0 },
  ];
  for (const tree of treePositions) {
    for (let i = 0; i < 90; i++) {
      const rx = tree.x + (prng() - 0.5) * 2.2;
      const ry = tree.y + (prng() - 0.5) * 2.2;
      const rz = -1.4 + prng() * 3.8;
      addPoint(rx, ry, rz, 4, 0.92);
    }
  }

  // 5. Static Buildings, Walls & Obstacles in 4 Quadrants (cls: 2, Z = -1.5m to +4.0m)
  const buildingBoxes = [
    { x1: 12, x2: 45, y1: 10, y2: 60, height: 4.2 }, // NE Building Block
    { x1: -45, x2: -12, y1: 10, y2: 60, height: 3.8 }, // NW Commercial Block
    { x1: 12, x2: 45, y1: -60, y2: -10, height: 3.6 }, // SE Plaza
    { x1: -45, x2: -12, y1: -60, y2: -10, height: 4.5 }, // SW Tower
  ];
  for (const b of buildingBoxes) {
    for (let x = b.x1; x <= b.x2; x += 0.8) {
      for (let y = b.y1; y <= b.y2; y += 0.8) {
        if (prng() > 0.08) continue;
        const z = -1.5 + prng() * b.height;
        addPoint(x, y, z, 2, 0.96);
      }
    }
  }

  // 6. Dynamic Objects: Moving Ahead Vehicle (cls: 3, center: [2.2, 14.0], height: 1.6m)
  for (let i = 0; i < 220; i++) {
    const x = 2.2 + (prng() - 0.5) * 1.8;
    const y = 14.0 + (prng() - 0.5) * 4.4;
    const z = -1.55 + prng() * 1.6;
    addPoint(x, y, z, 3, 0.97);
  }

  // 7. Dynamic Objects: Oncoming Vehicle (cls: 3, center: [-2.2, 38.0], height: 1.5m)
  for (let i = 0; i < 180; i++) {
    const x = -2.2 + (prng() - 0.5) * 1.8;
    const y = 38.0 + (prng() - 0.5) * 4.2;
    const z = -1.55 + prng() * 1.5;
    addPoint(x, y, z, 3, 0.95);
  }

  // 8. Dynamic Objects: Rear Trailing Vehicle (cls: 3, center: [-2.2, -18.0])
  for (let i = 0; i < 160; i++) {
    const x = -2.2 + (prng() - 0.5) * 1.8;
    const y = -18.0 + (prng() - 0.5) * 4.2;
    const z = -1.55 + prng() * 1.5;
    addPoint(x, y, z, 3, 0.94);
  }

  // Collect final cells array
  cellMap.forEach((c) => cells.push(c));

  const boundingBoxes: BoundingBox3D[] = [
    {
      id: 'dyn_veh_01',
      class_name: 'Dynamic Object (Ahead Vehicle)',
      confidence: 0.97,
      center: [2.2, 14.0, -0.75],
      size: [1.8, 4.4, 1.6],
      rotation_yaw: 0.0,
    },
    {
      id: 'dyn_veh_02',
      class_name: 'Dynamic Object (Oncoming Car)',
      confidence: 0.95,
      center: [-2.2, 38.0, -0.8],
      size: [1.8, 4.2, 1.5],
      rotation_yaw: Math.PI,
    },
    {
      id: 'dyn_veh_03',
      class_name: 'Dynamic Object (Rear Vehicle)',
      confidence: 0.94,
      center: [-2.2, -18.0, -0.8],
      size: [1.8, 4.2, 1.5],
      rotation_yaw: 0.0,
    },
  ];

  return { points, cells, boundingBoxes };
}

const initialScene = createStructuredVariableGridScene();

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

  // Default to 2.5D Elevation Map + Top-Down BEV + Pure Grid Map mode as requested
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
