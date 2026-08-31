'use client';

import React, { useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import { useLidarStore } from '@/stores/useLidarStore';
import { PointCloudViewer } from './PointCloudViewer';
import { FoveatedGridMesh } from './FoveatedGridMesh';
import { ConcentricRings } from './ConcentricRings';
import { BoundingBoxOverlay } from './BoundingBoxOverlay';
import { EgoVehicleMarker } from './EgoVehicleMarker';
import { CoordinateGizmo } from './CoordinateGizmo';
import { CompactLegends } from './CompactLegends';
import { CellInspectorTooltip } from './CellInspectorTooltip';
import {
  Camera,
  RotateCcw,
  Layers,
  Compass,
  Maximize2,
  Box,
  Eye,
} from 'lucide-react';
import { CAMERA_POSITIONS, DEFAULT_CAMERA_POSITION, CAMERA_TARGET } from '@/lib/constants';
import { ViewMode3D, CameraViewPreset } from '@/types/lidar';

export function LidarCanvas() {
  const viewMode3D = useLidarStore((state) => state.viewMode3D);
  const setViewMode3D = useLidarStore((state) => state.setViewMode3D);
  const cameraPreset = useLidarStore((state) => state.cameraPreset);
  const setCameraPreset = useLidarStore((state) => state.setCameraPreset);
  const layers = useLidarStore((state) => state.layers);
  const toggleLayer = useLidarStore((state) => state.toggleLayer);
  const metadata = useLidarStore((state) => state.metadata);
  const cells = useLidarStore((state) => state.cells);

  const controlsRef = useRef<any>(null);

  const setCameraView = (preset: CameraViewPreset) => {
    setCameraPreset(preset);
    if (controlsRef.current) {
      const config = CAMERA_POSITIONS[preset];
      if (controlsRef.current.object) {
        controlsRef.current.object.up.set(0, 0, 1);
        controlsRef.current.object.position.set(...config.pos);
      }
      controlsRef.current.target.set(...config.target);
      controlsRef.current.update();
    }
  };

  const resetCamera = () => {
    setCameraView('perspective');
  };

  const viewModes: { id: ViewMode3D; label: string; tag: string }[] = [
    { id: 'raw', label: '1. RAW LiDAR', tag: 'Intensity' },
    { id: 'semantic', label: '2. SEMANTIC', tag: 'AI' },
    { id: 'foveated', label: '3. FOVEATED', tag: 'Multi-Res' },
    { id: 'elevation', label: '4. ELEVATION', tag: 'Height' },
    { id: 'foveated_semantic', label: '5. FOV + SEMANTIC', tag: 'Default' },
    { id: 'foveated_elevation', label: '6. 2.5D ELEV MAP', tag: '2.5D' },
  ];

  return (
    <div className="relative w-full h-full bg-[#070A12] overflow-hidden select-none">
      {/* 3D WebGL Canvas */}
      <Canvas
        gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }}
        camera={{ position: DEFAULT_CAMERA_POSITION, fov: 45, near: 0.1, far: 500, up: [0, 0, 1] }}
      >
        {/* Scientific Studio Lighting */}
        <ambientLight intensity={0.9} />
        <directionalLight position={[30, -30, 50]} intensity={1.5} />
        <pointLight position={[0, 0, 20]} intensity={1.0} />

        {/* Camera Orbit Controls */}
        <OrbitControls
          ref={controlsRef}
          target={CAMERA_TARGET}
          maxDistance={300}
          minDistance={2}
          enableDamping
          dampingFactor={0.06}
        />

        {/* Reference Coordinate Ground Plane Grid */}
        <Grid
          position={[0, 35, -1.65]}
          args={[140, 140]}
          cellSize={5}
          cellThickness={0.5}
          cellColor="#172554"
          sectionSize={20}
          sectionThickness={1}
          sectionColor="#1E3A8A"
          fadeDistance={150}
          rotation={[-Math.PI / 2, 0, 0]}
        />

        {/* 3D Visualized Sub-Components */}
        <PointCloudViewer />
        <FoveatedGridMesh />
        <ConcentricRings />
        <BoundingBoxOverlay />
        <EgoVehicleMarker />
        <CoordinateGizmo />
      </Canvas>

      {/* Top Floating View Mode & Camera Toolbar */}
      <div className="absolute top-3 left-3 right-3 z-20 flex flex-wrap items-center justify-between gap-2 pointer-events-none">
        {/* 6 View Mode Selector Buttons */}
        <div className="flex items-center gap-1 bg-[#0A0E18]/90 backdrop-blur-md border border-border-color p-1 rounded-xl shadow-2xl pointer-events-auto overflow-x-auto">
          {viewModes.map((mode) => (
            <button
              key={mode.id}
              onClick={() => setViewMode3D(mode.id)}
              className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-mono font-bold tracking-tight transition-all ${
                viewMode3D === mode.id
                  ? 'bg-sky-600 text-white shadow-md shadow-sky-900/40 border border-sky-400/40'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-surface-highlight/60'
              }`}
            >
              <span>{mode.label}</span>
              <span
                className={`text-[9px] px-1 py-0.2 rounded uppercase ${
                  viewMode3D === mode.id
                    ? 'bg-sky-800 text-sky-200'
                    : 'bg-surface-highlight text-gray-400'
                }`}
              >
                {mode.tag}
              </span>
            </button>
          ))}
        </div>

        {/* Camera Preset Angles & Helper Toggles */}
        <div className="flex items-center gap-1 bg-[#0A0E18]/90 backdrop-blur-md border border-border-color p-1 rounded-xl shadow-2xl pointer-events-auto">
          <button
            onClick={() => setCameraView('perspective')}
            className={`px-2 py-1 rounded text-xs font-mono font-bold transition-colors ${
              cameraPreset === 'perspective'
                ? 'bg-brand-600 text-white'
                : 'text-gray-400 hover:text-white'
            }`}
            title="Perspective 3D Orbit View"
          >
            3D ORBIT
          </button>
          <button
            onClick={() => setCameraView('top')}
            className={`px-2 py-1 rounded text-xs font-mono font-bold transition-colors ${
              cameraPreset === 'top'
                ? 'bg-brand-600 text-white'
                : 'text-gray-400 hover:text-white'
            }`}
            title="Top-Down Bird's Eye View (BEV)"
          >
            TOP (BEV)
          </button>
          <button
            onClick={() => setCameraView('front')}
            className={`px-2 py-1 rounded text-xs font-mono font-bold transition-colors ${
              cameraPreset === 'front'
                ? 'bg-brand-600 text-white'
                : 'text-gray-400 hover:text-white'
            }`}
            title="Front Windshield View"
          >
            FRONT
          </button>
          <button
            onClick={() => setCameraView('side')}
            className={`px-2 py-1 rounded text-xs font-mono font-bold transition-colors ${
              cameraPreset === 'side'
                ? 'bg-brand-600 text-white'
                : 'text-gray-400 hover:text-white'
            }`}
            title="Side Cross-Section Elevation View"
          >
            SIDE
          </button>

          <div className="w-[1px] h-4 bg-border-color mx-1" />

          {/* Reset Camera Button */}
          <button
            onClick={resetCamera}
            className="p-1 rounded text-gray-400 hover:text-white hover:bg-surface-highlight transition-colors"
            title="Reset Camera View to Default"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Top-Left Live Status Badge */}
      <div className="absolute top-16 left-3 z-10 pointer-events-none flex flex-col gap-1 font-mono text-[11px]">
        <div className="bg-[#0A0E18]/80 backdrop-blur-md border border-border-color/80 px-2.5 py-1 rounded-lg text-gray-300 flex items-center gap-2 shadow-lg">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="font-bold text-white uppercase">
            {metadata ? `FRAME #${metadata.frame_id + 1}` : 'INITIALIZING'}
          </span>
          <span className="text-gray-500">|</span>
          <span className="text-sky-400">
            {metadata ? `${metadata.total_points.toLocaleString()} PTS` : '0 PTS'}
          </span>
          <span className="text-gray-500">|</span>
          <span className="text-amber-400">
            {`${cells.length.toLocaleString()} CELLS`}
          </span>
        </div>
      </div>

      {/* Contextual Floating Legends */}
      <CompactLegends />

      {/* Hover / Click Cell Inspector Tooltip */}
      <CellInspectorTooltip />
    </div>
  );
}
