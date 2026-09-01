'use client';

import React, { useRef, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import { useLidarStore } from '@/stores/useLidarStore';
import { PointCloudViewer } from './PointCloudViewer';
import { FoveatedGridMesh } from './FoveatedGridMesh';
import { ConcentricRings } from './ConcentricRings';
import { BoundingBoxOverlay } from './BoundingBoxOverlay';
import { EgoVehicleMarker } from './EgoVehicleMarker';
import { CoordinateGizmo } from './CoordinateGizmo';
import { CellInspectorTooltip } from './CellInspectorTooltip';
import { CellDetailDrawer } from './CellDetailDrawer';
import {
  RotateCcw,
} from 'lucide-react';
import { CAMERA_POSITIONS } from '@/lib/constants';
import { CameraViewPreset } from '@/types/lidar';

export function LidarCanvas() {
  const cameraPreset = useLidarStore((state) => state.cameraPreset);
  const setCameraPreset = useLidarStore((state) => state.setCameraPreset);
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

  // Guarantee that initial page mount starts in Top-Down Bird's Eye View (BEV)
  useEffect(() => {
    setCameraView('top');
  }, []);

  const resetCamera = () => {
    setCameraView('top');
  };

  return (
    <div className="relative w-full h-full bg-[#070A12] overflow-hidden select-none font-mono">
      {/* 3D WebGL Canvas with standard Z-up orientation */}
      <Canvas
        gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }}
        camera={{ position: [0, 15, 95], fov: 45, near: 0.1, far: 500, up: [0, 0, 1] }}
      >
        {/* Scientific Studio Lighting */}
        <ambientLight intensity={0.9} />
        <directionalLight position={[30, -30, 50]} intensity={1.5} />
        <pointLight position={[0, 0, 20]} intensity={1.0} />

        {/* Camera Orbit Controls */}
        <OrbitControls
          ref={controlsRef}
          target={[0, 15, 0]}
          maxDistance={300}
          minDistance={2}
          enableDamping
          dampingFactor={0.06}
        />

        {/* Reference Coordinate Ground Plane Grid */}
        <Grid
          position={[0, 15, -1.65]}
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

      {/* Top-Left Title Overlay for SIH Jury */}
      <div className="absolute top-3 left-4 flex items-center gap-2.5 pointer-events-none z-20 font-mono">
        <div className="flex items-center justify-center w-6 h-6 rounded bg-sky-500/20 text-sky-400 border border-sky-500/40">
          <span className="text-xs">⤢</span>
        </div>
        <div>
          <div className="text-xs font-bold text-white tracking-wider flex items-center gap-2">
            <span>2.5D FOVEATED ELEVATION GRID MAP</span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-sky-500/20 text-sky-300 border border-sky-500/40">
              VARIABLE RESOLUTION
            </span>
          </div>
          <div className="text-[10px] text-sky-400 font-semibold tracking-wide">
            5 cm near-field (0–10m) → 25 cm mid-field (10–50m) → 50 cm far-field (50–100m)
          </div>
        </div>
      </div>

      {/* Top Floating Camera Presets Toolbar */}
      <div className="absolute top-3 left-1/2 -translate-x-1/2 z-20 flex items-center gap-1 bg-[#0A0E18]/90 backdrop-blur-md border border-border-color p-1 rounded-xl shadow-2xl">
        <button
          onClick={() => setCameraView('top')}
          className={`px-2.5 py-1 rounded text-xs font-mono font-bold transition-colors ${
            cameraPreset === 'top'
              ? 'bg-brand-600 text-white'
              : 'text-gray-400 hover:text-white'
          }`}
          title="Top-Down Bird's Eye View (BEV)"
        >
          TOP (BEV)
        </button>
        <button
          onClick={() => setCameraView('perspective')}
          className={`px-2.5 py-1 rounded text-xs font-mono font-bold transition-colors ${
            cameraPreset === 'perspective'
              ? 'bg-brand-600 text-white'
              : 'text-gray-400 hover:text-white'
          }`}
          title="Perspective 3D Orbit View"
        >
          3D ORBIT
        </button>
        <button
          onClick={() => setCameraView('front')}
          className={`px-2.5 py-1 rounded text-xs font-mono font-bold transition-colors ${
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
          className={`px-2.5 py-1 rounded text-xs font-mono font-bold transition-colors ${
            cameraPreset === 'side'
              ? 'bg-brand-600 text-white'
              : 'text-gray-400 hover:text-white'
          }`}
          title="Side Cross-Section Elevation View"
        >
          SIDE
        </button>

        <div className="w-[1px] h-4 bg-border-color mx-0.5" />

        {/* Reset Camera Button */}
        <button
          onClick={resetCamera}
          className="p-1 rounded text-gray-400 hover:text-white hover:bg-surface-highlight transition-colors"
          title="Reset to Top-Down View"
        >
          <RotateCcw className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Top-Right FOVEATED ZONES Legend Card */}
      <div className="absolute top-3 right-4 bg-[#0A0E18]/90 backdrop-blur-md border border-border-color rounded-xl p-3 shadow-2xl text-[11px] space-y-2 pointer-events-none z-20">
        <div className="text-[10px] font-bold text-gray-400 uppercase tracking-wider border-b border-border-color/60 pb-1">
          FOVEATED ZONES
        </div>
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-sm bg-[#0284C7] shrink-0" />
            <div>
              <div className="font-bold text-white text-[10px]">ZONE 0 — FOVEAL (NEAR)</div>
              <div className="text-[9px] text-gray-400">0–10m @ 0.05m (5cm) / cell</div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-sm bg-[#16A34A] shrink-0" />
            <div>
              <div className="font-bold text-white text-[10px]">ZONE 1 — INTERMEDIATE</div>
              <div className="text-[9px] text-gray-400">10–50m @ ~0.25m (25cm) / cell</div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-sm bg-[#F59E0B] shrink-0" />
            <div>
              <div className="font-bold text-white text-[10px]">ZONE 2 — PERIPHERAL</div>
              <div className="text-[9px] text-gray-400">50–100m @ 0.50m (50cm) / cell</div>
            </div>
          </div>
          <div className="flex items-center gap-2 pt-1 border-t border-border-color/40">
            <span className="w-4 border-b-2 border-dashed border-white/70 inline-block" />
            <span className="text-[9px] text-gray-300">FOV BOUNDARY</span>
          </div>
        </div>
      </div>

      {/* Bottom Horizontal Elevation Gradient Bar */}
      <div className="absolute bottom-3 left-1/2 -translate-x-1/2 bg-[#0A0E18]/90 backdrop-blur-md border border-border-color/80 px-4 py-1.5 rounded-xl shadow-xl flex items-center gap-3 pointer-events-none text-[10px] z-20">
        <span className="text-gray-400 font-bold">ELEVATION (Z)</span>
        <span className="text-sky-400 font-bold">LOW</span>
        <div className="w-80 h-3 rounded-full bg-gradient-to-r from-blue-700 via-cyan-400 via-green-500 via-yellow-400 via-orange-500 to-red-600 relative flex justify-between px-1.5 text-[8px] text-black font-bold items-center shadow-inner">
          <span>-2.0m</span>
          <span>-1.0m</span>
          <span>0m</span>
          <span>+1.0m</span>
          <span>+2.0m</span>
          <span>+3.0m</span>
        </div>
        <span className="text-red-400 font-bold">HIGH</span>
      </div>

      {/* Hover Cell Inspector Tooltip */}
      <CellInspectorTooltip />

      {/* Deep Click Inspection Drawer */}
      <CellDetailDrawer />
    </div>
  );
}
