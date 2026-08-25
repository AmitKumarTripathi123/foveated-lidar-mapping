'use client';

import React from 'react';
import { useLidarStore } from '@/stores/useLidarStore';
import {
  Layers,
  Palette,
  Box,
  CircleDot,
  Mountain,
  Grid,
  Car,
  Compass,
} from 'lucide-react';
import { ColorMode } from '@/types/lidar';

export function LayerSelector() {
  const layers = useLidarStore((state) => state.layers);
  const toggleLayer = useLidarStore((state) => state.toggleLayer);
  const colorMode = useLidarStore((state) => state.colorMode);
  const setColorMode = useLidarStore((state) => state.setColorMode);
  const pointSize = useLidarStore((state) => state.pointSize);
  const setPointSize = useLidarStore((state) => state.setPointSize);
  const gridOpacity = useLidarStore((state) => state.gridOpacity);
  const setGridOpacity = useLidarStore((state) => state.setGridOpacity);

  const colorModes: { id: ColorMode; label: string }[] = [
    { id: 'semantic', label: 'Semantic' },
    { id: 'elevation', label: 'Elevation' },
    { id: 'traversability', label: 'Traversability' },
    { id: 'intensity', label: 'Intensity' },
  ];

  return (
    <div className="bg-[#0B0F19]/90 backdrop-blur-md border border-border-color rounded-xl p-3 shadow-lg flex flex-col gap-3 text-white text-xs font-mono">
      {/* Colormap Switcher */}
      <div className="flex flex-col gap-1.5">
        <div className="flex items-center gap-1.5 text-gray-400">
          <Palette className="w-3.5 h-3.5 text-sky-400" />
          <span className="font-bold tracking-wide">Colormap Mode:</span>
        </div>
        <div className="grid grid-cols-2 gap-1 bg-surface-highlight/60 p-1 rounded-lg border border-border-color/60">
          {colorModes.map((mode) => (
            <button
              key={mode.id}
              onClick={() => setColorMode(mode.id)}
              className={`py-1 rounded text-[10px] font-bold transition-all ${
                colorMode === mode.id
                  ? 'bg-sky-600 text-white shadow-sm'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              {mode.label}
            </button>
          ))}
        </div>
      </div>

      {/* Layer Visibility Toggles */}
      <div className="flex flex-col gap-1.5">
        <div className="flex items-center gap-1.5 text-gray-400">
          <Layers className="w-3.5 h-3.5 text-sky-400" />
          <span className="font-bold tracking-wide">Visualization Layers:</span>
        </div>

        <div className="flex flex-col gap-1">
          {/* Foveated 2.5D Grid */}
          <label className="flex items-center justify-between p-1.5 rounded-lg bg-surface-highlight/40 hover:bg-surface-highlight cursor-pointer transition-colors">
            <div className="flex items-center gap-2">
              <Mountain className="w-3.5 h-3.5 text-sky-400" />
              <span>Foveated 2.5D Elevation</span>
            </div>
            <input
              type="checkbox"
              checked={layers.foveatedGrid}
              onChange={() => toggleLayer('foveatedGrid')}
              className="w-4 h-4 rounded bg-surface border-gray-600 text-sky-600 focus:ring-0 cursor-pointer"
            />
          </label>

          {/* Adaptive Grid Wireframe */}
          <label className="flex items-center justify-between p-1.5 rounded-lg bg-surface-highlight/40 hover:bg-surface-highlight cursor-pointer transition-colors">
            <div className="flex items-center gap-2">
              <Grid className="w-3.5 h-3.5 text-sky-300" />
              <span>Adaptive Grid Density</span>
            </div>
            <input
              type="checkbox"
              checked={layers.adaptiveGridWireframe}
              onChange={() => toggleLayer('adaptiveGridWireframe')}
              className="w-4 h-4 rounded bg-surface border-gray-600 text-sky-600 focus:ring-0 cursor-pointer"
            />
          </label>

          {/* Semantic Points */}
          <label className="flex items-center justify-between p-1.5 rounded-lg bg-surface-highlight/40 hover:bg-surface-highlight cursor-pointer transition-colors">
            <div className="flex items-center gap-2">
              <CircleDot className="w-3.5 h-3.5 text-emerald-400" />
              <span>Semantic Point Cloud</span>
            </div>
            <input
              type="checkbox"
              checked={layers.semanticPoints}
              onChange={() => toggleLayer('semanticPoints')}
              className="w-4 h-4 rounded bg-surface border-gray-600 text-sky-600 focus:ring-0 cursor-pointer"
            />
          </label>

          {/* 3D Bounding Boxes */}
          <label className="flex items-center justify-between p-1.5 rounded-lg bg-surface-highlight/40 hover:bg-surface-highlight cursor-pointer transition-colors">
            <div className="flex items-center gap-2">
              <Box className="w-3.5 h-3.5 text-amber-400" />
              <span>3D Object Bounding Boxes</span>
            </div>
            <input
              type="checkbox"
              checked={layers.boundingBoxes}
              onChange={() => toggleLayer('boundingBoxes')}
              className="w-4 h-4 rounded bg-surface border-gray-600 text-sky-600 focus:ring-0 cursor-pointer"
            />
          </label>

          {/* Foveated Zone Rings */}
          <label className="flex items-center justify-between p-1.5 rounded-lg bg-surface-highlight/40 hover:bg-surface-highlight cursor-pointer transition-colors">
            <div className="flex items-center gap-2">
              <CircleDot className="w-3.5 h-3.5 text-purple-400" />
              <span>Resolution Zone Rings</span>
            </div>
            <input
              type="checkbox"
              checked={layers.zoneRings}
              onChange={() => toggleLayer('zoneRings')}
              className="w-4 h-4 rounded bg-surface border-gray-600 text-sky-600 focus:ring-0 cursor-pointer"
            />
          </label>

          {/* Ego Vehicle Footprint */}
          <label className="flex items-center justify-between p-1.5 rounded-lg bg-surface-highlight/40 hover:bg-surface-highlight cursor-pointer transition-colors">
            <div className="flex items-center gap-2">
              <Car className="w-3.5 h-3.5 text-sky-400" />
              <span>Ego Sensor Position</span>
            </div>
            <input
              type="checkbox"
              checked={layers.egoVehicle}
              onChange={() => toggleLayer('egoVehicle')}
              className="w-4 h-4 rounded bg-surface border-gray-600 text-sky-600 focus:ring-0 cursor-pointer"
            />
          </label>
        </div>
      </div>

      {/* Point Size & Opacity Controls */}
      <div className="flex flex-col gap-2 pt-1 border-t border-border-color/60">
        <div className="flex flex-col gap-1">
          <div className="flex justify-between items-center text-[10px] text-gray-400">
            <span>Point Cloud Size:</span>
            <span className="text-gray-200">{pointSize}px</span>
          </div>
          <input
            type="range"
            min={1.0}
            max={8.0}
            step={0.5}
            value={pointSize}
            onChange={(e) => setPointSize(parseFloat(e.target.value))}
            className="w-full h-1 bg-surface-highlight rounded appearance-none cursor-pointer accent-sky-500"
          />
        </div>

        <div className="flex flex-col gap-1">
          <div className="flex justify-between items-center text-[10px] text-gray-400">
            <span>2.5D Grid Opacity:</span>
            <span className="text-gray-200">{Math.round(gridOpacity * 100)}%</span>
          </div>
          <input
            type="range"
            min={0.1}
            max={1.0}
            step={0.05}
            value={gridOpacity}
            onChange={(e) => setGridOpacity(parseFloat(e.target.value))}
            className="w-full h-1 bg-surface-highlight rounded appearance-none cursor-pointer accent-sky-500"
          />
        </div>
      </div>
    </div>
  );
}
