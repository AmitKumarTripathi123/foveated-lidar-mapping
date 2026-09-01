'use client';

import React from 'react';
import { useLidarStore } from '@/stores/useLidarStore';
import { Radio, Layers } from 'lucide-react';
import { GridDisplayMode, GridRenderStyle } from '@/types/lidar';

export function Header() {
  const isConnected = useLidarStore((state) => state.isConnected);
  const connectionState = useLidarStore((state) => state.connectionState);
  const gridDisplayMode = useLidarStore((state) => state.gridDisplayMode);
  const setGridDisplayMode = useLidarStore((state) => state.setGridDisplayMode);
  const gridRenderStyle = useLidarStore((state) => state.gridRenderStyle);
  const setGridRenderStyle = useLidarStore((state) => state.setGridRenderStyle);
  const setViewMode3D = useLidarStore((state) => state.setViewMode3D);

  // Active view switcher mode
  const currentMode =
    gridDisplayMode === 'points'
      ? 'point_cloud'
      : gridDisplayMode === 'both'
      ? 'both'
      : gridRenderStyle === 'extruded_3d'
      ? '3d_elevation'
      : '25d_grid';

  const handleModeChange = (mode: 'point_cloud' | '25d_grid' | 'both' | '3d_elevation') => {
    if (mode === 'point_cloud') {
      setGridDisplayMode('points');
      setGridRenderStyle('top_down_2d');
      setViewMode3D('semantic');
    } else if (mode === '25d_grid') {
      setGridDisplayMode('grid');
      setGridRenderStyle('top_down_2d');
      setViewMode3D('foveated_elevation');
    } else if (mode === 'both') {
      setGridDisplayMode('both');
      setGridRenderStyle('top_down_2d');
      setViewMode3D('foveated_elevation');
    } else if (mode === '3d_elevation') {
      setGridDisplayMode('grid');
      setGridRenderStyle('extruded_3d');
      setViewMode3D('foveated_elevation');
    }
  };

  return (
    <header className="h-14 bg-[#080B14] border-b border-[#1E293B] px-5 flex items-center justify-between text-white select-none z-30 font-mono">
      {/* 1. Left Brand Title */}
      <div className="flex items-center gap-3">
        <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-sky-500/20 text-sky-400 border border-sky-500/40">
          <Radio className="w-4 h-4" />
        </div>
        <div>
          <h1 className="text-xs sm:text-sm font-bold tracking-wider text-white">
            ADAPTIVE VARIABLE-RESOLUTION 2.5D LiDAR MAPPING
          </h1>
          <p className="text-[10px] text-gray-400 hidden md:block">
            Variable-Resolution 2.5D Grid Engine for Dynamic Environment Perception
          </p>
        </div>
      </div>

      {/* 2. Center Pill Group: [ POINT CLOUD ] [ 2.5D GRID MAP ] [ BOTH (AGGREGATION) ] [ 3D ELEVATION ] */}
      <div className="flex items-center bg-[#0F172A] border border-[#1E293B] p-1 rounded-xl shadow-lg gap-1">
        <button
          onClick={() => handleModeChange('point_cloud')}
          className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
            currentMode === 'point_cloud'
              ? 'bg-[#6366F1] text-white shadow-md shadow-indigo-900/50'
              : 'text-gray-400 hover:text-gray-200'
          }`}
          title="Raw & Semantic 3D Point Cloud"
        >
          POINT CLOUD
        </button>

        <button
          onClick={() => handleModeChange('25d_grid')}
          className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
            currentMode === '25d_grid'
              ? 'bg-[#6366F1] text-white shadow-md shadow-indigo-900/50'
              : 'text-gray-400 hover:text-gray-200'
          }`}
          title="Variable-Resolution 2.5D Elevation Grid"
        >
          2.5D GRID MAP
        </button>

        <button
          onClick={() => handleModeChange('both')}
          className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
            currentMode === 'both'
              ? 'bg-[#6366F1] text-white shadow-md shadow-indigo-900/50'
              : 'text-gray-400 hover:text-gray-200'
          }`}
          title="Demonstrate points aggregated into 2.5D spatial cells"
        >
          BOTH (PROOF)
        </button>

        <button
          onClick={() => handleModeChange('3d_elevation')}
          className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
            currentMode === '3d_elevation'
              ? 'bg-[#6366F1] text-white shadow-md shadow-indigo-900/50'
              : 'text-gray-400 hover:text-gray-200'
          }`}
          title="3D Extruded Surface Columns"
        >
          3D ELEVATION
        </button>
      </div>

      {/* 3. Right Status Indicator */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 px-3 py-1 rounded-lg bg-surface-highlight/50 border border-border-color text-xs">
          <span
            className={`w-2 h-2 rounded-full ${
              connectionState === 'connected'
                ? 'bg-[#22C55E] animate-pulse'
                : connectionState === 'connecting'
                ? 'bg-[#EAB308] animate-ping'
                : connectionState === 'reconnecting'
                ? 'bg-[#F59E0B] animate-pulse'
                : 'bg-[#38BDF8] animate-pulse'
            }`}
          />
          <span
            className={`font-medium ${
              connectionState === 'connected'
                ? 'text-emerald-300'
                : connectionState === 'connecting'
                ? 'text-yellow-300'
                : connectionState === 'reconnecting'
                ? 'text-amber-300'
                : 'text-sky-300'
            }`}
          >
            {connectionState === 'connected'
              ? 'LIVE BACKEND (10 Hz)'
              : connectionState === 'connecting'
              ? 'CONNECTING...'
              : connectionState === 'reconnecting'
              ? 'RECONNECTING...'
              : 'SIMULATION STREAM (10 Hz)'}
          </span>
        </div>
      </div>
    </header>
  );
}
