'use client';

import React from 'react';
import { useLidarStore } from '@/stores/useLidarStore';
import { Radio } from 'lucide-react';
import { GridDisplayMode, GridRenderStyle } from '@/types/lidar';

export function Header() {
  const isConnected = useLidarStore((state) => state.isConnected);
  const gridDisplayMode = useLidarStore((state) => state.gridDisplayMode);
  const setGridDisplayMode = useLidarStore((state) => state.setGridDisplayMode);
  const gridRenderStyle = useLidarStore((state) => state.gridRenderStyle);
  const setGridRenderStyle = useLidarStore((state) => state.setGridRenderStyle);
  const setViewMode3D = useLidarStore((state) => state.setViewMode3D);

  // Active view switcher mode
  const currentMode =
    gridDisplayMode === 'points'
      ? 'point_cloud'
      : gridRenderStyle === 'extruded_3d'
      ? '3d_elevation'
      : '25d_grid';

  const handleModeChange = (mode: 'point_cloud' | '25d_grid' | '3d_elevation') => {
    if (mode === 'point_cloud') {
      setGridDisplayMode('points');
      setGridRenderStyle('top_down_2d');
      setViewMode3D('semantic');
    } else if (mode === '25d_grid') {
      setGridDisplayMode('grid');
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
            FOVEATED LiDAR 2.5D MAPPING
          </h1>
          <p className="text-[10px] text-gray-400 hidden md:block">
            Variable-Resolution Perception &amp; Semantic Elevation Map Engine
          </p>
        </div>
      </div>

      {/* 2. Center Pill Group: [ POINT CLOUD ] [ 2.5D GRID MAP ] [ 3D ELEVATION ] */}
      <div className="flex items-center bg-[#0F172A] border border-[#1E293B] p-1 rounded-xl shadow-lg">
        <button
          onClick={() => handleModeChange('point_cloud')}
          className={`px-3.5 py-1.5 rounded-lg text-xs font-bold transition-all ${
            currentMode === 'point_cloud'
              ? 'bg-[#6366F1] text-white shadow-md shadow-indigo-900/50'
              : 'text-gray-400 hover:text-gray-200'
          }`}
        >
          POINT CLOUD
        </button>

        <button
          onClick={() => handleModeChange('25d_grid')}
          className={`px-3.5 py-1.5 rounded-lg text-xs font-bold transition-all ${
            currentMode === '25d_grid'
              ? 'bg-[#6366F1] text-white shadow-md shadow-indigo-900/50'
              : 'text-gray-400 hover:text-gray-200'
          }`}
        >
          2.5D GRID MAP
        </button>

        <button
          onClick={() => handleModeChange('3d_elevation')}
          className={`px-3.5 py-1.5 rounded-lg text-xs font-bold transition-all ${
            currentMode === '3d_elevation'
              ? 'bg-[#6366F1] text-white shadow-md shadow-indigo-900/50'
              : 'text-gray-400 hover:text-gray-200'
          }`}
        >
          3D ELEVATION
        </button>
      </div>

      {/* 3. Right Status Badge: SYSTEM STATUS [ LIVE • ] */}
      <div className="flex items-center gap-2">
        <span className="text-[11px] text-gray-400 font-bold hidden sm:inline">
          SYSTEM STATUS
        </span>
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px] font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
          <span>LIVE</span>
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
        </div>
      </div>
    </header>
  );
}
