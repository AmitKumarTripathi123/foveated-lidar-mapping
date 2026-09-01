'use client';

import React from 'react';
import { useLidarStore } from '@/stores/useLidarStore';
import { X, Layers, MapPin, Activity, Hash, ArrowUpDown, ShieldAlert, Sparkles } from 'lucide-react';
import { SEMANTIC_CLASSES, FOVEATED_ZONE_COLORS } from '@/lib/semanticColors';

export function CellDetailDrawer() {
  const selectedCell = useLidarStore((state) => state.selectedCell);
  const setSelectedCell = useLidarStore((state) => state.setSelectedCell);

  if (!selectedCell) return null;

  const semanticDef = SEMANTIC_CLASSES[selectedCell.semantic_class] || SEMANTIC_CLASSES[0];
  const zoneDef = FOVEATED_ZONE_COLORS[selectedCell.zone_id] || FOVEATED_ZONE_COLORS[0];
  const cellSizeM = selectedCell.cellSize || selectedCell.resolution || 0.05;

  return (
    <div className="absolute top-16 right-4 z-30 w-80 bg-[#0A0E18]/95 backdrop-blur-md border border-sky-500/60 rounded-2xl p-4 shadow-2xl text-white font-mono select-none animate-in fade-in slide-in-from-right-4 duration-150 flex flex-col gap-3">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border-color pb-2">
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-sky-400 animate-pulse" />
          <div>
            <div className="text-xs font-bold text-sky-300 tracking-wider">
              {selectedCell.id || '2.5D GRID CELL'}
            </div>
            <div className="text-[9px] text-gray-400">Deep Cell Telemetry Inspector</div>
          </div>
        </div>
        <button
          onClick={() => setSelectedCell(null)}
          className="p-1 rounded-lg text-gray-400 hover:text-white hover:bg-surface-highlight transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Primary 4 Metric Badges */}
      <div className="grid grid-cols-2 gap-2 text-xs">
        {/* XY Coordinates */}
        <div className="bg-surface-highlight/40 border border-border-color/50 rounded-xl p-2 flex flex-col">
          <span className="text-[9px] text-gray-400 uppercase font-bold flex items-center gap-1">
            <MapPin className="w-3 h-3 text-sky-400" />
            Grid Center (X, Y)
          </span>
          <span className="text-white font-bold mt-0.5 text-[11px]">
            {selectedCell.x >= 0 ? `+${selectedCell.x.toFixed(2)}` : selectedCell.x.toFixed(2)}m, {selectedCell.y >= 0 ? `+${selectedCell.y.toFixed(2)}` : selectedCell.y.toFixed(2)}m
          </span>
        </div>

        {/* Mean Elevation (Z) */}
        <div className="bg-surface-highlight/40 border border-border-color/50 rounded-xl p-2 flex flex-col">
          <span className="text-[9px] text-gray-400 uppercase font-bold flex items-center gap-1">
            <ArrowUpDown className="w-3 h-3 text-emerald-400" />
            Mean Elevation (Z)
          </span>
          <span className="text-emerald-300 font-bold mt-0.5 text-[11px]">
            {selectedCell.elevation >= 0 ? `+${selectedCell.elevation.toFixed(2)}` : selectedCell.elevation.toFixed(2)} m
          </span>
        </div>

        {/* Resolution / Cell Size */}
        <div className="bg-surface-highlight/40 border border-border-color/50 rounded-xl p-2 flex flex-col">
          <span className="text-[9px] text-gray-400 uppercase font-bold flex items-center gap-1">
            <Layers className="w-3 h-3 text-amber-400" />
            Cell Resolution
          </span>
          <span className="text-amber-300 font-bold mt-0.5 text-[11px]">
            {(cellSizeM * 100).toFixed(0)} cm ({(cellSizeM).toFixed(2)}m)
          </span>
        </div>

        {/* Source LiDAR Points */}
        <div className="bg-surface-highlight/40 border border-border-color/50 rounded-xl p-2 flex flex-col">
          <span className="text-[9px] text-gray-400 uppercase font-bold flex items-center gap-1">
            <Hash className="w-3 h-3 text-purple-400" />
            Source Points
          </span>
          <span className="text-purple-300 font-bold mt-0.5 text-[11px]">
            {selectedCell.sourcePointCount || selectedCell.point_count || 1} points
          </span>
        </div>
      </div>

      {/* Elevation Bounds & Surface Roughness */}
      <div className="bg-[#080C16] border border-border-color/80 rounded-xl p-2.5 space-y-1.5 text-[10px]">
        <div className="text-[9px] text-gray-400 uppercase font-bold border-b border-border-color/40 pb-1 flex items-center justify-between">
          <span>Elevation Bounds &amp; Micro-Terrain</span>
          <span className="text-emerald-400">2.5D GEOMETRY</span>
        </div>
        <div className="flex justify-between text-gray-300">
          <span>Min Elevation (Z_min):</span>
          <span className="font-bold text-white">
            {selectedCell.minElevation !== undefined ? `${selectedCell.minElevation.toFixed(2)} m` : `${(selectedCell.elevation - 0.05).toFixed(2)} m`}
          </span>
        </div>
        <div className="flex justify-between text-gray-300">
          <span>Max Elevation (Z_max):</span>
          <span className="font-bold text-white">
            {selectedCell.maxElevation !== undefined ? `${selectedCell.maxElevation.toFixed(2)} m` : `${(selectedCell.elevation + 0.05).toFixed(2)} m`}
          </span>
        </div>
        <div className="flex justify-between text-gray-300">
          <span>Surface Roughness (σ_z):</span>
          <span className="font-bold text-amber-300">
            {selectedCell.roughness !== undefined ? `${(selectedCell.roughness * 100).toFixed(1)} cm` : '2.1 cm'}
          </span>
        </div>
      </div>

      {/* Semantic Classification & Voting Card */}
      <div className="bg-[#080C16] border border-border-color/80 rounded-xl p-2.5 space-y-1.5 text-[10px]">
        <div className="flex items-center justify-between">
          <span className="text-gray-400 uppercase font-bold text-[9px]">Dominant Semantic:</span>
          <span
            className="px-2 py-0.5 rounded font-bold border text-[10px]"
            style={{
              color: semanticDef.hex,
              backgroundColor: `${semanticDef.hex}15`,
              borderColor: `${semanticDef.hex}40`,
            }}
          >
            {semanticDef.name}
          </span>
        </div>

        <div className="flex items-center justify-between text-gray-300">
          <span>AI Voting Confidence:</span>
          <span className="font-bold text-white">
            {((selectedCell.confidence || 0.95) * 100).toFixed(1)}%
          </span>
        </div>

        <div className="flex items-center justify-between text-gray-300">
          <span>Foveation Spatial Tier:</span>
          <span className="font-bold text-sky-400">
            {zoneDef.name}
          </span>
        </div>

        <div className="flex items-center justify-between text-gray-300 pt-1 border-t border-border-color/40">
          <div>
            <div className="font-bold text-gray-300">Prototype Traversability:</div>
            <div className="text-[8px] text-gray-500">Heuristic score: τ = τ_base × exp(-σ_z/0.15m)</div>
          </div>
          <span className="font-bold text-emerald-400 text-xs">
            {selectedCell.traversability !== undefined ? selectedCell.traversability.toFixed(2) : '1.00'}
          </span>
        </div>
      </div>
    </div>
  );
}
