'use client';

import React from 'react';
import { useLidarStore } from '@/stores/useLidarStore';
import { X, Layers, MapPin, Activity, Sparkles, Hash, Gauge, Box } from 'lucide-react';
import { SEMANTIC_CLASSES, FOVEATED_ZONE_COLORS, getSemanticHex } from '@/lib/semanticColors';

export function CellDetailDrawer() {
  const selectedCell = useLidarStore((state) => state.selectedCell);
  const setSelectedCell = useLidarStore((state) => state.setSelectedCell);

  if (!selectedCell) return null;

  const semanticDef = SEMANTIC_CLASSES[selectedCell.semantic_class] || SEMANTIC_CLASSES[0];
  const zoneDef = FOVEATED_ZONE_COLORS[selectedCell.zone_id] || FOVEATED_ZONE_COLORS[0];

  return (
    <div className="absolute bottom-16 right-4 z-40 w-80 bg-[#0B0F19]/95 backdrop-blur-xl border border-sky-500/50 rounded-2xl p-3.5 shadow-2xl shadow-sky-950/60 font-mono text-white select-none animate-in fade-in slide-in-from-bottom-3 duration-200">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border-color/80 pb-2 mb-2.5">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-sky-950 border border-sky-500/40 text-sky-400">
            <Box className="w-4 h-4" />
          </div>
          <div>
            <div className="text-xs font-bold text-sky-300 tracking-wider">
              {selectedCell.id || 'GRID CELL INSPECTOR'}
            </div>
            <div className="text-[9px] text-gray-400">
              Foveated 2.5D Spatial Voxel
            </div>
          </div>
        </div>
        <button
          onClick={() => setSelectedCell(null)}
          className="p-1 rounded-lg text-gray-400 hover:text-white hover:bg-surface-highlight transition-colors"
          title="Close Cell Inspector"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Grid Properties */}
      <div className="grid grid-cols-2 gap-2 text-[11px] mb-3">
        {/* World Position */}
        <div className="bg-surface-highlight/40 border border-border-color/50 rounded-xl p-2 flex flex-col">
          <span className="text-[9px] text-gray-400 uppercase font-bold flex items-center gap-1">
            <MapPin className="w-3 h-3 text-sky-400" />
            World (X, Y)
          </span>
          <span className="text-white font-bold mt-0.5">
            ({selectedCell.x >= 0 ? `+${selectedCell.x.toFixed(2)}` : selectedCell.x.toFixed(2)}m, {selectedCell.y >= 0 ? `+${selectedCell.y.toFixed(2)}` : selectedCell.y.toFixed(2)}m)
          </span>
        </div>

        {/* Elevation Z */}
        <div className="bg-surface-highlight/40 border border-border-color/50 rounded-xl p-2 flex flex-col">
          <span className="text-[9px] text-gray-400 uppercase font-bold flex items-center gap-1">
            <Activity className="w-3 h-3 text-emerald-400" />
            Elevation (Z)
          </span>
          <span className="text-emerald-300 font-bold mt-0.5">
            {selectedCell.elevation >= 0 ? `+${selectedCell.elevation.toFixed(2)}` : selectedCell.elevation.toFixed(2)} m
          </span>
        </div>

        {/* Resolution / Cell Size */}
        <div className="bg-surface-highlight/40 border border-border-color/50 rounded-xl p-2 flex flex-col">
          <span className="text-[9px] text-gray-400 uppercase font-bold flex items-center gap-1">
            <Layers className="w-3 h-3 text-amber-400" />
            Cell Resolution
          </span>
          <span className="text-amber-300 font-bold mt-0.5">
            {(selectedCell.cellSize || selectedCell.resolution || 0.05) * 100} cm ({(selectedCell.cellSize || selectedCell.resolution || 0.05).toFixed(2)}m)
          </span>
        </div>

        {/* Source LiDAR Points */}
        <div className="bg-surface-highlight/40 border border-border-color/50 rounded-xl p-2 flex flex-col">
          <span className="text-[9px] text-gray-400 uppercase font-bold flex items-center gap-1">
            <Hash className="w-3 h-3 text-purple-400" />
            LiDAR Points
          </span>
          <span className="text-purple-300 font-bold mt-0.5">
            {selectedCell.sourcePointCount || selectedCell.point_count || 1} points
          </span>
        </div>
      </div>

      {/* Semantic Classification Card */}
      <div className="bg-[#0A0E18] border border-border-color rounded-xl p-2.5 space-y-1.5 text-[10px]">
        <div className="flex items-center justify-between">
          <span className="text-gray-400 uppercase font-bold text-[9px]">Semantic Class:</span>
          <span
            className="px-2 py-0.5 rounded font-bold border"
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
          <span>AI Confidence:</span>
          <span className="font-bold text-white">
            {((selectedCell.confidence || 0.95) * 100).toFixed(1)}%
          </span>
        </div>

        <div className="flex items-center justify-between text-gray-300">
          <span>Foveation Zone:</span>
          <span className="font-bold text-sky-400">
            {zoneDef.name}
          </span>
        </div>

        <div className="flex items-center justify-between text-gray-300">
          <span>Traversability Score:</span>
          <span className="font-bold text-emerald-400">
            {selectedCell.traversability !== undefined ? selectedCell.traversability.toFixed(2) : '1.00'}
          </span>
        </div>
      </div>
    </div>
  );
}
