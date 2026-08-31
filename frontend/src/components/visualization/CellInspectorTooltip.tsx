'use client';

import React from 'react';
import { useLidarStore } from '@/stores/useLidarStore';
import { Layers, MapPin, Activity, Sparkles, Hash, Eye } from 'lucide-react';
import { SEMANTIC_CLASSES, FOVEATED_ZONE_COLORS } from '@/lib/semanticColors';

export function CellInspectorTooltip() {
  const hoveredCell = useLidarStore((state) => state.hoveredCell);

  if (!hoveredCell) return null;

  const semanticDef = SEMANTIC_CLASSES[hoveredCell.semantic_class] || SEMANTIC_CLASSES[0];
  const zoneDef = FOVEATED_ZONE_COLORS[hoveredCell.zone_id] || FOVEATED_ZONE_COLORS[0];
  const cellSizeM = hoveredCell.cellSize || hoveredCell.resolution || 0.05;

  return (
    <div className="absolute top-16 right-4 z-30 pointer-events-none font-mono select-none animate-in fade-in zoom-in-95 duration-100">
      <div className="bg-[#0A0E18]/95 backdrop-blur-md border border-sky-500/60 rounded-xl p-3 shadow-2xl shadow-sky-950/50 text-white flex flex-col gap-1.5 w-64">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border-color pb-1.5">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-sky-400 animate-ping" />
            <span className="text-[11px] font-bold text-sky-300">
              {hoveredCell.id || 'GRID CELL'}
            </span>
          </div>
          <span
            className="text-[9px] px-1.5 py-0.2 rounded font-bold border"
            style={{
              color: semanticDef.hex,
              backgroundColor: `${semanticDef.hex}15`,
              borderColor: `${semanticDef.hex}40`,
            }}
          >
            {semanticDef.name}
          </span>
        </div>

        {/* Spatial Coordinates */}
        <div className="grid grid-cols-2 gap-1 text-[10px] text-gray-300">
          <div>
            <span className="text-gray-400">Position: </span>
            <span className="font-bold text-white">
              ({hoveredCell.x >= 0 ? `+${hoveredCell.x.toFixed(2)}` : hoveredCell.x.toFixed(2)}m, {hoveredCell.y >= 0 ? `+${hoveredCell.y.toFixed(2)}` : hoveredCell.y.toFixed(2)}m)
            </span>
          </div>
          <div>
            <span className="text-gray-400">Elevation: </span>
            <span className="font-bold text-emerald-300">
              {hoveredCell.elevation >= 0 ? `+${hoveredCell.elevation.toFixed(2)}` : hoveredCell.elevation.toFixed(2)} m
            </span>
          </div>
        </div>

        {/* Cell Resolution & Foveation Zone */}
        <div className="text-[10px] text-gray-300 border-t border-border-color/60 pt-1 flex flex-col gap-0.5">
          <div className="flex justify-between">
            <span className="text-gray-400">Resolution:</span>
            <span className="font-bold text-amber-300">
              {(cellSizeM * 100).toFixed(0)} cm × {(cellSizeM * 100).toFixed(0)} cm
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Foveation:</span>
            <span className="font-bold text-sky-400">
              {zoneDef.name.split('—')[0].trim()}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Occupancy:</span>
            <span className="font-bold text-emerald-400">
              OCCUPIED ({hoveredCell.sourcePointCount || hoveredCell.point_count || 1} pts)
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
