'use client';

import React from 'react';
import { useLidarStore } from '@/stores/useLidarStore';
import { Layers, MapPin, Activity, Sparkles, Hash, Eye, ShieldAlert } from 'lucide-react';
import { SEMANTIC_CLASSES, FOVEATED_ZONE_COLORS } from '@/lib/semanticColors';

export function CellInspectorTooltip() {
  const hoveredCell = useLidarStore((state) => state.hoveredCell);

  if (!hoveredCell) return null;

  const semanticDef = SEMANTIC_CLASSES[hoveredCell.semantic_class] || SEMANTIC_CLASSES[0];
  const zoneDef = FOVEATED_ZONE_COLORS[hoveredCell.zone_id] || FOVEATED_ZONE_COLORS[0];
  const cellSizeM = hoveredCell.cellSize || hoveredCell.resolution || 0.05;
  const zMean = hoveredCell.elevation;
  const zMin = hoveredCell.minElevation !== undefined ? hoveredCell.minElevation : zMean - 0.04;
  const zMax = hoveredCell.maxElevation !== undefined ? hoveredCell.maxElevation : zMean + 0.04;
  const roughnessCm = hoveredCell.roughness !== undefined ? (hoveredCell.roughness * 100).toFixed(1) : '2.1';
  const trav = hoveredCell.traversability !== undefined ? hoveredCell.traversability.toFixed(2) : '0.95';
  const conf = ((hoveredCell.confidence || 0.95) * 100).toFixed(1);
  const ptCount = hoveredCell.sourcePointCount || hoveredCell.point_count || 1;

  return (
    <div className="absolute top-16 right-4 z-30 pointer-events-none font-mono select-none animate-in fade-in zoom-in-95 duration-100">
      <div className="bg-[#0A0E18]/95 backdrop-blur-md border border-sky-500/70 rounded-xl p-3 shadow-2xl shadow-sky-950/60 text-white flex flex-col gap-1.5 w-72">
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

        {/* Spatial Coordinates & Zone */}
        <div className="grid grid-cols-2 gap-1 text-[10px] text-gray-300">
          <div>
            <span className="text-gray-400">Position (X, Y): </span>
            <span className="font-bold text-white block">
              {hoveredCell.x >= 0 ? `+${hoveredCell.x.toFixed(2)}` : hoveredCell.x.toFixed(2)}m, {hoveredCell.y >= 0 ? `+${hoveredCell.y.toFixed(2)}` : hoveredCell.y.toFixed(2)}m
            </span>
          </div>
          <div>
            <span className="text-gray-400">Zone &amp; Size: </span>
            <span className="font-bold text-amber-300 block">
              {(cellSizeM * 100).toFixed(0)}cm ({zoneDef.name.split('—')[0].trim()})
            </span>
          </div>
        </div>

        {/* Elevation Stats (Z_mean, Z_min, Z_max) */}
        <div className="bg-[#080C16] border border-border-color/60 rounded-lg p-1.5 text-[9px] flex flex-col gap-0.5">
          <div className="flex justify-between">
            <span className="text-gray-400">Elevation (Z_mean):</span>
            <span className="font-bold text-emerald-300">
              {zMean >= 0 ? `+${zMean.toFixed(2)}` : zMean.toFixed(2)} m
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Elevation Range:</span>
            <span className="font-bold text-gray-200">
              [{zMin.toFixed(2)}m ... {zMax.toFixed(2)}m]
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Roughness (σ_z):</span>
            <span className="font-bold text-amber-400">{roughnessCm} cm</span>
          </div>
        </div>

        {/* Semantic, Confidence, Traversability, Points */}
        <div className="text-[9px] text-gray-300 flex flex-col gap-0.5 border-t border-border-color/60 pt-1">
          <div className="flex justify-between">
            <span className="text-gray-400">AI Confidence:</span>
            <span className="font-bold text-sky-400">{conf}%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Traversability:</span>
            <span className="font-bold text-emerald-400">{trav} (Heuristic)</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Source Points:</span>
            <span className="font-bold text-purple-300">{ptCount} LiDAR pts</span>
          </div>
        </div>
      </div>
    </div>
  );
}
