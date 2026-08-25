'use client';

import React from 'react';
import { useLidarStore } from '@/stores/useLidarStore';
import { X, Layers, Box, Info } from 'lucide-react';
import { SEMANTIC_CLASSES, FOVEATED_ZONE_COLORS } from '@/lib/semanticColors';

export function CellInspectorTooltip() {
  const selectedCell = useLidarStore((state) => state.selectedCell);
  const hoveredCell = useLidarStore((state) => state.hoveredCell);
  const selectedBox = useLidarStore((state) => state.selectedBox);
  const setSelectedCell = useLidarStore((state) => state.setSelectedCell);
  const setSelectedBox = useLidarStore((state) => state.setSelectedBox);

  const activeCell = selectedCell || hoveredCell;

  if (!activeCell && !selectedBox) return null;

  return (
    <div className="absolute top-16 right-4 z-30 w-80 bg-[#0B0F19]/95 backdrop-blur-md border border-border-color rounded-xl p-3.5 shadow-2xl text-white font-mono animate-in fade-in duration-150">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border-color/80 pb-2 mb-2.5">
        <div className="flex items-center gap-2">
          {activeCell ? (
            <>
              <Layers className="w-4 h-4 text-sky-400" />
              <span className="text-xs font-bold tracking-wide text-sky-300">
                {selectedCell ? 'INSPECTED 2.5D CELL' : 'HOVERED CELL'}
              </span>
            </>
          ) : (
            <>
              <Box className="w-4 h-4 text-amber-400" />
              <span className="text-xs font-bold tracking-wide text-amber-300">
                INSPECTED 3D OBJECT
              </span>
            </>
          )}
        </div>
        {selectedCell || selectedBox ? (
          <button
            onClick={() => {
              setSelectedCell(null);
              setSelectedBox(null);
            }}
            className="text-gray-400 hover:text-white p-1 rounded-md hover:bg-surface-highlight transition-colors"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        ) : (
          <span className="text-[10px] text-gray-400">Live Hover</span>
        )}
      </div>

      {/* Cell Details */}
      {activeCell && (
        <div className="space-y-1.5 text-xs">
          <div className="flex justify-between items-center bg-surface-highlight/50 px-2 py-1 rounded">
            <span className="text-gray-400">Semantic Category:</span>
            <span
              className="font-bold px-1.5 py-0.5 rounded text-[11px]"
              style={{
                backgroundColor: `${SEMANTIC_CLASSES[activeCell.semantic_class]?.hex || '#64748B'}33`,
                color: SEMANTIC_CLASSES[activeCell.semantic_class]?.hex || '#64748B',
              }}
            >
              {activeCell.class_name} ({Math.round(activeCell.confidence * 100)}%)
            </span>
          </div>

          <div className="flex justify-between items-center px-2 py-0.5">
            <span className="text-gray-400">Spatial Resolution:</span>
            <span className="font-bold text-sky-400">
              {Math.round(activeCell.resolution * 100)} cm ({activeCell.resolution}m)
            </span>
          </div>

          <div className="flex justify-between items-center px-2 py-0.5">
            <span className="text-gray-400">Foveation Zone:</span>
            <span className="font-bold text-[11px]" style={{ color: FOVEATED_ZONE_COLORS[activeCell.zone_id]?.hex }}>
              {FOVEATED_ZONE_COLORS[activeCell.zone_id]?.name}
            </span>
          </div>

          <div className="flex justify-between items-center px-2 py-0.5">
            <span className="text-gray-400">Spatial Center (X, Y):</span>
            <span className="text-gray-200">
              {activeCell.x.toFixed(2)}m, {activeCell.y.toFixed(2)}m
            </span>
          </div>

          <div className="flex justify-between items-center px-2 py-0.5">
            <span className="text-gray-400">Ground Elevation (Z):</span>
            <span className="text-emerald-400 font-bold">
              {activeCell.elevation.toFixed(3)} m
            </span>
          </div>

          <div className="flex justify-between items-center px-2 py-0.5">
            <span className="text-gray-400">Traversability:</span>
            <span
              className={`font-bold ${
                activeCell.traversability >= 0.8
                  ? 'text-emerald-400'
                  : activeCell.traversability >= 0.3
                  ? 'text-amber-400'
                  : 'text-red-400'
              }`}
            >
              {(activeCell.traversability * 100).toFixed(0)}%{' '}
              {activeCell.traversability >= 0.8
                ? '(Drivable)'
                : activeCell.traversability >= 0.3
                ? '(Caution)'
                : '(Hazard)'}
            </span>
          </div>

          <div className="flex justify-between items-center px-2 py-0.5">
            <span className="text-gray-400">LiDAR Point Density:</span>
            <span className="text-gray-200">{activeCell.point_count} points</span>
          </div>

          <div className="flex justify-between items-center px-2 py-0.5">
            <span className="text-gray-400">Surface Roughness:</span>
            <span className="text-gray-200">±{activeCell.roughness.toFixed(3)} m</span>
          </div>
        </div>
      )}

      {/* Object Box Details */}
      {selectedBox && !selectedCell && (
        <div className="space-y-1.5 text-xs">
          <div className="flex justify-between items-center bg-surface-highlight/50 px-2 py-1 rounded">
            <span className="text-gray-400">Detected Object:</span>
            <span className="font-bold text-amber-400">{selectedBox.class_name}</span>
          </div>
          <div className="flex justify-between items-center px-2 py-0.5">
            <span className="text-gray-400">Confidence Score:</span>
            <span className="text-emerald-400 font-bold">
              {(selectedBox.confidence * 100).toFixed(1)}%
            </span>
          </div>
          <div className="flex justify-between items-center px-2 py-0.5">
            <span className="text-gray-400">Dimensions (L × W × H):</span>
            <span className="text-gray-200">
              {selectedBox.size[0]}m × {selectedBox.size[1]}m × {selectedBox.size[2]}m
            </span>
          </div>
          <div className="flex justify-between items-center px-2 py-0.5">
            <span className="text-gray-400">Center (X, Y, Z):</span>
            <span className="text-gray-200">
              {selectedBox.center[0]}m, {selectedBox.center[1]}m, {selectedBox.center[2]}m
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
