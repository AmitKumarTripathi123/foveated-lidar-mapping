'use client';

import React, { useState } from 'react';
import { useLidarStore } from '@/stores/useLidarStore';
import { SEMANTIC_CLASSES, FOVEATED_ZONE_COLORS } from '@/lib/semanticColors';
import { Palette, ChevronDown, ChevronUp, Layers, TrendingUp } from 'lucide-react';

export function CompactLegends() {
  const colorMode = useLidarStore((state) => state.colorMode);
  const layers = useLidarStore((state) => state.layers);
  const [isExpanded, setIsExpanded] = useState(true);

  return (
    <div className="absolute bottom-4 left-4 z-20 flex flex-col gap-2 max-w-sm font-mono select-none">
      <div className="bg-[#0B0F19]/90 backdrop-blur-md border border-border-color rounded-xl p-2.5 shadow-2xl text-xs text-white">
        {/* Toggle Header */}
        <div
          onClick={() => setIsExpanded(!isExpanded)}
          className="flex items-center justify-between cursor-pointer text-gray-300 hover:text-white pb-1"
        >
          <div className="flex items-center gap-1.5 font-bold text-[11px]">
            <Palette className="w-3.5 h-3.5 text-brand-500" />
            <span className="uppercase tracking-wider">
              {colorMode === 'semantic'
                ? 'Semantic Classes'
                : colorMode === 'elevation'
                ? 'Elevation Colormap'
                : colorMode === 'traversability'
                ? 'Traversability Gradient'
                : 'LiDAR Intensity'}
            </span>
          </div>
          {isExpanded ? (
            <ChevronDown className="w-3.5 h-3.5 text-gray-400" />
          ) : (
            <ChevronUp className="w-3.5 h-3.5 text-gray-400" />
          )}
        </div>

        {/* Expanded Content */}
        {isExpanded && (
          <div className="pt-2 border-t border-border-color/60 space-y-2">
            {/* 1. Semantic Class Palette */}
            {colorMode === 'semantic' && (
              <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[10px]">
                {Object.values(SEMANTIC_CLASSES).map((item) => (
                  <div key={item.id} className="flex items-center gap-1.5">
                    <span
                      className="w-2.5 h-2.5 rounded-sm shrink-0 border border-white/20"
                      style={{ backgroundColor: item.hex }}
                    />
                    <span className="text-gray-200 truncate">{item.name}</span>
                  </div>
                ))}
              </div>
            )}

            {/* 2. Elevation Gradient */}
            {colorMode === 'elevation' && (
              <div className="space-y-1">
                <div className="h-3 rounded w-full bg-gradient-to-r from-blue-600 via-cyan-400 via-emerald-400 via-yellow-400 to-red-500 border border-white/20" />
                <div className="flex justify-between text-[10px] text-gray-400 font-mono">
                  <span>Low (-2.5m)</span>
                  <span>Ground (-1.5m)</span>
                  <span>High (+3.0m)</span>
                </div>
              </div>
            )}

            {/* 3. Traversability Gradient */}
            {colorMode === 'traversability' && (
              <div className="space-y-1">
                <div className="h-3 rounded w-full bg-gradient-to-r from-red-500 via-amber-400 to-emerald-400 border border-white/20" />
                <div className="flex justify-between text-[10px] font-mono">
                  <span className="text-red-400">0.0 (Hazard/Obstacle)</span>
                  <span className="text-amber-400">0.5 (Caution)</span>
                  <span className="text-emerald-400">1.0 (Drivable)</span>
                </div>
              </div>
            )}

            {/* 4. Spatial Foveation Zones (When foveation layer or adaptive grid is on) */}
            {(layers.foveatedGrid || layers.zoneRings) && (
              <div className="pt-1.5 border-t border-border-color/40">
                <div className="text-[9px] font-bold text-gray-400 uppercase tracking-wide mb-1 flex items-center gap-1">
                  <Layers className="w-3 h-3 text-foveated-near" />
                  <span>Multi-Resolution Spatial Hierarchy</span>
                </div>
                <div className="grid grid-cols-2 gap-1 text-[9px]">
                  {Object.entries(FOVEATED_ZONE_COLORS).map(([id, z]) => (
                    <div key={id} className="flex items-center gap-1 bg-surface-highlight/40 px-1.5 py-0.5 rounded">
                      <span className="w-2 h-2 rounded-full" style={{ backgroundColor: z.hex }} />
                      <span className="text-gray-300">{z.name.split(' ')[0]}</span>
                      <span className="font-bold text-gray-100 ml-auto">{z.resolution.split(' ')[0]}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
