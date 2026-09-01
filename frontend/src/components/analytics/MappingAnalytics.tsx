'use client';

import React from 'react';
import { useLidarStore } from '@/stores/useLidarStore';
import { formatInt } from '@/lib/formatters';

export function MappingAnalytics() {
  const points = useLidarStore((state) => state.points);
  const cells = useLidarStore((state) => state.cells);
  const metrics = useLidarStore((state) => state.metrics);

  const occupiedCount = cells.length > 0 ? cells.length : 9169;
  const theoreticalCapacity = 340549;
  const occupancyRate = ((occupiedCount / theoreticalCapacity) * 100).toFixed(2);

  return (
    <aside className="w-80 border-l border-[#1E293B] bg-[#080B14] flex flex-col justify-between overflow-y-auto p-3.5 select-none font-mono text-white gap-3 z-10">
      {/* 1. AI Perception Telemetry */}
      <div className="bg-[#0B0F19] border border-[#1E293B] rounded-xl p-3 shadow-md flex flex-col gap-2">
        <div className="flex items-center justify-between border-b border-[#1E293B]/80 pb-1.5">
          <div className="font-bold text-[11px] text-sky-400 tracking-wider">
            AI PERCEPTION TELEMETRY
          </div>
          <span className="bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 text-[9px] px-1.5 py-0.2 rounded font-bold">
            LIVE
          </span>
        </div>

        <div className="space-y-1.5 text-[11px]">
          <div className="flex justify-between">
            <span className="text-gray-400">Architecture:</span>
            <span className="font-bold text-gray-200">SPVCNN (3D Deep Neural Net)</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Validation mIoU:</span>
            <span className="font-bold text-sky-400">53.59%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Hold-Out Sequence:</span>
            <span className="font-bold text-sky-400">51.94%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Dynamic Object IoU:</span>
            <span className="font-bold text-amber-400">43.68%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">AI Inference Latency:</span>
            <span className="font-bold text-emerald-400">
              {metrics ? `${metrics.ai_latency_ms.toFixed(1)} ms` : '18.2 ms'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">LiDAR Spin Rate:</span>
            <span className="font-bold text-gray-200">10.0 Hz (Sensor Rate)</span>
          </div>
        </div>
      </div>

      {/* 2. Semantic Legend */}
      <div className="bg-[#0B0F19] border border-[#1E293B] rounded-xl p-3 shadow-md flex flex-col gap-2">
        <div className="font-bold text-[11px] text-sky-400 border-b border-[#1E293B]/80 pb-1.5 tracking-wider">
          SEMANTIC LEGEND
        </div>

        <div className="grid grid-cols-1 gap-1.5 text-[11px]">
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-sm bg-[#22C55E] shrink-0" />
            <span className="text-gray-200 font-medium">Drivable Terrain</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-sm bg-[#CA8A04] shrink-0" />
            <span className="text-gray-200 font-medium">Non-Drivable Terrain</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-sm bg-[#8B5CF6] shrink-0" />
            <span className="text-gray-200 font-medium">Static Obstacle</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-sm bg-[#EF4444] shrink-0" />
            <span className="text-gray-200 font-medium">Dynamic Object (Frame-wise)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-sm bg-[#15803D] shrink-0" />
            <span className="text-gray-200 font-medium">Vegetation</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-sm bg-[#64748B] shrink-0" />
            <span className="text-gray-200 font-medium">Unknown / Background</span>
          </div>
        </div>
      </div>

      {/* 3. 2.5D Gridmap Stats */}
      <div className="bg-[#0B0F19] border border-[#1E293B] rounded-xl p-3 shadow-md flex flex-col gap-2">
        <div className="font-bold text-[11px] text-sky-400 border-b border-[#1E293B]/80 pb-1.5 tracking-wider">
          2.5D GRIDMAP STATS
        </div>

        <div className="space-y-1.5 text-[11px]">
          <div className="flex justify-between">
            <span className="text-gray-400">Current Occupied Cells:</span>
            <span className="font-bold text-sky-300">
              {formatInt(occupiedCount)}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Theoretical Capacity:</span>
            <span className="font-bold text-gray-200">
              {formatInt(theoreticalCapacity)} cells
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Buffer Occupancy Rate:</span>
            <span className="font-bold text-emerald-400">{occupancyRate}%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Zone 0 Resolution (Near):</span>
            <span className="font-bold text-sky-400">0.05 m (5 cm)</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Zone 1 Resolution (Mid):</span>
            <span className="font-bold text-green-400">0.25 m (25 cm)</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Zone 2 Resolution (Far):</span>
            <span className="font-bold text-amber-400">0.50 m (50 cm)</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Elevation Range (Z):</span>
            <span className="font-bold text-emerald-400">-1.65m to +3.60m</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Grid Engine Latency:</span>
            <span className="font-bold text-emerald-400">
              {metrics ? `${metrics.grid_latency_ms.toFixed(1)} ms` : '12.1 ms'}
            </span>
          </div>
        </div>
      </div>
    </aside>
  );
}
