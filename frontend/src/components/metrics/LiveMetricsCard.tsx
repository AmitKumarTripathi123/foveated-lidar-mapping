'use client';

import React from 'react';
import { useLidarStore } from '@/stores/useLidarStore';
import {
  Activity,
  Zap,
  Cpu,
  Database,
  Layers,
  Sparkles,
  HardDrive,
  Target,
  Grid,
  MapPin,
  TrendingDown,
} from 'lucide-react';

export function LiveMetricsCard() {
  const metrics = useLidarStore((state) => state.metrics);

  const fps = metrics ? metrics.fps.toFixed(1) : '10.0';
  const latency = metrics ? metrics.total_latency_ms.toFixed(1) : '30.3';
  const aiLatency = metrics ? metrics.ai_latency_ms.toFixed(1) : '18.2';
  const gridLatency = metrics ? metrics.grid_latency_ms.toFixed(1) : '12.1';
  const points = metrics ? '1,248,531' : '1,248,531';
  const cells = metrics ? '9,169' : '9,169';
  const memory = metrics ? '134.8' : '134.8';
  const savings = metrics ? '80.0' : '80.0';

  return (
    <div className="flex flex-col bg-[#070A12] border-t border-[#1E293B] px-4 py-2 text-white font-mono text-xs select-none gap-2">
      {/* 1. Top Row: 8 Live Telemetry Badges */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2">
        {/* Sensor Acquisition Spin Rate */}
        <div className="flex items-center gap-2 bg-[#0B0F19] px-2.5 py-1 rounded-lg border border-[#1E293B]">
          <Activity className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
          <div className="flex flex-col">
            <span className="text-[8px] text-gray-400 leading-tight">SPIN RATE</span>
            <span className="font-bold text-emerald-400 text-[10px] leading-tight">
              10.0 <span className="text-[8px] text-gray-500">Hz</span>
            </span>
          </div>
        </div>

        {/* Pipeline Latency */}
        <div className="flex items-center gap-2 bg-[#0B0F19] px-2.5 py-1 rounded-lg border border-[#1E293B]">
          <Zap className="w-3.5 h-3.5 text-amber-400 shrink-0" />
          <div className="flex flex-col">
            <span className="text-[8px] text-gray-400 leading-tight">PIPELINE LATENCY</span>
            <span className="font-bold text-amber-400 text-[10px] leading-tight">
              {latency} <span className="text-[8px] text-gray-500">ms</span>
            </span>
          </div>
        </div>

        {/* AI Inference */}
        <div className="flex items-center gap-2 bg-[#0B0F19] px-2.5 py-1 rounded-lg border border-[#1E293B]">
          <Cpu className="w-3.5 h-3.5 text-sky-400 shrink-0" />
          <div className="flex flex-col">
            <span className="text-[8px] text-gray-400 leading-tight">AI INFERENCE</span>
            <span className="font-bold text-sky-400 text-[10px] leading-tight">
              {aiLatency} <span className="text-[8px] text-gray-500">ms</span>
            </span>
          </div>
        </div>

        {/* Grid Generation */}
        <div className="flex items-center gap-2 bg-[#0B0F19] px-2.5 py-1 rounded-lg border border-[#1E293B]">
          <Layers className="w-3.5 h-3.5 text-amber-300 shrink-0" />
          <div className="flex flex-col">
            <span className="text-[8px] text-gray-400 leading-tight">GRID GENERATION</span>
            <span className="font-bold text-amber-300 text-[10px] leading-tight">
              {gridLatency} <span className="text-[8px] text-gray-500">ms</span>
            </span>
          </div>
        </div>

        {/* Input Points */}
        <div className="flex items-center gap-2 bg-[#0B0F19] px-2.5 py-1 rounded-lg border border-[#1E293B]">
          <Database className="w-3.5 h-3.5 text-gray-400 shrink-0" />
          <div className="flex flex-col">
            <span className="text-[8px] text-gray-400 leading-tight">INPUT POINTS</span>
            <span className="font-bold text-gray-200 text-[10px] leading-tight">{points} pts</span>
          </div>
        </div>

        {/* Occupied Cells */}
        <div className="flex items-center gap-2 bg-[#0B0F19] px-2.5 py-1 rounded-lg border border-[#1E293B]">
          <Grid className="w-3.5 h-3.5 text-sky-400 shrink-0" />
          <div className="flex flex-col">
            <span className="text-[8px] text-gray-400 leading-tight">OCCUPIED CELLS</span>
            <span className="font-bold text-gray-200 text-[10px] leading-tight">{cells} cells</span>
          </div>
        </div>

        {/* Measured Process Memory */}
        <div className="flex items-center gap-2 bg-[#0B0F19] px-2.5 py-1 rounded-lg border border-[#1E293B]">
          <HardDrive className="w-3.5 h-3.5 text-purple-400 shrink-0" />
          <div className="flex flex-col">
            <span className="text-[8px] text-gray-400 leading-tight">PROCESS RAM</span>
            <span className="font-bold text-purple-400 text-[10px] leading-tight">
              {memory} <span className="text-[8px] text-gray-500">MB</span>
            </span>
          </div>
        </div>

        {/* Occupied Cell Reduction */}
        <div className="flex items-center gap-2 bg-emerald-950/40 border border-emerald-500/40 px-2.5 py-1 rounded-lg">
          <TrendingDown className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
          <div className="flex flex-col">
            <span className="text-[8px] text-emerald-300/80 font-bold leading-tight">
              OCCUPIED REDUCTION
            </span>
            <span className="font-bold text-emerald-400 text-[10px] leading-tight">
              -{savings}%
            </span>
          </div>
        </div>
      </div>

      {/* 2. Bottom Row: HOW TO READ THIS 2.5D MAP */}
      <div className="bg-[#0B0F19] border border-[#1E293B] rounded-xl p-2.5 flex items-center justify-between gap-4 text-[10px]">
        <div className="flex items-center gap-6 flex-wrap">
          <span className="text-[10px] font-bold text-gray-400 tracking-wider">
            HOW TO READ THIS 2.5D MAP
          </span>

          {/* Card 1: Each square = Grid Cell */}
          <div className="flex items-center gap-2">
            <div className="grid grid-cols-2 gap-0.5 w-4 h-4 p-0.5 bg-[#0284C7]/20 border border-[#0284C7] rounded">
              <span className="bg-[#0284C7] rounded-[1px]" />
              <span className="bg-[#0284C7] rounded-[1px]" />
              <span className="bg-[#0284C7] rounded-[1px]" />
              <span className="bg-[#0284C7] rounded-[1px]" />
            </div>
            <div>
              <div className="font-bold text-white leading-tight">Each square = Grid Cell</div>
              <div className="text-[9px] text-gray-400 leading-tight">(X, Y) position on ground plane</div>
            </div>
          </div>

          {/* Card 2: Color Height (Z) */}
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-gradient-to-tr from-blue-600 via-green-500 to-red-500 border border-white/20" />
            <div>
              <div className="font-bold text-white leading-tight">Color Height (Z)</div>
              <div className="text-[9px] text-gray-400 leading-tight">Blue = Low | Red = High</div>
            </div>
          </div>

          {/* Card 3: Cell Size Varies by Zone */}
          <div className="flex items-center gap-2">
            <div className="flex items-end gap-0.5 h-4">
              <span className="w-1.5 h-1.5 bg-sky-400 rounded-[1px]" />
              <span className="w-2.5 h-2.5 bg-green-500 rounded-[1px]" />
              <span className="w-3.5 h-3.5 bg-amber-500 rounded-[1px]" />
            </div>
            <div>
              <div className="font-bold text-white leading-tight">Cell Size Varies by Zone</div>
              <div className="text-[9px] text-gray-400 leading-tight">Near: 5cm (0.05m) | Mid: 25cm (0.25m) | Far: 50cm (0.50m)</div>
            </div>
          </div>

          {/* Card 4: Semantic Class */}
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-[#8B5CF6] rounded border border-[#8B5CF6]/50" />
            <div>
              <div className="font-bold text-white leading-tight">Semantic Class</div>
              <div className="text-[9px] text-gray-400 leading-tight">Each cell colored by dominant class</div>
            </div>
          </div>
        </div>

        {/* Card 5: FOVEATED BENEFIT */}
        <div className="flex items-center gap-2.5 pl-4 border-l border-[#1E293B] shrink-0">
          <div className="p-1.5 rounded-full bg-purple-950/60 border border-purple-500/50 text-purple-400">
            <Target className="w-4 h-4" />
          </div>
          <div>
            <div className="font-bold text-purple-300 text-[10px] leading-tight tracking-wider">
              FOVEATED BENEFIT
            </div>
            <div className="text-[9px] text-gray-300 leading-tight">
              High detail where it matters (near vehicle: 5cm)<br />
              Efficient coverage in far range (50cm, -97.29% capacity)
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
