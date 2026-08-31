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
  MemoryStick,
} from 'lucide-react';

export function LiveMetricsCard() {
  const metrics = useLidarStore((state) => state.metrics);

  const fps = metrics ? metrics.fps.toFixed(1) : '10.0';
  const latency = metrics ? metrics.total_latency_ms.toFixed(1) : '30.3';
  const aiLatency = metrics ? metrics.ai_latency_ms.toFixed(1) : '18.2';
  const gridLatency = metrics ? metrics.grid_latency_ms.toFixed(1) : '12.1';
  const points = metrics ? metrics.raw_point_count.toLocaleString() : '15,240';
  const cells = metrics ? metrics.cell_count.toLocaleString() : '8,450';
  const memory = metrics ? metrics.memory_ram_mb.toFixed(0) : '135';
  const savings = metrics ? metrics.compression_ratio_percent.toFixed(1) : '82.8';

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2 bg-[#0A0E18]/95 backdrop-blur-md border-t border-border-color px-3 py-2 text-white font-mono text-xs shadow-2xl select-none">
      {/* 1. FPS */}
      <div className="flex items-center gap-2 bg-[#0B0F19] px-2.5 py-1.5 rounded-lg border border-border-color/60">
        <Activity className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
        <div className="flex flex-col">
          <span className="text-[9px] text-gray-400 leading-tight">FRAME RATE</span>
          <span className="font-bold text-emerald-400 text-[11px] leading-tight">
            {fps} <span className="text-[9px] text-gray-500">Hz</span>
          </span>
        </div>
      </div>

      {/* 2. Total Latency */}
      <div className="flex items-center gap-2 bg-[#0B0F19] px-2.5 py-1.5 rounded-lg border border-border-color/60">
        <Zap className="w-3.5 h-3.5 text-amber-400 shrink-0" />
        <div className="flex flex-col">
          <span className="text-[9px] text-gray-400 leading-tight">PIPELINE LATENCY</span>
          <span className="font-bold text-amber-400 text-[11px] leading-tight">
            {latency} <span className="text-[9px] text-gray-500">ms</span>
          </span>
        </div>
      </div>

      {/* 3. AI Inference */}
      <div className="flex items-center gap-2 bg-[#0B0F19] px-2.5 py-1.5 rounded-lg border border-border-color/60">
        <Cpu className="w-3.5 h-3.5 text-sky-400 shrink-0" />
        <div className="flex flex-col">
          <span className="text-[9px] text-gray-400 leading-tight">AI INFERENCE</span>
          <span className="font-bold text-sky-400 text-[11px] leading-tight">
            {aiLatency} <span className="text-[9px] text-gray-500">ms</span>
          </span>
        </div>
      </div>

      {/* 4. Grid Construction */}
      <div className="flex items-center gap-2 bg-[#0B0F19] px-2.5 py-1.5 rounded-lg border border-border-color/60">
        <Layers className="w-3.5 h-3.5 text-amber-300 shrink-0" />
        <div className="flex flex-col">
          <span className="text-[9px] text-gray-400 leading-tight">GRID GENERATION</span>
          <span className="font-bold text-amber-300 text-[11px] leading-tight">
            {gridLatency} <span className="text-[9px] text-gray-500">ms</span>
          </span>
        </div>
      </div>

      {/* 5. Raw Points */}
      <div className="flex items-center gap-2 bg-[#0B0F19] px-2.5 py-1.5 rounded-lg border border-border-color/60">
        <Database className="w-3.5 h-3.5 text-gray-400 shrink-0" />
        <div className="flex flex-col">
          <span className="text-[9px] text-gray-400 leading-tight">INPUT POINTS</span>
          <span className="font-bold text-gray-200 text-[11px] leading-tight">{points}</span>
        </div>
      </div>

      {/* 6. 2.5D Cells */}
      <div className="flex items-center gap-2 bg-[#0B0F19] px-2.5 py-1.5 rounded-lg border border-border-color/60">
        <Layers className="w-3.5 h-3.5 text-gray-400 shrink-0" />
        <div className="flex flex-col">
          <span className="text-[9px] text-gray-400 leading-tight">OCCUPIED CELLS</span>
          <span className="font-bold text-gray-200 text-[11px] leading-tight">{cells}</span>
        </div>
      </div>

      {/* 7. Host RAM */}
      <div className="flex items-center gap-2 bg-[#0B0F19] px-2.5 py-1.5 rounded-lg border border-border-color/60">
        <Cpu className="w-3.5 h-3.5 text-purple-400 shrink-0" />
        <div className="flex flex-col">
          <span className="text-[9px] text-gray-400 leading-tight">HOST MEMORY</span>
          <span className="font-bold text-purple-400 text-[11px] leading-tight">
            {memory} <span className="text-[9px] text-gray-500">MB</span>
          </span>
        </div>
      </div>

      {/* 8. Memory & Cell Reduction */}
      <div className="flex items-center gap-2 bg-emerald-950/50 border border-emerald-500/50 px-2.5 py-1.5 rounded-lg">
        <Sparkles className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
        <div className="flex flex-col">
          <span className="text-[9px] text-emerald-300/80 font-bold leading-tight">
            MEMORY REDUCTION
          </span>
          <span className="font-bold text-emerald-400 text-[11px] leading-tight">
            -{savings}%
          </span>
        </div>
      </div>
    </div>
  );
}
