'use client';

import React from 'react';
import { useLidarStore } from '@/stores/useLidarStore';
import { PipelineStageId } from '@/types/lidar';
import {
  Database,
  Radio,
  Cpu,
  CircleDot,
  Target,
  Grid,
  Mountain,
  BarChart3,
} from 'lucide-react';

interface StageDef {
  id: PipelineStageId;
  label: string;
  sublabel: string;
  icon: React.ElementType;
  status: 'COMPLETE' | 'LIVE' | 'READY';
}

export function ResearchPipeline() {
  const activePipelineStage = useLidarStore((state) => state.activePipelineStage);
  const applyPipelinePreset = useLidarStore((state) => state.applyPipelinePreset);

  const stages: StageDef[] = [
    {
      id: 'data',
      label: '1. DATA PIPELINE',
      sublabel: 'LiDAR Ingestion & Normalization',
      icon: Database,
      status: 'COMPLETE',
    },
    {
      id: 'raw',
      label: '2. RAW LIDAR SCAN',
      sublabel: 'Unfiltered 3D Point Cloud',
      icon: Radio,
      status: 'COMPLETE',
    },
    {
      id: 'ai',
      label: '3. AI PERCEPTION',
      sublabel: 'Semantic Segmentation Model',
      icon: Cpu,
      status: 'COMPLETE',
    },
    {
      id: 'semantic',
      label: '4. SEMANTIC CLOUD',
      sublabel: 'Per-Point Class Annotations',
      icon: CircleDot,
      status: 'COMPLETE',
    },
    {
      id: 'foveation',
      label: '5. FOVEATION ATTENTION',
      sublabel: 'Spatial Multi-Resolution Zones',
      icon: Target,
      status: 'COMPLETE',
    },
    {
      id: 'variable_grid',
      label: '6. VARIABLE GRID',
      sublabel: 'Foveated Grid Generation',
      icon: Grid,
      status: 'COMPLETE',
    },
    {
      id: 'elevation_25d',
      label: '7. 2.5D ELEVATION MAP',
      sublabel: 'Terrain, Height & Traversability',
      icon: Mountain,
      status: 'LIVE',
    },
    {
      id: 'benchmark',
      label: '8. BENCHMARK ANALYSIS',
      sublabel: 'Uniform vs Foveated Efficiency',
      icon: BarChart3,
      status: 'READY',
    },
  ];

  return (
    <div className="bg-[#0B0F19]/90 backdrop-blur-md border border-[#1E293B] rounded-xl p-3 shadow-lg flex flex-col gap-2 font-mono text-xs text-white">
      <div className="flex items-center justify-between border-b border-[#1E293B]/80 pb-2">
        <div className="font-bold text-[11px] text-sky-400 tracking-wider">
          RESEARCH PIPELINE
        </div>
        <span className="text-[9px] text-emerald-400 bg-emerald-950/60 border border-emerald-800/60 px-1.5 py-0.5 rounded font-bold">
          LIVE
        </span>
      </div>

      <div className="flex flex-col gap-1">
        {stages.map((stage) => {
          const Icon = stage.icon;
          const isActive = activePipelineStage === stage.id || (activePipelineStage === 'elevation_25d' && stage.id === 'elevation_25d');

          return (
            <button
              key={stage.id}
              onClick={() => applyPipelinePreset(stage.id)}
              className={`flex items-center justify-between p-2 rounded-xl text-left transition-all border ${
                isActive
                  ? 'bg-[#4338CA]/30 border-[#6366F1] text-white shadow-md shadow-indigo-950/50'
                  : 'bg-[#0F172A]/40 border-transparent hover:bg-[#1E293B]/50 text-gray-300'
              }`}
            >
              <div className="flex items-center gap-2">
                <div
                  className={`p-1.5 rounded-lg ${
                    isActive ? 'bg-[#6366F1] text-white' : 'bg-[#1E293B] text-gray-400'
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                </div>
                <div className="flex flex-col">
                  <span className="font-bold text-[11px] leading-tight tracking-tight">
                    {stage.label}
                  </span>
                  <span className="text-[9px] text-gray-400 leading-tight">
                    {stage.sublabel}
                  </span>
                </div>
              </div>

              {/* Status Indicator */}
              <div className="flex items-center gap-1 shrink-0 ml-1.5">
                <span
                  className={`text-[8px] font-bold px-1.5 py-0.2 rounded border ${
                    stage.status === 'LIVE'
                      ? 'bg-[#6366F1]/20 text-[#A5B4FC] border-[#6366F1]/40'
                      : stage.status === 'COMPLETE'
                      ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                      : 'bg-sky-500/10 text-sky-400 border-sky-500/20'
                  }`}
                >
                  {stage.status}
                </span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
