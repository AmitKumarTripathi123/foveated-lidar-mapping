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
  CheckCircle2,
  PlayCircle,
} from 'lucide-react';

interface StageDef {
  id: PipelineStageId;
  label: string;
  sublabel: string;
  icon: React.ElementType;
  status: 'READY' | 'PROCESSING' | 'COMPLETE';
}

export function ResearchPipeline() {
  const activePipelineStage = useLidarStore((state) => state.activePipelineStage);
  const applyPipelinePreset = useLidarStore((state) => state.applyPipelinePreset);
  const playbackState = useLidarStore((state) => state.playbackState);

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
      label: '2. RAW LiDAR SCAN',
      sublabel: 'Unfiltered 3D Point Cloud',
      icon: Radio,
      status: 'COMPLETE',
    },
    {
      id: 'ai',
      label: '3. AI PERCEPTION',
      sublabel: 'Semantic Segmentation Model',
      icon: Cpu,
      status: playbackState === 'running' ? 'PROCESSING' : 'COMPLETE',
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
      sublabel: '5cm Near to 50cm Peripheral',
      icon: Grid,
      status: 'COMPLETE',
    },
    {
      id: 'elevation_25d',
      label: '7. 2.5D ELEVATION MAP',
      sublabel: 'Terrain, Height & Traversability',
      icon: Mountain,
      status: 'COMPLETE',
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
    <div className="bg-[#0B0F19]/90 backdrop-blur-md border border-border-color rounded-xl p-3 shadow-lg flex flex-col gap-2 font-mono text-xs text-white">
      <div className="flex items-center justify-between border-b border-border-color/60 pb-2">
        <div className="flex items-center gap-1.5 font-bold text-[11px] text-sky-400">
          <Target className="w-3.5 h-3.5 text-sky-400" />
          <span className="uppercase tracking-wider">Research Pipeline</span>
        </div>
        <span className="text-[9px] text-emerald-400 bg-emerald-950/60 border border-emerald-800/60 px-1.5 py-0.5 rounded font-bold">
          LIVE SYSTEM
        </span>
      </div>

      <div className="flex flex-col gap-1">
        {stages.map((stage) => {
          const Icon = stage.icon;
          const isActive = activePipelineStage === stage.id;

          return (
            <button
              key={stage.id}
              onClick={() => applyPipelinePreset(stage.id)}
              className={`flex items-center justify-between p-2 rounded-lg text-left transition-all border ${
                isActive
                  ? 'bg-sky-950/60 border-sky-500/60 text-white shadow-md shadow-sky-950/40'
                  : 'bg-surface-highlight/30 border-transparent hover:bg-surface-highlight/70 text-gray-300'
              }`}
            >
              <div className="flex items-center gap-2">
                <div
                  className={`p-1 rounded ${
                    isActive ? 'bg-sky-600 text-white' : 'bg-surface text-gray-400'
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
              <div className="flex items-center gap-1 shrink-0 ml-2">
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    stage.status === 'PROCESSING'
                      ? 'bg-amber-400 animate-pulse'
                      : stage.status === 'COMPLETE'
                      ? 'bg-emerald-400'
                      : 'bg-sky-400'
                  }`}
                />
                <span
                  className={`text-[9px] font-bold ${
                    stage.status === 'PROCESSING'
                      ? 'text-amber-400'
                      : stage.status === 'COMPLETE'
                      ? 'text-emerald-400'
                      : 'text-sky-400'
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
