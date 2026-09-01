'use client';

import React from 'react';
import { useLidarStore } from '@/stores/useLidarStore';
import {
  Sparkles,
  ChevronRight,
  ChevronLeft,
  X,
  Radio,
  Layers,
  BarChart3,
  Cpu,
  Eye,
  MapPin,
} from 'lucide-react';

interface PresentationStep {
  step: number;
  title: string;
  tag: string;
  script: string;
  icon: any;
}

const PRESENTATION_STEPS: PresentationStep[] = [
  {
    step: 1,
    title: '1. Raw Unfiltered LiDAR Stream',
    tag: 'Stage 1 — Input Ingestion',
    script:
      'Here is the unfiltered 3D LiDAR point cloud received directly from the sensor (10 Hz). Processing this raw volume uniformly causes severe memory bottlenecks.',
    icon: Radio,
  },
  {
    step: 2,
    title: '2. Deep Learning Semantic Segmentation',
    tag: 'Stage 2 — SPVCNN Inference',
    script:
      'Our deep neural network performs point-wise classification into 4 super-classes: Drivable (Blue), Non-Drivable (Yellow), Static Obstacles (Red), and Dynamic Vehicles (Green) with 53.59% mIoU.',
    icon: Cpu,
  },
  {
    step: 3,
    title: '3. Distance-Adaptive Spatial Foveation',
    tag: 'Stage 3 — Attention Geometry',
    script:
      'We partition surrounding space into 3 foveated zones: Zone 0 (0–10m @ 5cm), Zone 1 (10–50m @ 25cm), and Zone 2 (50–100m @ 50cm) inspired by human vision focus.',
    icon: Eye,
  },
  {
    step: 4,
    title: '4. Variable-Resolution 2.5D Elevation Grid',
    tag: 'Stage 4 — Spatial Quantization',
    script:
      'Raw points are aggregated into a multi-resolution 2.5D elevation grid with heights and prototype traversability heuristic scores, drastically compressing data while retaining crucial obstacle profiles.',
    icon: Layers,
  },
  {
    step: 5,
    title: '5. 2.5D Semantic Elevation Map',
    tag: 'Stage 5 — Navigation Surface',
    script:
      'The generated 2.5D elevation layer provides height gradients and traversability analysis for path planners, detecting curbs, ramps, depressions, and terrain slope.',
    icon: MapPin,
  },
  {
    step: 6,
    title: '6. Quantitative Performance Benchmark',
    tag: 'Stage 6 — Empirical Proof',
    script:
      'Comparing uniform 5cm mapping vs our 3-Zone Foveated model proves an 80%+ reduction in occupied cell count, -97.3% theoretical capacity reduction, and a 4.6x grid generation speedup.',
    icon: BarChart3,
  },
];

export function PresentationMode() {
  const isPresentationMode = useLidarStore((state) => state.isPresentationMode);
  const setIsPresentationMode = useLidarStore((state) => state.setIsPresentationMode);
  const presentationStep = useLidarStore((state) => state.presentationStep);
  const applyPresentationStep = useLidarStore((state) => state.applyPresentationStep);

  if (!isPresentationMode) return null;

  const currentStep = PRESENTATION_STEPS[presentationStep - 1] || PRESENTATION_STEPS[0];
  const StepIcon = currentStep.icon;

  const handleNext = () => {
    if (presentationStep < 6) {
      applyPresentationStep(presentationStep + 1);
    }
  };

  const handlePrev = () => {
    if (presentationStep > 1) {
      applyPresentationStep(presentationStep - 1);
    }
  };

  return (
    <div className="absolute top-16 left-1/2 -translate-x-1/2 z-40 w-full max-w-3xl px-4 animate-in fade-in slide-in-from-top-4 duration-200 font-mono select-none">
      <div className="bg-[#0B0F19]/95 backdrop-blur-xl border border-sky-500/60 rounded-2xl p-4 shadow-2xl shadow-sky-950/50 text-white flex flex-col gap-3">
        {/* Top Header & Step Navigation */}
        <div className="flex items-center justify-between border-b border-border-color/80 pb-2.5">
          <div className="flex items-center gap-2">
            <span className="flex h-2 w-2 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-sky-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-sky-500"></span>
            </span>
            <span className="text-xs font-bold text-sky-400 tracking-wider">
              PRESENTATION MODE — STEP {presentationStep} OF 6
            </span>
          </div>

          {/* 6 Step Pills */}
          <div className="flex items-center gap-1.5">
            {PRESENTATION_STEPS.map((s) => (
              <button
                key={s.step}
                onClick={() => applyPresentationStep(s.step)}
                className={`w-7 h-7 rounded-lg text-xs font-bold transition-all flex items-center justify-center ${
                  presentationStep === s.step
                    ? 'bg-sky-600 text-white shadow-md shadow-sky-800/50 border border-sky-400'
                    : 'bg-surface-highlight/60 text-gray-400 hover:text-white'
                }`}
              >
                {s.step}
              </button>
            ))}
          </div>

          <button
            onClick={() => setIsPresentationMode(false)}
            className="p-1 rounded-lg text-gray-400 hover:text-white hover:bg-surface-highlight transition-colors"
            title="Exit Presentation Mode"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content Body */}
        <div className="flex items-start gap-3">
          <div className="p-3 rounded-xl bg-sky-950/60 border border-sky-500/40 text-sky-400 shrink-0">
            <StepIcon className="w-6 h-6" />
          </div>

          <div className="flex flex-col gap-1 flex-1">
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-bold text-white tracking-wide">
                {currentStep.title}
              </h3>
              <span className="text-[10px] bg-sky-900/60 text-sky-300 border border-sky-700/60 px-2 py-0.5 rounded font-semibold">
                {currentStep.tag}
              </span>
            </div>
            <p className="text-xs text-sky-200/90 font-sans leading-relaxed">
              &ldquo;{currentStep.script}&rdquo;
            </p>
          </div>
        </div>

        {/* Footer Controls */}
        <div className="flex items-center justify-between pt-2 border-t border-border-color/60 text-xs">
          <button
            onClick={handlePrev}
            disabled={presentationStep === 1}
            className={`flex items-center gap-1 px-3 py-1.5 rounded-lg font-bold transition-all ${
              presentationStep === 1
                ? 'opacity-40 cursor-not-allowed text-gray-500 bg-surface-highlight/30'
                : 'text-gray-200 hover:text-white bg-surface-highlight hover:bg-surface-highlight/80'
            }`}
          >
            <ChevronLeft className="w-4 h-4" />
            <span>PREVIOUS</span>
          </button>

          <span className="text-[10px] text-gray-400 font-sans">
            Guided Evaluation Flow
          </span>

          <button
            onClick={handleNext}
            disabled={presentationStep === 6}
            className={`flex items-center gap-1 px-4 py-1.5 rounded-lg font-bold transition-all ${
              presentationStep === 6
                ? 'opacity-40 cursor-not-allowed text-gray-500 bg-surface-highlight/30'
                : 'text-white bg-sky-600 hover:bg-sky-500 shadow-md shadow-sky-900/40'
            }`}
          >
            <span>{presentationStep === 6 ? 'FINAL BENCHMARK' : 'NEXT STEP'}</span>
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
