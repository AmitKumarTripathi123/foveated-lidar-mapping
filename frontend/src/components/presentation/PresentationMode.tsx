'use client';

import React from 'react';
import { useLidarStore } from '@/stores/useLidarStore';
import {
  ChevronLeft,
  ChevronRight,
  X,
  Radio,
  Cpu,
  Target,
  Grid,
  Mountain,
  BarChart3,
  Sparkles,
} from 'lucide-react';

interface StepDetail {
  step: number;
  title: string;
  subtitle: string;
  script: string;
  icon: React.ElementType;
  tag: string;
}

const PRESENTATION_STEPS: StepDetail[] = [
  {
    step: 1,
    title: 'STEP 1 — RAW LiDAR SCAN',
    subtitle: 'Massive Unfiltered Spatial Point Cloud',
    script:
      'This is the raw 3D LiDAR point cloud. It produces tens of thousands of unstructured (X, Y, Z, Intensity) measurements every second, posing heavy computational burdens.',
    icon: Radio,
    tag: 'Raw 3D Scan',
  },
  {
    step: 2,
    title: 'STEP 2 — AI PERCEPTION',
    subtitle: 'Deep Learning Semantic Segmentation',
    script:
      'Our deep learning perception pipeline classifies every LiDAR point into semantic categories: drivable road surface, sidewalks, vehicles, pedestrians, poles, and terrain in real time.',
    icon: Cpu,
    tag: 'PointNet++ / DL',
  },
  {
    step: 3,
    title: 'STEP 3 — FOVEATED ATTENTION',
    subtitle: 'Bio-Inspired Multi-Resolution Hierarchy',
    script:
      'We do not represent every region at the same spatial resolution. We mimic human foveal vision: dense attention near the ego vehicle, transitioning to coarser peripheral zones.',
    icon: Target,
    tag: 'Foveation Theory',
  },
  {
    step: 4,
    title: 'STEP 4 — VARIABLE-RESOLUTION ADAPTIVE GRID',
    subtitle: '5cm Near-Field to 50cm Peripheral Discretization',
    script:
      'The critical near/foveal region (0-10m) receives fine 5cm resolution for obstacle avoidance, while peripheral corridors use 10cm, 25cm, and 50cm cells to slash memory.',
    icon: Grid,
    tag: 'Adaptive Cells',
  },
  {
    step: 5,
    title: 'STEP 5 — 2.5D SEMANTIC ELEVATION MAP',
    subtitle: 'Ground Elevation, Surface Roughness & Traversability',
    script:
      'We aggregate points into a compact 2.5D representation storing ground elevation (Z), surface roughness, dominant semantic class, and drivability rating for navigation planners.',
    icon: Mountain,
    tag: '2.5D Elevation Map',
  },
  {
    step: 6,
    title: 'STEP 6 — BENCHMARK & EFFICIENCY ANALYSIS',
    subtitle: 'Quantitative Comparison: Uniform vs. Foveated Grid',
    script:
      'We benchmark our foveated representation against a standard uniform 5cm grid, achieving ~82.8% memory reduction and 4.6x processing speedup with zero loss in near-field safety.',
    icon: BarChart3,
    tag: 'Quantitative Proof',
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
              SIH PRESENTATION MODE — STEP {presentationStep} OF 6
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
            Guided SIH Evaluation Flow
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
