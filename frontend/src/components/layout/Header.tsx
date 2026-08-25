'use client';

import React from 'react';
import { useLidarStore } from '@/stores/useLidarStore';
import {
  Radio,
  BarChart3,
  Wifi,
  WifiOff,
  Sparkles,
  Presentation,
  Sliders,
} from 'lucide-react';

export function Header() {
  const isConnected = useLidarStore((state) => state.isConnected);
  const setIsComparisonOpen = useLidarStore((state) => state.setIsComparisonOpen);
  const isPresentationMode = useLidarStore((state) => state.isPresentationMode);
  const setIsPresentationMode = useLidarStore((state) => state.setIsPresentationMode);
  const activeDatasetId = useLidarStore((state) => state.activeDatasetId);

  return (
    <header className="h-14 bg-[#0A0E18] border-b border-border-color px-4 flex items-center justify-between text-white select-none z-20 font-mono">
      {/* Brand Title */}
      <div className="flex items-center gap-3">
        <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-gradient-to-tr from-sky-600 to-emerald-500 shadow-md shadow-sky-950/50">
          <Radio className="w-4 h-4 text-white" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xs sm:text-sm font-bold tracking-wider">
              FOVEATED LiDAR 2.5D MAPPING
            </h1>
            <span className="bg-sky-500/20 text-sky-400 border border-sky-500/40 text-[9px] px-1.5 py-0.2 rounded font-bold">
              SIH 2026 WORKSTATION
            </span>
          </div>
          <p className="text-[10px] text-gray-400 hidden md:block">
            Variable-Resolution Perception & Semantic Elevation Map Engine
          </p>
        </div>
      </div>

      {/* Action Buttons & Status */}
      <div className="flex items-center gap-2 sm:gap-3">
        {/* Presentation Mode Button */}
        <button
          onClick={() => setIsPresentationMode(!isPresentationMode)}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all shadow-md ${
            isPresentationMode
              ? 'bg-sky-600 text-white border border-sky-400 shadow-sky-900/40'
              : 'bg-surface-highlight hover:bg-surface-highlight/80 text-sky-300 border border-sky-500/30'
          }`}
        >
          <Presentation className="w-3.5 h-3.5" />
          <span className="hidden sm:inline">PRESENTATION</span> MODE
        </button>

        {/* Benchmark Comparison Button */}
        <button
          onClick={() => setIsComparisonOpen(true)}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white text-xs font-bold shadow-md shadow-emerald-950/40 transition-all hover:scale-105"
        >
          <BarChart3 className="w-3.5 h-3.5" />
          <span className="hidden sm:inline">BENCHMARK</span> COMPARISON
        </button>

        {/* Live Stream Connection Status */}
        <div
          className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs border ${
            isConnected
              ? 'bg-emerald-950/60 text-emerald-400 border-emerald-800'
              : 'bg-red-950/60 text-red-400 border-red-800'
          }`}
        >
          {isConnected ? (
            <>
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-[10px] font-bold">WS LIVE</span>
            </>
          ) : (
            <>
              <WifiOff className="w-3 h-3 text-red-400" />
              <span className="text-[10px] font-bold">OFFLINE</span>
            </>
          )}
        </div>
      </div>
    </header>
  );
}
