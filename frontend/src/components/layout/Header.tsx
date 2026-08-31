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
              AUTONOMOUS WORKSTATION
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
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold bg-emerald-600/80 hover:bg-emerald-600 text-white border border-emerald-400/50 shadow-md shadow-emerald-950/40 transition-all"
        >
          <BarChart3 className="w-3.5 h-3.5" />
          <span className="hidden sm:inline">BENCHMARK</span> COMPARISON
        </button>

        {/* Live WebSocket Connection Status Badge */}
        <div
          className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[11px] font-bold border transition-colors ${
            isConnected
              ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
              : 'bg-rose-500/10 text-rose-400 border-rose-500/30'
          }`}
          title={isConnected ? 'Real-Time WebSocket Streaming Connected' : 'Disconnected from WebSocket Server'}
        >
          {isConnected ? (
            <>
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <Wifi className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">WS LIVE</span>
            </>
          ) : (
            <>
              <span className="w-2 h-2 rounded-full bg-rose-400" />
              <WifiOff className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">OFFLINE</span>
            </>
          )}
        </div>
      </div>
    </header>
  );
}
