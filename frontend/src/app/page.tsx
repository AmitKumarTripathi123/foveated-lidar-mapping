'use client';

import React, { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { Header } from '@/components/layout/Header';
import { Sidebar } from '@/components/layout/Sidebar';
import { StatusBar } from '@/components/layout/StatusBar';
import { MappingAnalytics } from '@/components/analytics/MappingAnalytics';
import { PipelineTopTabs } from '@/components/pipeline/PipelineTopTabs';
import { PresentationMode } from '@/components/presentation/PresentationMode';
import { ComparisonModal } from '@/components/comparison/ComparisonModal';
import { useWebSocketStream } from '@/hooks/useWebSocketStream';

// Dynamically import 3D WebGL Canvas to prevent server-side hydration mismatches
const LidarCanvas = dynamic(
  () => import('@/components/visualization/LidarCanvas').then((mod) => mod.LidarCanvas),
  {
    ssr: false,
    loading: () => (
      <div className="w-full h-full bg-[#070A12] flex items-center justify-center text-sky-400 font-mono text-xs select-none">
        <span className="w-2 h-2 rounded-full bg-sky-400 animate-ping mr-2.5" />
        <span>INITIALIZING 3D AUTONOMOUS WORKSTATION...</span>
      </div>
    ),
  }
);

export default function DashboardPage() {
  const [mounted, setMounted] = useState(false);

  // Initialize WebSocket real-time connection hook
  useWebSocketStream();

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-[#070A12] select-none font-mono">
      {/* 1. System Header Bar */}
      <Header />

      {/* 2. Main Middle Section: 3-Column Autonomous Workstation */}
      <div className="flex flex-1 overflow-hidden relative">
        {/* Left Panel: Research Pipeline + Sequence Control */}
        <Sidebar />

        {/* Center Main Viewport: Live 3D/2.5D Interactive WebGL Engine */}
        <main className="flex-1 h-full relative flex flex-col overflow-hidden bg-[#070A12]">
          {/* Top 6-Stage Pipeline Navigation Tabs */}
          <PipelineTopTabs />

          {/* Central 3D WebGL Canvas */}
          <div className="flex-1 relative overflow-hidden">
            {mounted ? <LidarCanvas /> : null}

            {/* Presentation Mode Guided Overlay */}
            <PresentationMode />
          </div>
        </main>

        {/* Right Panel: AI Telemetry, Semantic Legend & 2.5D Stats */}
        <MappingAnalytics />
      </div>

      {/* 3. Bottom Live Telemetry & HOW TO READ THIS 2.5D MAP Banner */}
      <StatusBar />

      {/* 4. Quantitative Benchmark Modal */}
      <ComparisonModal />
    </div>
  );
}
