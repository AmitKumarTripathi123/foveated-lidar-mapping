'use client';

import React from 'react';
import { ResearchPipeline } from '../pipeline/ResearchPipeline';
import { DatasetSelector } from '../controls/DatasetSelector';
import { PlaybackControls } from '../controls/PlaybackControls';
import { LayerSelector } from '../controls/LayerSelector';
import { ResolutionConfig } from '../controls/ResolutionConfig';

export function Sidebar() {
  return (
    <aside className="w-80 bg-[#0C101A] border-r border-border-color flex flex-col h-full overflow-y-auto p-3 gap-3 select-none">
      {/* 1. Research Pipeline Control (Top Hero Widget) */}
      <ResearchPipeline />

      {/* 2. Dataset & Sequence Selector */}
      <DatasetSelector />

      {/* 3. Sequence Playback Controls */}
      <PlaybackControls />

      {/* 4. Layer Visibility & Colormaps */}
      <LayerSelector />

      {/* 5. Foveated Spatial Resolution Specs */}
      <ResolutionConfig />
    </aside>
  );
}
