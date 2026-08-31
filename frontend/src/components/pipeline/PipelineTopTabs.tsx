'use client';

import React from 'react';
import { useLidarStore } from '@/stores/useLidarStore';
import { PipelineStageId } from '@/types/lidar';

export function PipelineTopTabs() {
  const activePipelineStage = useLidarStore((state) => state.activePipelineStage);
  const applyPipelinePreset = useLidarStore((state) => state.applyPipelinePreset);
  const setGridDisplayMode = useLidarStore((state) => state.setGridDisplayMode);
  const setGridRenderStyle = useLidarStore((state) => state.setGridRenderStyle);

  const tabs: { id: PipelineStageId; num: string; label: string; tag: string }[] = [
    { id: 'raw', num: '1.', label: 'RAW LIDAR', tag: 'INTENSITY' },
    { id: 'semantic', num: '2.', label: 'AI SEMANTIC', tag: 'SEGMENTATION' },
    { id: 'foveation', num: '3.', label: 'FOVEATED', tag: 'MULTI-RES' },
    { id: 'data', num: '4.', label: 'ELEVATION', tag: 'HEIGHT MAP' },
    { id: 'variable_grid', num: '5.', label: 'FOV + SEMANTIC', tag: 'OVERLAY' },
    { id: 'elevation_25d', num: '6.', label: '2.5D ELEVATION MAP', tag: 'OUTPUT' },
  ];

  const handleTabClick = (tabId: PipelineStageId) => {
    applyPipelinePreset(tabId);
    if (tabId === 'elevation_25d' || tabId === 'foveation') {
      setGridDisplayMode('grid');
      setGridRenderStyle('top_down_2d');
    } else if (tabId === 'raw' || tabId === 'semantic' || tabId === 'data') {
      setGridDisplayMode('points');
      setGridRenderStyle('top_down_2d');
    } else if (tabId === 'variable_grid') {
      setGridDisplayMode('both');
      setGridRenderStyle('top_down_2d');
    }
  };

  return (
    <div className="flex items-center gap-2 p-2 bg-[#080B14] border-b border-[#1E293B] overflow-x-auto select-none font-mono">
      {tabs.map((t) => {
        const isActive = activePipelineStage === t.id || (activePipelineStage === 'elevation_25d' && t.id === 'elevation_25d');

        return (
          <button
            key={t.id}
            onClick={() => handleTabClick(t.id)}
            className={`flex-1 min-w-[130px] flex flex-col items-center justify-center py-2 px-3 rounded-xl border transition-all ${
              isActive
                ? 'bg-[#0284C7] border-sky-400 text-white shadow-lg shadow-sky-950/60 font-bold'
                : 'bg-[#0B0F19] border-[#1E293B] text-gray-400 hover:text-white hover:bg-[#111827]'
            }`}
          >
            <div className="flex items-center gap-1.5 text-xs">
              <span className={isActive ? 'text-sky-200' : 'text-gray-500'}>{t.num}</span>
              <span>{t.label}</span>
            </div>
            <div
              className={`text-[9px] uppercase tracking-wider ${
                isActive ? 'text-sky-100 font-semibold' : 'text-gray-500'
              }`}
            >
              {t.tag}
            </div>
          </button>
        );
      })}
    </div>
  );
}
