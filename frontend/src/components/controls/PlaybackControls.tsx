'use client';

import React from 'react';
import { useLidarStore } from '@/stores/useLidarStore';
import { useWebSocketStream } from '@/hooks/useWebSocketStream';
import {
  Play,
  Pause,
  Square,
  SkipBack,
  SkipForward,
  Gauge,
} from 'lucide-react';

export function PlaybackControls() {
  const playbackState = useLidarStore((state) => state.playbackState);
  const currentFrameIdx = useLidarStore((state) => state.currentFrameIdx);
  const totalFrames = useLidarStore((state) => state.totalFrames);
  const targetFps = useLidarStore((state) => state.targetFps);
  const setTargetFps = useLidarStore((state) => state.setTargetFps);

  const { play, pause, stop, seek, setFps } = useWebSocketStream();

  const handleFpsChange = (newFps: number) => {
    setTargetFps(newFps);
    setFps(newFps);
  };

  return (
    <div className="bg-surface/80 backdrop-blur-md border border-border-color rounded-xl p-3 shadow-lg flex flex-col gap-3">
      {/* Frame Scrubber */}
      <div className="flex flex-col gap-1">
        <div className="flex justify-between items-center text-xs font-mono">
          <span className="text-gray-400">Sequence Frame:</span>
          <span className="text-brand-500 font-bold">
            {currentFrameIdx + 1} <span className="text-gray-500">/ {totalFrames}</span>
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={Math.max(1, totalFrames - 1)}
          value={currentFrameIdx}
          onChange={(e) => seek(parseInt(e.target.value, 10))}
          className="w-full h-1.5 bg-surface-highlight rounded-lg appearance-none cursor-pointer accent-brand-500 hover:accent-brand-600 transition-all"
        />
      </div>

      {/* Buttons Bar */}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          {/* Step Back */}
          <button
            onClick={() => seek(Math.max(0, currentFrameIdx - 1))}
            className="p-2 rounded-lg bg-surface-highlight hover:bg-gray-700 text-gray-300 hover:text-white transition-colors"
            title="Previous Frame"
          >
            <SkipBack className="w-4 h-4" />
          </button>

          {/* Play / Pause Toggle */}
          {playbackState === 'running' ? (
            <button
              onClick={pause}
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-amber-600 hover:bg-amber-500 text-white font-mono text-xs font-bold shadow-md transition-colors"
            >
              <Pause className="w-4 h-4" />
              <span>PAUSE</span>
            </button>
          ) : (
            <button
              onClick={() => play(targetFps)}
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-brand-600 hover:bg-brand-500 text-white font-mono text-xs font-bold shadow-md transition-colors"
            >
              <Play className="w-4 h-4" />
              <span>STREAM</span>
            </button>
          )}

          {/* Step Forward */}
          <button
            onClick={() => seek(Math.min(totalFrames - 1, currentFrameIdx + 1))}
            className="p-2 rounded-lg bg-surface-highlight hover:bg-gray-700 text-gray-300 hover:text-white transition-colors"
            title="Next Frame"
          >
            <SkipForward className="w-4 h-4" />
          </button>

          {/* Stop / Reset */}
          <button
            onClick={stop}
            className="p-2 rounded-lg bg-surface-highlight hover:bg-red-900/50 hover:text-red-400 text-gray-400 transition-colors"
            title="Stop / Reset to Frame 0"
          >
            <Square className="w-4 h-4" />
          </button>
        </div>

        {/* FPS Presets */}
        <div className="flex items-center gap-1 bg-surface-highlight/70 px-2 py-1 rounded-lg border border-border-color text-xs font-mono">
          <Gauge className="w-3.5 h-3.5 text-gray-400" />
          {[5, 10, 20].map((rate) => (
            <button
              key={rate}
              onClick={() => handleFpsChange(rate)}
              className={`px-1.5 py-0.5 rounded text-[11px] font-bold transition-colors ${
                targetFps === rate
                  ? 'bg-brand-600 text-white'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
            >
              {rate}Hz
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
