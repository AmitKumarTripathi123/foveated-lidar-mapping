'use client';

import React from 'react';
import { FOVEATED_ZONE_COLORS } from '@/lib/semanticColors';
import { Target, Info, ShieldCheck } from 'lucide-react';

export function ResolutionConfig() {
  return (
    <div className="bg-[#0B0F19]/90 backdrop-blur-md border border-border-color rounded-xl p-3 shadow-lg flex flex-col gap-2.5 text-white font-mono text-xs">
      <div className="flex items-center justify-between text-gray-400">
        <div className="flex items-center gap-1.5">
          <Target className="w-3.5 h-3.5 text-sky-400" />
          <span className="font-bold tracking-wide">3-Zone Foveation:</span>
        </div>
        <span className="text-[9px] text-sky-400 bg-sky-500/10 px-1.5 py-0.5 rounded border border-sky-500/30 font-bold">
          Hesai 40-Beam
        </span>
      </div>

      <div className="flex flex-col gap-1.5">
        {Object.entries(FOVEATED_ZONE_COLORS).map(([id, zone]) => (
          <div
            key={id}
            className="flex items-center justify-between p-2 rounded-lg bg-surface-highlight/40 border border-border-color/40"
          >
            <div className="flex items-center gap-2">
              <span
                className="w-2.5 h-2.5 rounded-full"
                style={{ backgroundColor: zone.hex }}
              />
              <div className="flex flex-col">
                <span className="text-gray-200 text-[11px] font-bold">{zone.name}</span>
                <span className="text-gray-400 text-[9px]">Radius: {zone.radius}</span>
              </div>
            </div>
            <span className="font-bold text-sky-400 text-[11px]">{zone.resolution}</span>
          </div>
        ))}
      </div>

      <div className="flex items-start gap-1.5 text-[10px] text-gray-300 bg-surface-highlight/30 p-2 rounded-lg border border-border-color/40">
        <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
        <span>
          5cm near-field grid cells preserve exact curb &amp; obstacle boundaries, while 50cm far-field reduces theoretical spatial cell capacity by 97.29% and frame occupied cells by ~80%.
        </span>
      </div>
    </div>
  );
}
