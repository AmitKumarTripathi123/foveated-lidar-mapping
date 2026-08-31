'use client';

import React, { useEffect } from 'react';
import { useLidarStore } from '@/stores/useLidarStore';
import { fetchDatasets, switchDataset } from '@/lib/api';
import { Database, Disc } from 'lucide-react';

export function DatasetSelector() {
  const datasets = useLidarStore((state) => state.datasets);
  const setDatasets = useLidarStore((state) => state.setDatasets);
  const activeDatasetId = useLidarStore((state) => state.activeDatasetId);
  const setActiveDatasetId = useLidarStore((state) => state.setActiveDatasetId);

  useEffect(() => {
    async function load() {
      try {
        const data = await fetchDatasets();
        if (data && data.datasets) {
          setDatasets(data.datasets);
        }
      } catch (err) {
        console.error('Failed to load dataset list:', err);
      }
    }
    load();
  }, [setDatasets]);

  const handleSelect = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newId = e.target.value;
    setActiveDatasetId(newId);
    try {
      await switchDataset(newId);
    } catch (err) {
      console.error('Failed to switch dataset sequence:', err);
    }
  };

  return (
    <div className="space-y-2 font-mono">
      <div className="flex items-center justify-between text-xs text-gray-400 font-bold uppercase tracking-wider">
        <div className="flex items-center gap-1.5">
          <Database className="w-3.5 h-3.5 text-brand-500" />
          <span>LiDAR Sequence:</span>
        </div>
      </div>

      <div className="relative">
        <select
          value={activeDatasetId}
          onChange={handleSelect}
          className="w-full bg-surface-highlight border border-border-color rounded-lg px-3 py-2 text-gray-200 text-xs font-mono focus:outline-none focus:border-brand-500 appearance-none cursor-pointer pr-8"
        >
          {datasets.length > 0 ? (
            datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name.replace(/^SIH\s+/i, '')} ({d.total_frames} frames)
              </option>
            ))
          ) : (
            <option value="sih_urban_demo_01">Urban Driving Sequence 01 (100 frames)</option>
          )}
        </select>
        <Disc className="w-4 h-4 text-gray-400 absolute right-2.5 top-2.5 pointer-events-none" />
      </div>
    </div>
  );
}
