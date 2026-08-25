'use client';

import React, { useEffect } from 'react';
import { useLidarStore } from '@/stores/useLidarStore';
import { fetchDatasets, loadDataset } from '@/lib/api';
import { Database, Disc } from 'lucide-react';

export function DatasetSelector() {
  const datasets = useLidarStore((state) => state.datasets);
  const setDatasets = useLidarStore((state) => state.setDatasets);
  const activeDatasetId = useLidarStore((state) => state.activeDatasetId);
  const setActiveDatasetId = useLidarStore((state) => state.setActiveDatasetId);

  useEffect(() => {
    fetchDatasets()
      .then((data) => {
        setDatasets(data);
      })
      .catch((err) => {
        console.warn('Backend offline or using fallback mock datasets:', err);
      });
  }, [setDatasets]);

  const handleSelect = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const id = e.target.value;
    setActiveDatasetId(id);
    try {
      await loadDataset(id);
    } catch (err) {
      console.error('Failed to load dataset:', err);
    }
  };

  return (
    <div className="bg-surface/80 backdrop-blur-md border border-border-color rounded-xl p-3 shadow-lg flex flex-col gap-2 text-white font-mono text-xs">
      <div className="flex items-center gap-1.5 text-gray-400">
        <Database className="w-3.5 h-3.5 text-brand-500" />
        <span className="font-bold tracking-wide">LiDAR Sequence:</span>
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
                {d.name} ({d.total_frames} frames)
              </option>
            ))
          ) : (
            <option value="sih_urban_demo_01">SIH Urban Driving Sequence 01 (100 frames)</option>
          )}
        </select>
        <Disc className="w-4 h-4 text-gray-400 absolute right-2.5 top-2.5 pointer-events-none" />
      </div>
    </div>
  );
}
