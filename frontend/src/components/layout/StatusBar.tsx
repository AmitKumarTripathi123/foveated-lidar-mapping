'use client';

import React from 'react';
import { LiveMetricsCard } from '../metrics/LiveMetricsCard';

export function StatusBar() {
  return (
    <footer className="w-full select-none z-20">
      <LiveMetricsCard />
    </footer>
  );
}
