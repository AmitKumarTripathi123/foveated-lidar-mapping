'use client';

import React from 'react';
import { useLidarStore } from '@/stores/useLidarStore';

export function CoordinateGizmo() {
  const layers = useLidarStore((state) => state.layers);

  if (!layers.coordinateAxes) return null;

  return (
    <group position={[-15, -15, -1.6]}>
      {/* X Axis: Red (East/Right) */}
      <mesh position={[2.5, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
        <cylinderGeometry args={[0.08, 0.08, 5, 8]} />
        <meshBasicMaterial color="#EF4444" />
      </mesh>
      <mesh position={[5.2, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
        <coneGeometry args={[0.25, 0.6, 8]} />
        <meshBasicMaterial color="#EF4444" />
      </mesh>

      {/* Y Axis: Green (Forward/North) */}
      <mesh position={[0, 2.5, 0]}>
        <cylinderGeometry args={[0.08, 0.08, 5, 8]} />
        <meshBasicMaterial color="#22C55E" />
      </mesh>
      <mesh position={[0, 5.2, 0]}>
        <coneGeometry args={[0.25, 0.6, 8]} />
        <meshBasicMaterial color="#22C55E" />
      </mesh>

      {/* Z Axis: Blue (Elevation/Up) */}
      <mesh position={[0, 0, 2.5]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.08, 0.08, 5, 8]} />
        <meshBasicMaterial color="#3B82F6" />
      </mesh>
      <mesh position={[0, 0, 5.2]} rotation={[Math.PI / 2, 0, 0]}>
        <coneGeometry args={[0.25, 0.6, 8]} />
        <meshBasicMaterial color="#3B82F6" />
      </mesh>
    </group>
  );
}
