'use client';

import React from 'react';
import { Html } from '@react-three/drei';
import { useLidarStore } from '@/stores/useLidarStore';

export function EgoVehicleMarker() {
  const layers = useLidarStore((state) => state.layers);

  if (!layers.egoVehicle) return null;

  return (
    <group position={[0, 0, -1.5]}>
      {/* Vehicle Footprint (4.5m x 1.8m x 1.4m standard autonomous car) */}
      <mesh position={[0, 0, 0.7]}>
        <boxGeometry args={[1.8, 4.4, 1.3]} />
        <meshStandardMaterial
          color="#0284C7"
          transparent
          opacity={0.35}
          wireframe={false}
        />
      </mesh>

      {/* Vehicle Wireframe Cage */}
      <mesh position={[0, 0, 0.7]}>
        <boxGeometry args={[1.82, 4.42, 1.32]} />
        <meshBasicMaterial color="#38BDF8" wireframe />
      </mesh>

      {/* Forward Heading Direction Arrow */}
      <mesh position={[0, 3.2, 0.7]} rotation={[0, 0, -Math.PI / 2]}>
        <coneGeometry args={[0.5, 1.4, 16]} />
        <meshBasicMaterial color="#22C55E" />
      </mesh>

      {/* LiDAR Sensor Puck Marker atop roof */}
      <mesh position={[0, 0.2, 1.45]}>
        <cylinderGeometry args={[0.2, 0.2, 0.25, 16]} />
        <meshStandardMaterial color="#F59E0B" emissive="#F59E0B" emissiveIntensity={0.6} />
      </mesh>

      {/* Origin Ring */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.8, 0.95, 32]} />
        <meshBasicMaterial color="#38BDF8" transparent opacity={0.6} />
      </mesh>

      {/* Sensor Label Tag */}
      <Html position={[0, -2.6, 1.8]} center>
        <div className="px-2 py-0.5 rounded bg-surface/90 border border-sky-500/50 text-[9px] font-mono font-bold text-sky-400 backdrop-blur-md whitespace-nowrap shadow-md">
          EGO VEHICLE (0, 0, 0)
        </div>
      </Html>
    </group>
  );
}
