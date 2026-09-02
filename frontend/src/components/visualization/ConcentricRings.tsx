'use client';

import React from 'react';
import * as THREE from 'three';
import { Html } from '@react-three/drei';
import { useLidarStore } from '@/stores/useLidarStore';
import { FOVEATED_ZONE_COLORS } from '@/lib/semanticColors';

interface RingData {
  radius: number;
  label: string;
  zone: string;
  res: string;
  color: string;
  dash: boolean;
}

const RINGS: RingData[] = [
  {
    radius: 10,
    label: '10m',
    zone: 'ZONE 0 — FOVEAL / NEAR',
    res: '5 cm cells (0.05m)',
    color: '#38BDF8', // Cyan / Near-field
    dash: false,
  },
  {
    radius: 50,
    label: '50m',
    zone: 'ZONE 1 — INTERMEDIATE',
    res: '~25 cm cells (0.25m)',
    color: '#F59E0B', // Amber / Mid-field
    dash: true,
  },
  {
    radius: 100,
    label: '100m',
    zone: 'ZONE 2 — PERIPHERAL / FAR',
    res: '50 cm cells (0.50m)',
    color: '#A855F7', // Purple / Far-field boundary
    dash: true,
  },
];

export function ConcentricRings() {
  const layers = useLidarStore((state) => state.layers);

  if (!layers.zoneRings) return null;

  return (
    <group position={[0, 0, -1.55]}>
      {RINGS.map((ring) => (
        <group key={ring.radius}>
          {/* Circular Ground Ring Line */}
          <lineLoop>
            <bufferGeometry
              attach="geometry"
              onUpdate={(geo) => {
                const points: number[] = [];
                const segments = 128;
                for (let i = 0; i <= segments; i++) {
                  const theta = (i / segments) * Math.PI * 2;
                  points.push(
                    Math.cos(theta) * ring.radius,
                    Math.sin(theta) * ring.radius,
                    0
                  );
                }
                geo.setAttribute(
                  'position',
                  new THREE.Float32BufferAttribute(points, 3)
                );
              }}
            />
            <lineBasicMaterial
              attach="material"
              color={ring.color}
              transparent
              opacity={ring.radius === 100 ? 0.45 : 0.35}
              linewidth={1}
            />
          </lineLoop>

          {/* Compact Non-Blocking Perimeter HUD Tag */}
          <Html
            position={[
              ring.radius * 0.95,
              ring.radius * 0.3,
              0.2,
            ]}
            center
            distanceFactor={55}
          >
            <div className="bg-[#0A0E18]/85 backdrop-blur-md border border-white/20 px-2 py-0.5 rounded text-[8.5px] font-mono text-white shadow-xl pointer-events-none select-none flex items-center gap-1.5 whitespace-nowrap">
              <span
                className="w-1.5 h-1.5 rounded-full inline-block"
                style={{ backgroundColor: ring.color }}
              />
              <span className="font-bold" style={{ color: ring.color }}>{ring.label}</span>
              <span className="text-gray-300">({ring.res})</span>
            </div>
          </Html>
        </group>
      ))}
    </group>
  );
}

// Inline helper for Float32BufferAttribute in Three.js JSX
import { Float32BufferAttribute } from 'three';
