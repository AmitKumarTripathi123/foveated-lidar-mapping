'use client';

import React from 'react';
import { Html } from '@react-three/drei';
import { useLidarStore } from '@/stores/useLidarStore';
import { FOVEATED_ZONE_COLORS } from '@/lib/semanticColors';

const ZONES = [
  {
    id: 0,
    radius: 10,
    title: 'ZONE 0 — FOVEAL (NEAR)',
    resolution: 'Δs = 5 cm (0.05m)',
    color: FOVEATED_ZONE_COLORS[0].hex,
    desc: 'Dense precision obstacle & curb grid',
  },
  {
    id: 1,
    radius: 40,
    title: 'ZONE 1 — INTERMEDIATE (MID)',
    resolution: 'Δs = 15 cm (0.15m)',
    color: FOVEATED_ZONE_COLORS[1].hex,
    desc: 'Lane & dynamic vehicle corridor',
  },
  {
    id: 2,
    radius: 90,
    title: 'ZONE 2 — PERIPHERY (FAR)',
    resolution: 'Δs = 50 cm (0.50m)',
    color: FOVEATED_ZONE_COLORS[2].hex,
    desc: 'Macro corridor tracking (~85% memory savings)',
  },
];

export function ConcentricRings() {
  const layers = useLidarStore((state) => state.layers);

  if (!layers.zoneRings) return null;

  return (
    <group position={[0, 0, -1.6]}>
      {ZONES.map((zone, idx) => (
        <group key={zone.id}>
          {/* Outer Boundary Ring */}
          <mesh rotation={[-Math.PI / 2, 0, 0]}>
            <ringGeometry args={[zone.radius - 0.15, zone.radius + 0.15, 96]} />
            <meshBasicMaterial
              color={zone.color}
              transparent
              opacity={0.45}
              side={2}
            />
          </mesh>

          {/* Semi-transparent Zone Disc Area Indicator */}
          <mesh rotation={[-Math.PI / 2, 0, 0]}>
            <ringGeometry
              args={[idx === 0 ? 0 : ZONES[idx - 1].radius, zone.radius - 0.18, 64]}
            />
            <meshBasicMaterial
              color={zone.color}
              transparent
              opacity={0.025}
              side={2}
            />
          </mesh>

          {/* Spatial Resolution Scientific HUD Tag */}
          <Html
            position={[
              zone.radius * 0.72 * (idx % 2 === 0 ? 1 : -1),
              zone.radius * 0.72,
              0.4,
            ]}
            center
          >
            <div
              className="px-2.5 py-1 rounded-md text-[10px] font-mono tracking-tight shadow-xl border backdrop-blur-md whitespace-nowrap pointer-events-none flex flex-col items-start gap-0.5"
              style={{
                backgroundColor: '#0A0E18ee',
                borderColor: `${zone.color}77`,
                boxShadow: `0 0 12px ${zone.color}22`,
              }}
            >
              <div className="flex items-center gap-1.5 font-bold" style={{ color: zone.color }}>
                <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: zone.color }} />
                <span>{zone.title}</span>
              </div>
              <div className="text-[9px] text-gray-300 font-semibold">
                {zone.resolution}
              </div>
            </div>
          </Html>
        </group>
      ))}
    </group>
  );
}
