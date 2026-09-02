'use client';

import React from 'react';
import { Html } from '@react-three/drei';
import { useLidarStore } from '@/stores/useLidarStore';
import { SEMANTIC_CLASSES } from '@/lib/semanticColors';

export function BoundingBoxOverlay() {
  const boundingBoxes = useLidarStore((state) => state.boundingBoxes);
  const layers = useLidarStore((state) => state.layers);
  const setSelectedBox = useLidarStore((state) => state.setSelectedBox);

  if (!layers.boundingBoxes || boundingBoxes.length === 0) return null;

  return (
    <group>
      {boundingBoxes.map((box) => {
        const [cx, cy, cz] = box.center;
        const [sx, sy, sz] = box.size;

        // Class color
        const color = box.class_name === 'Vehicle' 
          ? SEMANTIC_CLASSES[4].hex 
          : (box.class_name === 'Pedestrian' ? SEMANTIC_CLASSES[5].hex : '#06B6D4');

        return (
          <group
            key={box.id}
            position={[cx, cy, cz]}
            rotation={[0, 0, box.rotation_yaw]}
            onClick={(e) => {
              e.stopPropagation();
              setSelectedBox(box);
            }}
          >
            {/* Box Wireframe */}
            <mesh>
              <boxGeometry args={[sx, sy, sz]} />
              <meshBasicMaterial
                color={color}
                wireframe
                transparent
                opacity={0.85}
              />
            </mesh>

            {/* Translucent Solid Fill */}
            <mesh>
              <boxGeometry args={[sx, sy, sz]} />
              <meshBasicMaterial
                color={color}
                transparent
                opacity={0.15}
              />
            </mesh>

            {/* 3D Compact Label */}
            <Html position={[0, 0, sz / 2 + 0.35]} center>
              <div
                className="px-1.5 py-0.5 rounded text-[9px] font-mono font-bold tracking-tight shadow-md border whitespace-nowrap cursor-pointer hover:scale-105 transition-transform"
                style={{
                  backgroundColor: '#0a0e18dd',
                  borderColor: color,
                  color: color,
                }}
              >
                {box.class_name.replace(/^Dynamic Object \((.+)\)$/, '$1').toUpperCase()} {Math.round(box.confidence * 100)}%
              </div>
            </Html>
          </group>
        );
      })}
    </group>
  );
}
