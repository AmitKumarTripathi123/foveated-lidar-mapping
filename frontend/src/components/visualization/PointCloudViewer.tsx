'use client';

import React, { useMemo, useEffect, useRef } from 'react';
import * as THREE from 'three';
import { useLidarStore } from '@/stores/useLidarStore';
import {
  getSemanticColor,
  getElevationColor,
  getTraversabilityColor,
} from '@/lib/semanticColors';

export function PointCloudViewer() {
  const points = useLidarStore((state) => state.points);
  const colorMode = useLidarStore((state) => state.colorMode);
  const layers = useLidarStore((state) => state.layers);
  const pointSize = useLidarStore((state) => state.pointSize);

  const geometryRef = useRef<THREE.BufferGeometry>(null);

  const { positions, colors } = useMemo(() => {
    const pointCount = points.length;
    const pos = new Float32Array(pointCount * 3);
    const col = new Float32Array(pointCount * 3);

    for (let i = 0; i < pointCount; i++) {
      const pt = points[i];
      const idx = i * 3;

      pos[idx] = pt.x;
      pos[idx + 1] = pt.y;
      pos[idx + 2] = pt.z;

      let rgb: [number, number, number] = [0.8, 0.8, 0.8];

      if (colorMode === 'semantic') {
        rgb = getSemanticColor(pt.semantic_class);
      } else if (colorMode === 'elevation') {
        rgb = getElevationColor(pt.z);
      } else if (colorMode === 'traversability') {
        rgb = getTraversabilityColor(pt.semantic_class === 1 ? 1.0 : (pt.semantic_class === 2 ? 0.4 : 0.0));
      } else if (colorMode === 'intensity') {
        const val = pt.intensity || 1.0;
        rgb = [val, val, val];
      }

      col[idx] = rgb[0];
      col[idx + 1] = rgb[1];
      col[idx + 2] = rgb[2];
    }

    return { positions: pos, colors: col };
  }, [points, colorMode]);

  useEffect(() => {
    if (geometryRef.current) {
      geometryRef.current.setAttribute(
        'position',
        new THREE.BufferAttribute(positions, 3)
      );
      geometryRef.current.setAttribute(
        'color',
        new THREE.BufferAttribute(colors, 3)
      );
      geometryRef.current.computeBoundingSphere();
    }
  }, [positions, colors]);

  if (!layers.semanticPoints && !layers.rawPoints) return null;

  return (
    <points>
      <bufferGeometry ref={geometryRef}>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
        />
        <bufferAttribute
          attach="attributes-color"
          args={[colors, 3]}
        />
      </bufferGeometry>
      <pointsMaterial
        size={pointSize}
        vertexColors
        sizeAttenuation={false}
        transparent
        opacity={0.9}
      />
    </points>
  );
}
