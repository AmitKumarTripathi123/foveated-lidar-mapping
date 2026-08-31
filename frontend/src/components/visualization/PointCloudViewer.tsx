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
  const gridDisplayMode = useLidarStore((state) => state.gridDisplayMode);
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

      let rgb: [number, number, number] = [0.13, 0.77, 0.37];

      if (colorMode === 'semantic') {
        rgb = getSemanticColor(pt.semantic_class);
      } else if (colorMode === 'elevation') {
        rgb = getElevationColor(pt.z);
      } else if (colorMode === 'traversability') {
        rgb = getTraversabilityColor(pt.semantic_class === 0 ? 1.0 : (pt.semantic_class === 1 ? 0.35 : 0.0));
      } else if (colorMode === 'intensity') {
        const val = pt.intensity || 0.85;
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

  // If in pure GRID MAP mode, do not render point cloud so the clean 2.5D grid engine is highlighted
  if (gridDisplayMode === 'grid') return null;
  if (!layers.semanticPoints && !layers.rawPoints && gridDisplayMode !== 'both') return null;

  const pointOpacity = gridDisplayMode === 'both' ? 0.38 : 0.92;
  const renderedPointSize = gridDisplayMode === 'both' ? Math.max(2.0, pointSize * 0.8) : pointSize;

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
        size={renderedPointSize}
        vertexColors
        sizeAttenuation={false}
        transparent
        opacity={pointOpacity}
      />
    </points>
  );
}
