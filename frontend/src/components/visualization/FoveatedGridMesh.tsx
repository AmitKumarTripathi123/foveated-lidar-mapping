'use client';

import React, { useMemo, useRef, useEffect } from 'react';
import * as THREE from 'three';
import { useLidarStore } from '@/stores/useLidarStore';
import {
  getSemanticColor,
  getElevationColor,
  getTraversabilityColor,
} from '@/lib/semanticColors';

const tempObject = new THREE.Object3D();
const tempColor = new THREE.Color();

export function FoveatedGridMesh() {
  const cells = useLidarStore((state) => state.cells);
  const layers = useLidarStore((state) => state.layers);
  const colorMode = useLidarStore((state) => state.colorMode);
  const gridOpacity = useLidarStore((state) => state.gridOpacity);
  const setSelectedCell = useLidarStore((state) => state.setSelectedCell);
  const setHoveredCell = useLidarStore((state) => state.setHoveredCell);

  const instancedMeshRef = useRef<THREE.InstancedMesh>(null);
  const wireframeMeshRef = useRef<THREE.InstancedMesh>(null);

  // Group cells or render up to 12,000 active cells via InstancedMesh
  const displayCells = useMemo(() => {
    return cells.slice(0, 12000);
  }, [cells]);

  useEffect(() => {
    if (!instancedMeshRef.current || displayCells.length === 0) return;

    for (let i = 0; i < displayCells.length; i++) {
      const cell = displayCells[i];
      const res = cell.resolution;

      // Position box at cell center
      const height = Math.max(0.08, res * 0.4);
      tempObject.position.set(cell.x, cell.y, cell.elevation);
      tempObject.scale.set(res * 0.96, res * 0.96, height);
      tempObject.updateMatrix();

      instancedMeshRef.current.setMatrixAt(i, tempObject.matrix);
      if (wireframeMeshRef.current) {
        wireframeMeshRef.current.setMatrixAt(i, tempObject.matrix);
      }

      let rgb: [number, number, number] = [0.2, 0.6, 0.9];
      if (colorMode === 'semantic') {
        rgb = getSemanticColor(cell.semantic_class);
      } else if (colorMode === 'elevation') {
        rgb = getElevationColor(cell.elevation);
      } else if (colorMode === 'traversability') {
        rgb = getTraversabilityColor(cell.traversability);
      }

      tempColor.setRGB(rgb[0], rgb[1], rgb[2]);
      instancedMeshRef.current.setColorAt(i, tempColor);
    }

    instancedMeshRef.current.instanceMatrix.needsUpdate = true;
    if (instancedMeshRef.current.instanceColor) {
      instancedMeshRef.current.instanceColor.needsUpdate = true;
    }

    if (wireframeMeshRef.current) {
      wireframeMeshRef.current.instanceMatrix.needsUpdate = true;
    }
  }, [displayCells, colorMode]);

  if (!layers.foveatedGrid && !layers.traversabilityMap && !layers.adaptiveGridWireframe) {
    return null;
  }

  return (
    <group>
      {/* 1. Solid Semi-Transparent Cell Blocks */}
      <instancedMesh
        ref={instancedMeshRef}
        args={[undefined, undefined, displayCells.length]}
        onClick={(e) => {
          e.stopPropagation();
          if (e.instanceId !== undefined && displayCells[e.instanceId]) {
            setSelectedCell(displayCells[e.instanceId]);
          }
        }}
        onPointerMove={(e) => {
          e.stopPropagation();
          if (e.instanceId !== undefined && displayCells[e.instanceId]) {
            setHoveredCell(displayCells[e.instanceId]);
          }
        }}
        onPointerOut={() => {
          setHoveredCell(null);
        }}
      >
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial
          transparent
          opacity={gridOpacity}
          roughness={0.35}
          metalness={0.15}
        />
      </instancedMesh>

      {/* 2. Adaptive Variable-Resolution Wireframe Grid Outlines */}
      {layers.adaptiveGridWireframe && (
        <instancedMesh
          ref={wireframeMeshRef}
          args={[undefined, undefined, displayCells.length]}
        >
          <boxGeometry args={[1, 1, 1]} />
          <meshBasicMaterial
            color="#38BDF8"
            wireframe
            transparent
            opacity={0.35}
          />
        </instancedMesh>
      )}
    </group>
  );
}
