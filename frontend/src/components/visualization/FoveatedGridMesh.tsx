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
  const viewMode3D = useLidarStore((state) => state.viewMode3D);
  const gridOpacity = useLidarStore((state) => state.gridOpacity);
  const setSelectedCell = useLidarStore((state) => state.setSelectedCell);
  const setHoveredCell = useLidarStore((state) => state.setHoveredCell);

  const instancedMeshRef = useRef<THREE.InstancedMesh>(null);
  const wireframeMeshRef = useRef<THREE.InstancedMesh>(null);

  // Group cells or render up to 15,000 active cells via InstancedMesh
  const displayCells = useMemo(() => {
    return cells.slice(0, 15000);
  }, [cells]);

  useEffect(() => {
    if (!instancedMeshRef.current || displayCells.length === 0) return;

    const zBase = -1.65; // Standard ground datum height

    for (let i = 0; i < displayCells.length; i++) {
      const cell = displayCells[i];
      const res = cell.resolution;

      // True 2.5D Elevation Column Height Calculation
      let height = 0.12;
      let centerZ = cell.elevation;

      if (viewMode3D === 'foveated_elevation' || colorMode === 'elevation') {
        // In 2.5D elevation mode, extrude column from ground datum (-1.65m) up to cell elevation
        const columnHeight = Math.max(0.12, cell.elevation - zBase);
        height = columnHeight;
        centerZ = zBase + height / 2;
      } else {
        // In foveated voxel mode, height scaled according to distance resolution
        height = Math.max(0.12, res * 0.45);
        centerZ = cell.elevation;
      }

      tempObject.position.set(cell.x, cell.y, centerZ);
      tempObject.scale.set(res * 0.94, res * 0.94, height);
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
      } else {
        rgb = getSemanticColor(cell.semantic_class);
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
  }, [displayCells, colorMode, viewMode3D]);

  if (!layers.foveatedGrid && !layers.traversabilityMap && !layers.adaptiveGridWireframe) {
    return null;
  }

  const effectiveOpacity = viewMode3D === 'foveated_elevation' ? 0.92 : gridOpacity;

  return (
    <group>
      {/* 1. Solid 2.5D Elevation & Foveated Cell Blocks */}
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
          opacity={effectiveOpacity}
          roughness={0.3}
          metalness={0.1}
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
            color={viewMode3D === 'foveated_elevation' ? '#60A5FA' : '#38BDF8'}
            wireframe
            transparent
            opacity={viewMode3D === 'foveated_elevation' ? 0.45 : 0.3}
          />
        </instancedMesh>
      )}
    </group>
  );
}
