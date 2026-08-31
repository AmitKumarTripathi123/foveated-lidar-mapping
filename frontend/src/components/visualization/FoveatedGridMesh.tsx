'use client';

import React, { useMemo, useRef, useEffect, useState } from 'react';
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
  const gridRenderStyle = useLidarStore((state) => state.gridRenderStyle);
  const gridOpacity = useLidarStore((state) => state.gridOpacity);
  const selectedCell = useLidarStore((state) => state.selectedCell);
  const setSelectedCell = useLidarStore((state) => state.setSelectedCell);
  const setHoveredCell = useLidarStore((state) => state.setHoveredCell);

  const instancedMeshRef = useRef<THREE.InstancedMesh>(null);
  const wireframeMeshRef = useRef<THREE.InstancedMesh>(null);
  const highlightMeshRef = useRef<THREE.Mesh>(null);

  // Take up to 20,000 structured variable grid cells
  const displayCells = useMemo(() => {
    return cells.slice(0, 20000);
  }, [cells]);

  useEffect(() => {
    if (!instancedMeshRef.current || displayCells.length === 0) return;

    const zBase = -1.65; // Ground reference datum

    for (let i = 0; i < displayCells.length; i++) {
      const cell = displayCells[i];
      const res = cell.cellSize || cell.resolution || 0.05;

      let height = 0.06;
      let centerZ = -1.58;

      if (gridRenderStyle === 'extruded_3d' || viewMode3D === 'foveated_elevation' && colorMode === 'elevation') {
        // 3D Column Extrusion
        height = Math.max(0.10, cell.elevation - zBase);
        centerZ = zBase + height / 2;
      } else {
        // 2D Planar XY Grid Tile with slight elevation offset
        height = 0.06;
        centerZ = Math.max(-1.62, cell.elevation * 0.15 - 1.58);
      }

      // Exact square cell dimensions: res x res
      tempObject.position.set(cell.x, cell.y, centerZ);
      tempObject.scale.set(res * 0.94, res * 0.94, height);
      tempObject.updateMatrix();

      instancedMeshRef.current.setMatrixAt(i, tempObject.matrix);
      if (wireframeMeshRef.current) {
        wireframeMeshRef.current.setMatrixAt(i, tempObject.matrix);
      }

      // Semantic or Elevation coloring
      let rgb: [number, number, number] = [0.13, 0.77, 0.37];
      if (colorMode === 'elevation') {
        rgb = getElevationColor(cell.elevation);
      } else if (colorMode === 'traversability') {
        rgb = getTraversabilityColor(cell.traversability);
      } else {
        rgb = getSemanticColor(cell.semantic_class);
      }

      // Subtle brightness modulation by elevation for 2D map depth perception
      if (colorMode === 'semantic') {
        const heightMod = Math.min(1.25, Math.max(0.75, 1.0 + (cell.elevation + 1.6) * 0.15));
        tempColor.setRGB(
          Math.min(1.0, rgb[0] * heightMod),
          Math.min(1.0, rgb[1] * heightMod),
          Math.min(1.0, rgb[2] * heightMod)
        );
      } else {
        tempColor.setRGB(rgb[0], rgb[1], rgb[2]);
      }

      instancedMeshRef.current.setColorAt(i, tempColor);
    }

    instancedMeshRef.current.instanceMatrix.needsUpdate = true;
    if (instancedMeshRef.current.instanceColor) {
      instancedMeshRef.current.instanceColor.needsUpdate = true;
    }

    if (wireframeMeshRef.current) {
      wireframeMeshRef.current.instanceMatrix.needsUpdate = true;
    }
  }, [displayCells, colorMode, viewMode3D, gridRenderStyle]);

  if (!layers.foveatedGrid && !layers.traversabilityMap && !layers.adaptiveGridWireframe) {
    return null;
  }

  return (
    <group>
      {/* 1. Solid Geometric 2.5D Grid Cell Tiles */}
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
          roughness={0.25}
          metalness={0.08}
        />
      </instancedMesh>

      {/* 2. Structured Variable-Resolution Cell Outlines (Grid Boundaries) */}
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

      {/* 3. Selected Cell Glowing Border Highlight */}
      {selectedCell && (
        <mesh
          position={[selectedCell.x, selectedCell.y, -1.5]}
          scale={[selectedCell.cellSize * 1.05, selectedCell.cellSize * 1.05, 0.2]}
        >
          <boxGeometry args={[1, 1, 1]} />
          <meshBasicMaterial color="#FFFFFF" wireframe />
        </mesh>
      )}
    </group>
  );
}
