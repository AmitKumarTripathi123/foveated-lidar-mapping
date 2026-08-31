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
  const gridRenderStyle = useLidarStore((state) => state.gridRenderStyle);
  const gridOpacity = useLidarStore((state) => state.gridOpacity);
  const selectedCell = useLidarStore((state) => state.selectedCell);
  const setSelectedCell = useLidarStore((state) => state.setSelectedCell);
  const setHoveredCell = useLidarStore((state) => state.setHoveredCell);

  const instancedMeshRef = useRef<THREE.InstancedMesh>(null);
  const wireframeMeshRef = useRef<THREE.InstancedMesh>(null);

  const displayCells = useMemo(() => {
    return cells.slice(0, 25000);
  }, [cells]);

  useEffect(() => {
    if (!instancedMeshRef.current || displayCells.length === 0) return;

    const zBase = -1.65; // Ground reference datum

    for (let i = 0; i < displayCells.length; i++) {
      const cell = displayCells[i];
      const res = cell.cellSize || cell.resolution || 0.10;

      let height = 0.08;
      let centerZ = -1.56;

      if (gridRenderStyle === 'extruded_3d' || viewMode3D === 'foveated_elevation' && colorMode === 'elevation') {
        // 3D Column Extrusion
        height = Math.max(0.12, cell.elevation - zBase);
        centerZ = zBase + height / 2;
      } else {
        // 2.5D Planar Grid Tile on Ground
        height = 0.08;
        centerZ = -1.56;
      }

      // Exact square cell dimensions: res x res
      tempObject.position.set(cell.x, cell.y, centerZ);
      tempObject.scale.set(res * 0.94, res * 0.94, height);
      tempObject.updateMatrix();

      instancedMeshRef.current.setMatrixAt(i, tempObject.matrix);
      if (wireframeMeshRef.current) {
        wireframeMeshRef.current.setMatrixAt(i, tempObject.matrix);
      }

      // Dynamic Semantic / Zone / Elevation coloring
      if (colorMode === 'elevation') {
        const rgb = getElevationColor(cell.elevation);
        tempColor.setRGB(rgb[0], rgb[1], rgb[2]);
      } else if (colorMode === 'traversability') {
        const rgb = getTraversabilityColor(cell.traversability);
        tempColor.setRGB(rgb[0], rgb[1], rgb[2]);
      } else {
        // Check for purple static obstacles / red dynamic vehicles
        if (cell.semantic_class === 2) {
          tempColor.setHex(0x8B5CF6); // Purple
        } else if (cell.semantic_class === 3) {
          tempColor.setHex(0xEF4444); // Red
        } else if (cell.semantic_class === 1) {
          tempColor.setHex(0xCA8A04); // Yellow Non-Drivable
        } else if (cell.semantic_class === 4) {
          tempColor.setHex(0x15803D); // Vegetation
        } else {
          // Drivable Terrain colored by Foveation Zone (matching reference image)
          if (cell.zone_id === 0) {
            tempColor.setHex(0x0284C7); // Blue Zone 0
          } else if (cell.zone_id === 1) {
            tempColor.setHex(0x16A34A); // Green Zone 1
          } else {
            tempColor.setHex(0xF59E0B); // Orange/Yellow Zone 2
          }
        }
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
      {/* 1. Solid Geometric 3D/2.5D Grid Cell Mesh */}
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

      {/* 2. Crisp Dark Cell Outlines (Grid Boundaries) */}
      <instancedMesh
        ref={wireframeMeshRef}
        args={[undefined, undefined, displayCells.length]}
      >
        <boxGeometry args={[1, 1, 1]} />
        <meshBasicMaterial
          color="#05070D"
          wireframe
          transparent
          opacity={0.85}
        />
      </instancedMesh>

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
