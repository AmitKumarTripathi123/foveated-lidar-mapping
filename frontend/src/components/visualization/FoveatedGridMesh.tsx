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

  const displayCells = cells;

  useEffect(() => {
    if (!instancedMeshRef.current || displayCells.length === 0) return;

    const zBase = -1.65; // Ground reference datum

    for (let i = 0; i < displayCells.length; i++) {
      const cell = displayCells[i];
      const res = cell.cellSize || cell.resolution || 0.10;

      let height = 0.10;
      let centerZ = -1.56;

      if (gridRenderStyle === 'extruded_3d' || (viewMode3D === 'foveated_elevation' && colorMode === 'elevation')) {
        // 3D Column Extrusion
        height = Math.max(0.12, (cell.elevation - zBase) * 1.2);
        centerZ = zBase + height / 2;
      } else {
        // 2.5D Elevation Plane:
        // Elevation is visually represented by cell height and vertical position:
        const elevationOffset = Math.max(0, cell.elevation - zBase);
        height = Math.max(0.06, elevationOffset * 0.45 + 0.06);
        centerZ = zBase + height / 2;
      }

      // Spatially continuous cells: 0.985 scale ensures adjacent cells meet seamlessly
      // while subtle wireframe outlines preserve crisp variable-resolution boundaries
      tempObject.position.set(cell.x, cell.y, centerZ);
      tempObject.scale.set(res * 0.985, res * 0.985, height);
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
        // True Semantic Coloring matching the Semantic Legend
        if (cell.semantic_class === 2) {
          tempColor.setHex(0x8B5CF6); // Static Obstacle / Building (Purple)
        } else if (cell.semantic_class === 3) {
          tempColor.setHex(0xEF4444); // Dynamic Object / Vehicle (Red)
        } else if (cell.semantic_class === 1) {
          tempColor.setHex(0xCA8A04); // Non-Drivable Curb / Sidewalk (Amber)
        } else if (cell.semantic_class === 4) {
          tempColor.setHex(0x15803D); // Vegetation (Dark Green)
        } else {
          // Drivable Terrain: Unified Green across all zones!
          // This ensures the roadway reads as ONE continuous ground plane!
          tempColor.setHex(0x22C55E); // Green (#22C55E)
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
      {layers.adaptiveGridWireframe && (
        <instancedMesh
          ref={wireframeMeshRef}
          args={[undefined, undefined, displayCells.length]}
        >
          <boxGeometry args={[1, 1, 1]} />
          <meshBasicMaterial
            color="#020617"
            wireframe
            transparent
            opacity={0.65}
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
