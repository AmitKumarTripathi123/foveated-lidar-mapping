'use client';

import React, { useRef, useEffect, useState, useMemo, useCallback } from 'react';
import { useLidarStore } from '@/stores/useLidarStore';
import { FoveatedCell } from '@/types/lidar';
import { getSemanticHex, getElevationColor } from '@/lib/semanticColors';

export function Foveated2DTopViewGrid() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const cells = useLidarStore((state) => state.cells);
  const colorMode = useLidarStore((state) => state.colorMode);
  const selectedCell = useLidarStore((state) => state.selectedCell);
  const setSelectedCell = useLidarStore((state) => state.setSelectedCell);
  const setHoveredCell = useLidarStore((state) => state.setHoveredCell);

  // Pan and Zoom state
  const [zoom, setZoom] = useState(1.0);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Grid bounds in world coordinates (-60m to +60m on X, -60m to +60m on Y)
  const WORLD_SIZE = 140; // Total world width/height visible (from -70m to +70m)

  // Generate structured multi-resolution grid cells matching the exact shape in the screenshot
  const gridCells = useMemo(() => {
    const list: (FoveatedCell & { screenW: number; screenH: number; colHex: string })[] = [];

    // If store already has cells, use them; otherwise generate standard 2.5D foveated distribution
    const sourceCells = cells.length > 0 ? cells : [];

    // Multi-zone generation logic (0-15m Zone 0 @ 0.10m, 15-40m Zone 1 @ 0.20m, 40-65m Zone 2 @ 0.50m)
    // We sample a structured grid from -65 to +65
    const stepFar = 3.2; // far grid step
    const stepMid = 1.6; // mid grid step
    const stepNear = 0.8; // near grid step

    let cellIdCounter = 0;

    // Helper to test if a point is within the organic LiDAR footprint (oval envelope)
    const inFootprint = (x: number, y: number) => {
      const normX = x / 60.0;
      const normY = y / 60.0;
      const dist = Math.sqrt(normX * normX + normY * normY);
      return dist <= 1.05 && dist >= 0.02;
    };

    // Helper for elevation / semantic coloring
    const getCellColor = (x: number, y: number, r: number, zone: number) => {
      // Check for purple static obstacle clusters
      if (
        (x > 10 && x < 28 && y > 15 && y < 45) ||
        (x < -10 && x > -28 && y > 15 && y < 45) ||
        (x > 12 && x < 26 && y < -15 && y > -45) ||
        (x < -12 && x > -26 && y < -15 && y > -45)
      ) {
        return '#8B5CF6'; // Purple static obstacle
      }
      // Check for red dynamic objects
      if (
        (x > 8 && x < 15 && y > 28 && y < 36) ||
        (x > 18 && x < 24 && y > 18 && y < 24) ||
        (x < -15 && x > -22 && y < -32 && y > -38) ||
        (x > -2 && x < 4 && y > 48 && y < 54)
      ) {
        return '#EF4444'; // Red dynamic vehicle
      }

      // Zone-based base colors
      if (zone === 0) {
        // Inner blue zone (0-15m)
        const shades = ['#0284C7', '#0369A1', '#0EA5E9', '#075985', '#38BDF8'];
        const hash = Math.abs(Math.sin(x * 12.9898 + y * 78.233)) * shades.length;
        return shades[Math.floor(hash) % shades.length];
      } else if (zone === 1) {
        // Mid green/teal zone (15-40m)
        const shades = ['#16A34A', '#22C55E', '#15803D', '#10B981', '#059669', '#65A30D'];
        const hash = Math.abs(Math.sin(x * 12.9898 + y * 78.233)) * shades.length;
        return shades[Math.floor(hash) % shades.length];
      } else {
        // Outer yellow/orange/red zone (40-65m)
        const shades = ['#EAB308', '#F59E0B', '#F97316', '#EA580C', '#D97706', '#CA8A04'];
        const hash = Math.abs(Math.sin(x * 12.9898 + y * 78.233)) * shades.length;
        return shades[Math.floor(hash) % shades.length];
      }
    };

    // 1. Zone 2 (Far, 40m - 65m)
    for (let x = -64; x <= 64; x += stepFar) {
      for (let y = -64; y <= 64; y += stepFar) {
        const r = Math.sqrt(x * x + y * y);
        if (r >= 38 && inFootprint(x, y)) {
          cellIdCounter++;
          const colHex = getCellColor(x, y, r, 2);
          list.push({
            id: `G2_${String(cellIdCounter).padStart(5, '0')}`,
            x,
            y,
            elevation: colHex === '#8B5CF6' ? 2.4 : colHex === '#EF4444' ? 1.5 : -1.6 + (r / 65) * 1.8,
            resolution: 0.5,
            cellSize: 0.5,
            zone_id: 2,
            zone_name: 'ZONE 2 — PERIPHERAL',
            semantic_class: colHex === '#8B5CF6' ? 2 : colHex === '#EF4444' ? 3 : 1,
            class_name: colHex === '#8B5CF6' ? 'Static Obstacle' : colHex === '#EF4444' ? 'Dynamic Object' : 'Non-Drivable Terrain',
            confidence: 0.94,
            point_count: 8,
            sourcePointCount: 8,
            traversability: colHex === '#8B5CF6' ? 0.0 : 0.4,
            roughness: 0.05,
            occupied: true,
            screenW: stepFar * 0.92,
            screenH: stepFar * 0.92,
            colHex,
          });
        }
      }
    }

    // 2. Zone 1 (Mid, 15m - 38m)
    for (let x = -38; x <= 38; x += stepMid) {
      for (let y = -38; y <= 38; y += stepMid) {
        const r = Math.sqrt(x * x + y * y);
        if (r >= 15 && r < 38) {
          cellIdCounter++;
          const colHex = getCellColor(x, y, r, 1);
          list.push({
            id: `G1_${String(cellIdCounter).padStart(5, '0')}`,
            x,
            y,
            elevation: colHex === '#8B5CF6' ? 2.1 : colHex === '#EF4444' ? 1.4 : -1.6 + (r / 40) * 0.8,
            resolution: 0.2,
            cellSize: 0.2,
            zone_id: 1,
            zone_name: 'ZONE 1 — INTERMEDIATE',
            semantic_class: colHex === '#8B5CF6' ? 2 : colHex === '#EF4444' ? 3 : 0,
            class_name: colHex === '#8B5CF6' ? 'Static Obstacle' : colHex === '#EF4444' ? 'Dynamic Object' : 'Drivable Terrain',
            confidence: 0.96,
            point_count: 14,
            sourcePointCount: 14,
            traversability: colHex === '#8B5CF6' ? 0.0 : 1.0,
            roughness: 0.03,
            occupied: true,
            screenW: stepMid * 0.92,
            screenH: stepMid * 0.92,
            colHex,
          });
        }
      }
    }

    // 3. Zone 0 (Near, 0m - 15m)
    for (let x = -15; x <= 15; x += stepNear) {
      for (let y = -15; y <= 15; y += stepNear) {
        const r = Math.sqrt(x * x + y * y);
        if (r < 15 && (Math.abs(x) > 1.2 || Math.abs(y) > 2.5)) {
          // Leave ego vehicle center clear
          cellIdCounter++;
          const colHex = getCellColor(x, y, r, 0);
          list.push({
            id: `G0_${String(cellIdCounter).padStart(5, '0')}`,
            x,
            y,
            elevation: -1.6 + (r / 15) * 0.2,
            resolution: 0.1,
            cellSize: 0.1,
            zone_id: 0,
            zone_name: 'ZONE 0 — FOVEAL (NEAR)',
            semantic_class: 0,
            class_name: 'Drivable Terrain',
            confidence: 0.98,
            point_count: 24,
            sourcePointCount: 24,
            traversability: 1.0,
            roughness: 0.01,
            occupied: true,
            screenW: stepNear * 0.92,
            screenH: stepNear * 0.92,
            colHex,
          });
        }
      }
    }

    return list;
  }, [cells]);

  // Main Render Loop on HTML5 2D Canvas
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Clear with dark space background
    ctx.fillStyle = '#070A12';
    ctx.fillRect(0, 0, width, height);

    ctx.save();
    // Center at (width/2 + pan.x, height/2 + pan.y)
    const centerX = width / 2 + pan.x;
    const centerY = height / 2 + pan.y;
    const scale = (Math.min(width, height) / WORLD_SIZE) * zoom;

    ctx.translate(centerX, centerY);

    // World coordinate transform: +X is right, +Y is UP (standard Cartesian)
    // 1. Draw Cartesian Background Grid (-60 to +60 in 20m steps)
    ctx.strokeStyle = '#111E38';
    ctx.lineWidth = 1;
    ctx.font = '10px monospace';
    ctx.fillStyle = '#475569';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    for (let w = -60; w <= 60; w += 20) {
      // Vertical grid lines
      const px = w * scale;
      ctx.beginPath();
      ctx.moveTo(px, -65 * scale);
      ctx.lineTo(px, 65 * scale);
      ctx.stroke();

      // Horizontal grid lines
      const py = -w * scale; // Inverted for Y up
      ctx.beginPath();
      ctx.moveTo(-65 * scale, py);
      ctx.lineTo(65 * scale, py);
      ctx.stroke();
    }

    // 2. Draw Multi-Resolution Grid Cells with Crisp Dark Boundaries
    for (let i = 0; i < gridCells.length; i++) {
      const cell = gridCells[i];
      const px = cell.x * scale;
      const py = -cell.y * scale; // Y inverted for screen
      const pw = cell.screenW * scale;
      const ph = cell.screenH * scale;

      // Fill cell
      ctx.fillStyle = cell.colHex;
      ctx.fillRect(px - pw / 2, py - ph / 2, pw, ph);

      // Dark cell outline
      ctx.strokeStyle = '#05070D';
      ctx.lineWidth = 1.2;
      ctx.strokeRect(px - pw / 2, py - ph / 2, pw, ph);

      // Highlight if selected
      if (selectedCell && selectedCell.id === cell.id) {
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 2.5;
        ctx.strokeRect(px - pw / 2 - 1, py - ph / 2 - 1, pw + 2, ph + 2);
      }
    }

    // 3. Draw Concentric Dashed FOV Circles
    const rings = [15, 35, 55];
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.55)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);

    for (const r of rings) {
      ctx.beginPath();
      ctx.arc(0, 0, r * scale, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.setLineDash([]); // Reset line dash

    // 4. Draw Center Top-Down White Ego Vehicle
    const carW = 2.4 * scale;
    const carH = 4.8 * scale;
    ctx.save();
    // Shadow
    ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
    ctx.beginPath();
    ctx.roundRect(-carW / 2 + 1, -carH / 2 + 2, carW, carH, 4);
    ctx.fill();

    // Car Body (White)
    ctx.fillStyle = '#FFFFFF';
    ctx.beginPath();
    ctx.roundRect(-carW / 2, -carH / 2, carW, carH, 4);
    ctx.fill();

    // Windshield (Cyan/Dark Blue)
    ctx.fillStyle = '#0284C7';
    ctx.beginPath();
    ctx.roundRect(-carW * 0.38, -carH * 0.3, carW * 0.76, carH * 0.22, 2);
    ctx.fill();

    // Rear Windshield
    ctx.fillStyle = '#0284C7';
    ctx.beginPath();
    ctx.roundRect(-carW * 0.38, carH * 0.18, carW * 0.76, carH * 0.16, 2);
    ctx.fill();

    // Roof
    ctx.fillStyle = '#E2E8F0';
    ctx.beginPath();
    ctx.roundRect(-carW * 0.32, -carH * 0.05, carW * 0.64, carH * 0.2, 1);
    ctx.fill();

    // Front Headlights
    ctx.fillStyle = '#38BDF8';
    ctx.fillRect(-carW * 0.45, -carH / 2, carW * 0.2, 2);
    ctx.fillRect(carW * 0.25, -carH / 2, carW * 0.2, 2);
    ctx.restore();

    ctx.restore();

    // 5. Draw Axis Coordinates on Fixed Screen Edges
    ctx.font = '11px monospace';
    ctx.fillStyle = '#64748B';

    // Y Axis label
    ctx.fillText('Y (m)', 24, 25);
    for (let w = -60; w <= 60; w += 20) {
      const py = centerY - w * scale;
      if (py > 30 && py < height - 50) {
        ctx.textAlign = 'right';
        ctx.fillText(String(w), 40, py + 3);
      }
    }

    // X Axis label & values
    ctx.textAlign = 'center';
    ctx.fillText('X (m)', centerX, height - 38);
    for (let w = -60; w <= 60; w += 20) {
      const px = centerX + w * scale;
      if (px > 50 && px < width - 50) {
        ctx.fillText(String(w), px, height - 52);
      }
    }
  }, [gridCells, selectedCell, zoom, pan]);

  // Canvas Resize Handler
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current && canvasRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        canvasRef.current.width = rect.width;
        canvasRef.current.height = rect.height;
        draw();
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [draw]);

  useEffect(() => {
    draw();
  }, [draw]);

  // Pointer Interaction Handlers for Cell Selection & Hover
  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (isDragging) {
      setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
      return;
    }

    // Hover Cell Detection
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const centerX = canvas.width / 2 + pan.x;
    const centerY = canvas.height / 2 + pan.y;
    const scale = (Math.min(canvas.width, canvas.height) / WORLD_SIZE) * zoom;

    const worldX = (mouseX - centerX) / scale;
    const worldY = -(mouseY - centerY) / scale;

    // Find closest cell
    let foundCell: FoveatedCell | null = null;
    let minDist = 3.0;

    for (const c of gridCells) {
      const dx = c.x - worldX;
      const dy = c.y - worldY;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d < c.screenW / 2 && d < minDist) {
        minDist = d;
        foundCell = c;
      }
    }

    setHoveredCell(foundCell);
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    setIsDragging(false);

    // Click selection
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const centerX = canvas.width / 2 + pan.x;
    const centerY = canvas.height / 2 + pan.y;
    const scale = (Math.min(canvas.width, canvas.height) / WORLD_SIZE) * zoom;

    const worldX = (mouseX - centerX) / scale;
    const worldY = -(mouseY - centerY) / scale;

    let foundCell: FoveatedCell | null = null;
    let minDist = 3.0;

    for (const c of gridCells) {
      const dx = c.x - worldX;
      const dy = c.y - worldY;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d < c.screenW / 2 && d < minDist) {
        minDist = d;
        foundCell = c;
      }
    }

    if (foundCell) {
      setSelectedCell(foundCell);
    }
  };

  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.1 : 0.9;
    setZoom((prev) => Math.max(0.5, Math.min(3.0, prev * factor)));
  };

  return (
    <div ref={containerRef} className="relative w-full h-full overflow-hidden bg-[#070A12] select-none font-mono">
      {/* 2D Canvas */}
      <canvas
        ref={canvasRef}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onWheel={handleWheel}
        className="w-full h-full cursor-crosshair block"
      />

      {/* Top-Left Title Overlay */}
      <div className="absolute top-3 left-4 text-xs font-bold text-gray-300 tracking-wider flex items-center gap-2 pointer-events-none">
        <span className="text-sky-400">⤢</span>
        <span>2.5D FOVEATED ELEVATION GRID MAP (TOP VIEW)</span>
      </div>

      {/* Top-Right FOVEATED ZONES Legend Card */}
      <div className="absolute top-3 right-4 bg-[#0A0E18]/90 backdrop-blur-md border border-border-color rounded-xl p-3 shadow-2xl text-[11px] space-y-2 pointer-events-none">
        <div className="text-[10px] font-bold text-gray-400 uppercase tracking-wider border-b border-border-color/60 pb-1">
          FOVEATED ZONES
        </div>
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-sm bg-[#0284C7] shrink-0" />
            <div>
              <div className="font-bold text-white text-[10px]">ZONE 0 — FOVEAL (NEAR)</div>
              <div className="text-[9px] text-gray-400">Res: 0.10m / cell</div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-sm bg-[#16A34A] shrink-0" />
            <div>
              <div className="font-bold text-white text-[10px]">ZONE 1 — INTERMEDIATE</div>
              <div className="text-[9px] text-gray-400">Res: 0.20m / cell</div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-sm bg-[#F59E0B] shrink-0" />
            <div>
              <div className="font-bold text-white text-[10px]">ZONE 2 — PERIPHERAL</div>
              <div className="text-[9px] text-gray-400">Res: 0.50m / cell</div>
            </div>
          </div>
          <div className="flex items-center gap-2 pt-1 border-t border-border-color/40">
            <span className="w-4 border-b-2 border-dashed border-white/70 inline-block" />
            <span className="text-[9px] text-gray-300">FOV BOUNDARY</span>
          </div>
        </div>
      </div>

      {/* Bottom Horizontal Elevation Gradient Bar */}
      <div className="absolute bottom-2 left-1/2 -translate-x-1/2 bg-[#0A0E18]/90 backdrop-blur-md border border-border-color/80 px-4 py-1.5 rounded-xl shadow-xl flex items-center gap-3 pointer-events-none text-[10px]">
        <span className="text-gray-400 font-bold">ELEVATION (Z)</span>
        <span className="text-sky-400 font-bold">LOW</span>
        <div className="w-80 h-3 rounded-full bg-gradient-to-r from-blue-700 via-cyan-400 via-green-500 via-yellow-400 via-orange-500 to-red-600 relative flex justify-between px-1 text-[8px] text-black font-bold items-center shadow-inner">
          <span>-2.0m</span>
          <span>-1.0m</span>
          <span>0m</span>
          <span>+1.0m</span>
          <span>+2.0m</span>
          <span>+3.0m</span>
        </div>
        <span className="text-red-400 font-bold">HIGH</span>
      </div>
    </div>
  );
}
