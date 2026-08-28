"""
Phase 19.1 Performance & Hardware Telemetry Collector.
Measures GPU VRAM, CUDA allocations, CPU utilization, RAM, and runtime frame statistics.
"""

from dataclasses import dataclass, field
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import psutil
import torch


@dataclass
class TelemetrySnapshot:
    timestamp: float
    # GPU Metrics
    gpu_available: bool
    gpu_name: str
    gpu_allocated_mb: float
    gpu_reserved_mb: float
    gpu_peak_allocated_mb: float
    gpu_utilization_pct: Optional[float] = None
    gpu_temperature_c: Optional[float] = None
    # CPU Metrics
    cpu_percent: float = 0.0
    process_rss_mb: float = 0.0
    system_ram_used_pct: float = 0.0
    # Runtime Metrics
    fps: float = 0.0
    dropped_frames: int = 0
    queue_depth: int = 0


class TelemetryCollector:
    """Collects comprehensive hardware and process telemetry."""

    def __init__(self, device: torch.device):
        self.device = device
        self.process = psutil.Process(os.getpid())

    def capture_snapshot(
        self,
        fps: float = 0.0,
        dropped_frames: int = 0,
        queue_depth: int = 0,
    ) -> TelemetrySnapshot:
        """Capture an instantaneous telemetry record."""
        gpu_avail = self.device.type == "cuda" and torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_avail else "CPU"
        gpu_alloc = round(torch.cuda.memory_allocated(self.device) / (1024**2), 2) if gpu_avail else 0.0
        gpu_res = round(torch.cuda.memory_reserved(self.device) / (1024**2), 2) if gpu_avail else 0.0
        gpu_peak = round(torch.cuda.max_memory_allocated(self.device) / (1024**2), 2) if gpu_avail else 0.0

        # Process CPU & RAM
        cpu_pct = psutil.cpu_percent(interval=None)
        rss_mb = round(self.process.memory_info().rss / (1024**2), 2)
        ram_pct = psutil.virtual_memory().percent

        return TelemetrySnapshot(
            timestamp=time.time(),
            gpu_available=gpu_avail,
            gpu_name=gpu_name,
            gpu_allocated_mb=gpu_alloc,
            gpu_reserved_mb=gpu_res,
            gpu_peak_allocated_mb=gpu_peak,
            gpu_utilization_pct=None, # Marked None if vendor NVML driver not exposed
            gpu_temperature_c=None,
            cpu_percent=cpu_pct,
            process_rss_mb=rss_mb,
            system_ram_used_pct=ram_pct,
            fps=fps,
            dropped_frames=dropped_frames,
            queue_depth=queue_depth,
        )

    def to_dict(self, snapshot: TelemetrySnapshot) -> Dict[str, Any]:
        """Convert snapshot to JSON-serializable dictionary."""
        return {
            "gpu": {
                "available": snapshot.gpu_available,
                "name": snapshot.gpu_name,
                "allocated_mb": snapshot.gpu_allocated_mb,
                "reserved_mb": snapshot.gpu_reserved_mb,
                "peak_allocated_mb": snapshot.gpu_peak_allocated_mb,
                "utilization_pct": snapshot.gpu_utilization_pct if snapshot.gpu_utilization_pct is not None else {"available": False, "reason": "NVML query unavailable on Windows WDDM"},
                "temperature_c": snapshot.gpu_temperature_c if snapshot.gpu_temperature_c is not None else {"available": False, "reason": "NVML thermal sensor unavailable"},
            },
            "cpu": {
                "cpu_utilization_pct": snapshot.cpu_percent,
                "process_rss_mb": snapshot.process_rss_mb,
                "system_ram_used_pct": snapshot.system_ram_used_pct,
            },
            "runtime": {
                "fps": snapshot.fps,
                "dropped_frames": snapshot.dropped_frames,
                "queue_depth": snapshot.queue_depth,
            }
        }
