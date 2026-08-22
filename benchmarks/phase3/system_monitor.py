"""
Process-level System Resource Monitor for Phase 3 Benchmark.
Tracks RAM (RSS), peak memory, CPU utilization, and GPU availability.
"""

import os
import psutil
import torch
from typing import Dict, Any


class SystemResourceMonitor:
    """Monitors process-level CPU, RAM, and GPU memory."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.peak_ram_mb = 0.0
        self.initial_ram_mb = self.get_ram_mb()
        self.peak_ram_mb = self.initial_ram_mb
        # Prime cpu_percent
        self.process.cpu_percent(interval=None)

    def get_ram_mb(self) -> float:
        """Returns current Resident Set Size (RSS) in MB."""
        rss_bytes = self.process.memory_info().rss
        ram_mb = rss_bytes / (1024.0 * 1024.0)
        if ram_mb > self.peak_ram_mb:
            self.peak_ram_mb = ram_mb
        return round(ram_mb, 2)

    def get_cpu_percent(self) -> float:
        """Returns process-level CPU utilization percentage."""
        return round(self.process.cpu_percent(interval=None), 1)

    def snapshot(self) -> Dict[str, float]:
        """Takes a current resource snapshot."""
        return {
            "ram_mb": self.get_ram_mb(),
            "cpu_percent": self.get_cpu_percent()
        }

    def get_summary(self) -> Dict[str, Any]:
        """Returns benchmark resource summary."""
        cuda_avail = torch.cuda.is_available()
        gpu_info = {
            "gpu_available": cuda_avail,
            "gpu_name": torch.cuda.get_device_name(0) if cuda_avail else "UNAVAILABLE",
            "gpu_memory_mb": round(torch.cuda.memory_allocated(0) / (1024 ** 2), 2) if cuda_avail else "UNAVAILABLE"
        }

        return {
            "initial_ram_mb": round(self.initial_ram_mb, 2),
            "peak_ram_mb": round(self.peak_ram_mb, 2),
            "final_ram_mb": self.get_ram_mb(),
            "gpu": gpu_info
        }
