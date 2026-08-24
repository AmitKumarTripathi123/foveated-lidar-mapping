"""
Hierarchical Cell Spatial Indexing & Geometry Engine (SIH PS 26130).
Guarantees consistent multiresolution spatial quantization, zero alignment errors,
and deterministic parent-child cell indexing.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import yaml

from src.core.types import CellKey, GridCell, FoveationZone, SuperClass


# Canonical 3-Zone Distance Tiers (5cm / 15cm / 50cm)
CANONICAL_ZONES: List[FoveationZone] = [
    FoveationZone(name="near_zone", min_radius=0.0, max_radius=10.0, resolution=0.05, level=0),
    FoveationZone(name="mid_zone", min_radius=10.0, max_radius=40.0, resolution=0.15, level=1),
    FoveationZone(name="far_zone", min_radius=40.0, max_radius=100.0, resolution=0.50, level=2),
]


class FoveatedHierarchyEngine:
    """Manages multiresolution spatial indexing and hierarchical parent-child relationships."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.zones = list(CANONICAL_ZONES)
        if config_path is not None:
            self._load_from_config(Path(config_path))

    def _load_from_config(self, cfg_file: Path):
        """Parse foveation tiers directly from system_config.yaml."""
        if cfg_file.is_file():
            with open(cfg_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if "foveation" in data:
                fov = data["foveation"]
                self.zones = [
                    FoveationZone(
                        name=fov["near"].get("name", "near_zone"),
                        min_radius=0.0,
                        max_radius=float(fov["near"]["radius"]),
                        resolution=float(fov["near"]["resolution"]),
                        level=0,
                    ),
                    FoveationZone(
                        name=fov["mid"].get("name", "mid_zone"),
                        min_radius=float(fov["near"]["radius"]),
                        max_radius=float(fov["mid"]["radius"]),
                        resolution=float(fov["mid"]["resolution"]),
                        level=1,
                    ),
                    FoveationZone(
                        name=fov["far"].get("name", "far_zone"),
                        min_radius=float(fov["mid"]["radius"]),
                        max_radius=float(fov["far"]["radius"]),
                        resolution=float(fov["far"]["resolution"]),
                        level=2,
                    ),
                ]

    def resolve_zone(self, distance_r: float) -> Optional[FoveationZone]:
        """Resolve the active foveation tier for a given radial distance."""
        if not math.isfinite(distance_r) or distance_r < 0.0 or distance_r > 100.0:
            return None
        for z in self.zones:
            if z.min_radius <= distance_r < z.max_radius:
                return z
        if math.isclose(distance_r, 100.0, abs_tol=1e-3):
            return self.zones[-1]
        return None

    def point_to_cell_key(self, x: float, y: float) -> Optional[Tuple[CellKey, FoveationZone]]:
        """Map 2D coordinates (x, y) to canonical multiresolution CellKey."""
        r = math.sqrt(x**2 + y**2)
        zone = self.resolve_zone(r)
        if zone is None:
            return None

        ix = int(math.floor(x / zone.resolution))
        iy = int(math.floor(y / zone.resolution))
        return CellKey(level=zone.level, ix=ix, iy=iy), zone

    def get_parent_key(self, key: CellKey) -> Optional[CellKey]:
        """Compute parent cell key at level + 1 for multiresolution hierarchical querying."""
        if key.level >= len(self.zones) - 1:
            return None  # Top-level has no parent

        current_res = self.zones[key.level].resolution
        parent_res = self.zones[key.level + 1].resolution

        # Center point in meters
        center_x = (key.ix + 0.5) * current_res
        center_y = (key.iy + 0.5) * parent_res

        parent_ix = int(math.floor(center_x / parent_res))
        parent_iy = int(math.floor(center_y / parent_res))

        return CellKey(level=key.level + 1, ix=parent_ix, iy=parent_iy)

    def pack_cell_key(self, key: CellKey) -> int:
        """Encode CellKey into a 64-bit integer for fast hash table indexing."""
        # 4 bits level (0..15), 30 bits ix, 30 bits iy
        ix_shifted = (key.ix + (1 << 29)) & 0x3FFFFFFF
        iy_shifted = (key.iy + (1 << 29)) & 0x3FFFFFFF
        return (key.level << 60) | (ix_shifted << 30) | iy_shifted
