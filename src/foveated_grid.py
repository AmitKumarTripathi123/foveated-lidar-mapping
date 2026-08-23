"""
Phase-2 Foveated 2.5D Spatial Grid Correctness Engine.
Establishes the exact, deterministic mathematical mapping:
  LiDAR Point (x, y, z, intensity)
      ↓
  Horizontal Distance r = sqrt(x^2 + y^2)
      ↓
  Distance Band Resolution s(r)
      ↓
  2D XY Cell (ix = floor(x/s), iy = floor(y/s))
      ↓
  Elevation Aggregation (mean, min, max) + Semantic Priority Aggregation
      ↓
  2.5D Multi-Layer Grid Map (GridMap25D)
"""

import math
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd

try:
    import foveated_grid_cpp
    HAS_CPP_GRID = True
except ImportError:
    HAS_CPP_GRID = False

from src.types import (
    SuperClass,
    PointCloudFrame,
    FoveationBand,
    CellState,
    GridCell25D,
    FoveatedGridConfig
)


DEFAULT_FROZEN_BANDS: List[FoveationBand] = [
    FoveationBand(name="near_field", min_range=0.0, max_range=10.0, voxel_size=0.05),
    FoveationBand(name="mid_near_field", min_range=10.0, max_range=30.0, voxel_size=0.10),
    FoveationBand(name="mid_far_field", min_range=30.0, max_range=60.0, voxel_size=0.25),
    FoveationBand(name="far_field", min_range=60.0, max_range=100.0, voxel_size=0.50),
]


def distance_to_band(
    distance: float,
    bands: Optional[List[FoveationBand]] = None
) -> Optional[FoveationBand]:
    """
    Resolves the exact FoveationBand for a given horizontal LiDAR distance r = sqrt(x^2 + y^2).
    Uses strict half-open intervals: [min_range, max_range).
    Returns None if distance is invalid (NaN, Inf, negative, or >= 100.0m).
    """
    if not isinstance(distance, (int, float)) or not math.isfinite(distance):
        return None
    if distance < 0.0 or distance >= 100.0:
        return None

    active_bands = bands if bands is not None else DEFAULT_FROZEN_BANDS
    for band in active_bands:
        if band.min_range <= distance < band.max_range:
            return band

    return None


def distance_to_resolution(
    distance: float,
    bands: Optional[List[FoveationBand]] = None
) -> Optional[float]:
    """
    Maps horizontal distance r to its cell resolution:
      [0.0, 10.0)  -> 0.05 m
      [10.0, 30.0) -> 0.10 m
      [30.0, 60.0) -> 0.25 m
      [60.0, 100.0)-> 0.50 m
      >= 100.0 m   -> None (Out of operational range)
    """
    band = distance_to_band(distance, bands=bands)
    return band.voxel_size if band is not None else None


def xy_to_cell(x: float, y: float, resolution: float) -> Tuple[int, int]:
    """
    Maps continuous 2D coordinates (x, y) to discrete cell index (ix, iy)
    using mathematical floor:
      ix = floor(x / resolution)
      iy = floor(y / resolution)
    Correctly handles negative coordinates without truncation towards zero.
    """
    if not math.isfinite(x) or not math.isfinite(y) or resolution <= 0.0:
        raise ValueError(f"Invalid inputs to xy_to_cell: x={x}, y={y}, resolution={resolution}")
    ix = int(math.floor(round(x / resolution, 9)))
    iy = int(math.floor(round(y / resolution, 9)))
    return (ix, iy)



def cell_to_bounds(ix: int, iy: int, resolution: float) -> Tuple[float, float, float, float]:
    """Returns spatial bounding box (min_x, max_x, min_y, max_y) for cell (ix, iy)."""
    min_x = ix * resolution
    max_x = (ix + 1) * resolution
    min_y = iy * resolution
    max_y = (iy + 1) * resolution
    return (min_x, max_x, min_y, max_y)


def point_to_cell(
    x: float,
    y: float,
    z: float,
    semantic_class: int = SuperClass.IGNORE_LABEL,
    confidence: float = 1.0,
    bands: Optional[List[FoveationBand]] = None
) -> Optional[GridCell25D]:
    """
    Transforms a single 3D LiDAR point to its corresponding 2.5D spatial cell.
    Returns None if point is out of range or non-finite.
    """
    if not math.isfinite(x) or not math.isfinite(y) or not math.isfinite(z):
        return None

    r = math.sqrt(x * x + y * y)
    band = distance_to_band(r, bands=bands)
    if band is None:
        return None

    ix, iy = xy_to_cell(x, y, band.voxel_size)

    # Calculate traversability score
    trav = 0.0
    if semantic_class == SuperClass.DRIVABLE_TERRAIN:
        trav = 1.0
    elif semantic_class == SuperClass.NON_DRIVABLE_TERRAIN:
        trav = 0.2
    else:
        trav = 0.0

    return GridCell25D(
        ix=ix,
        iy=iy,
        resolution=band.voxel_size,
        point_count=1,
        elevation_mean=float(z),
        elevation_min=float(z),
        elevation_max=float(z),
        semantic_class=int(semantic_class),
        confidence=float(confidence),
        traversability=trav,
        state=CellState.OCCUPIED,
        band_name=band.name
    )


class GridMap25D:
    """
    High-Performance Multi-Layer 2.5D Foveated Grid Map Representation.
    Stores columnar vectorized spatial layers for sub-millisecond querying and export,
    while providing backward-compatible lazy dictionary access.
    """
    def __init__(
        self,
        bands: Optional[List[FoveationBand]] = None,
        frame_id: str = "000000",
        timestamp: float = 0.0,
        sequence_id: str = "00",
        # Vectorized layer arrays
        bands_arr: Optional[np.ndarray] = None,
        ix_arr: Optional[np.ndarray] = None,
        iy_arr: Optional[np.ndarray] = None,
        res_arr: Optional[np.ndarray] = None,
        counts_arr: Optional[np.ndarray] = None,
        mean_z_arr: Optional[np.ndarray] = None,
        min_z_arr: Optional[np.ndarray] = None,
        max_z_arr: Optional[np.ndarray] = None,
        classes_arr: Optional[np.ndarray] = None,
        conf_arr: Optional[np.ndarray] = None,
        trav_arr: Optional[np.ndarray] = None,
        semantic_counts_list: Optional[List[Dict[int, int]]] = None
    ):
        self.bands = list(bands) if bands is not None else list(DEFAULT_FROZEN_BANDS)
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.sequence_id = sequence_id

        self._bands = bands_arr if bands_arr is not None else np.empty(0, dtype=object)
        self._ix = ix_arr if ix_arr is not None else np.empty(0, dtype=np.int64)
        self._iy = iy_arr if iy_arr is not None else np.empty(0, dtype=np.int64)
        self._res = res_arr if res_arr is not None else np.empty(0, dtype=np.float32)
        self._counts = counts_arr if counts_arr is not None else np.empty(0, dtype=np.int64)
        self._mean_z = mean_z_arr if mean_z_arr is not None else np.empty(0, dtype=np.float32)
        self._min_z = min_z_arr if min_z_arr is not None else np.empty(0, dtype=np.float32)
        self._max_z = max_z_arr if max_z_arr is not None else np.empty(0, dtype=np.float32)
        self._classes = classes_arr if classes_arr is not None else np.empty(0, dtype=np.int64)
        self._conf = conf_arr if conf_arr is not None else np.empty(0, dtype=np.float32)
        self._trav = trav_arr if trav_arr is not None else np.empty(0, dtype=np.float32)
        self._semantic_counts_list = semantic_counts_list

        self._cells_dict: Optional[Dict[Tuple[str, int, int], GridCell25D]] = None
        self._custom_cells: Dict[Tuple[str, int, int], GridCell25D] = {}

    def __len__(self) -> int:
        return len(self._ix) + len(self._custom_cells)

    def _populate_cells_dict(self):
        if self._cells_dict is None:
            self._cells_dict = {}
            for i in range(len(self._ix)):
                b_name = str(self._bands[i])
                ix = int(self._ix[i])
                iy = int(self._iy[i])
                sem_counts = self._semantic_counts_list[i] if (self._semantic_counts_list is not None and i < len(self._semantic_counts_list)) else {int(self._classes[i]): int(self._counts[i])}
                cell = GridCell25D(
                    ix=ix,
                    iy=iy,
                    resolution=float(self._res[i]),
                    point_count=int(self._counts[i]),
                    elevation_mean=float(self._mean_z[i]),
                    elevation_min=float(self._min_z[i]),
                    elevation_max=float(self._max_z[i]),
                    semantic_class=int(self._classes[i]),
                    confidence=float(self._conf[i]),
                    traversability=float(self._trav[i]),
                    state=CellState.OCCUPIED,
                    band_name=b_name,
                    semantic_counts=sem_counts
                )
                self._cells_dict[(b_name, ix, iy)] = cell
            self._cells_dict.update(self._custom_cells)


    @property
    def cells(self) -> Dict[Tuple[str, int, int], GridCell25D]:
        self._populate_cells_dict()
        return self._cells_dict

    @property
    def num_cells(self) -> int:
        if self._cells_dict is not None:
            return len(self._cells_dict)
        return len(self._ix) + len(self._custom_cells)

    @property
    def num_occupied_cells(self) -> int:
        if self._cells_dict is not None:
            return sum(1 for c in self._cells_dict.values() if c.state == CellState.OCCUPIED)
        return len(self._ix) + sum(1 for c in self._custom_cells.values() if c.state == CellState.OCCUPIED)

    def get_cell(self, band_name: str, ix: int, iy: int) -> GridCell25D:
        """
        Retrieves cell at (band_name, ix, iy).
        If cell was never observed, returns an UNKNOWN state cell with NaN elevation.
        """
        key = (band_name, ix, iy)
        if self._cells_dict is not None:
            if key in self._cells_dict:
                return self._cells_dict[key]
        elif key in self._custom_cells:
            return self._custom_cells[key]
        else:
            if len(self._ix) > 0:
                match = (self._bands == band_name) & (self._ix == ix) & (self._iy == iy)
                indices = np.flatnonzero(match)
                if len(indices) > 0:
                    idx = indices[0]
                    return GridCell25D(
                        ix=ix,
                        iy=iy,
                        resolution=float(self._res[idx]),
                        point_count=int(self._counts[idx]),
                        elevation_mean=float(self._mean_z[idx]),
                        elevation_min=float(self._min_z[idx]),
                        elevation_max=float(self._max_z[idx]),
                        semantic_class=int(self._classes[idx]),
                        confidence=float(self._conf[idx]),
                        traversability=float(self._trav[idx]),
                        state=CellState.OCCUPIED,
                        band_name=band_name
                    )

        # Find band resolution
        res = 0.05
        for b in self.bands:
            if b.name == band_name:
                res = b.voxel_size
                break

        return GridCell25D(
            ix=ix,
            iy=iy,
            resolution=res,
            point_count=0,
            elevation_mean=float("nan"),
            elevation_min=float("nan"),
            elevation_max=float("nan"),
            semantic_class=SuperClass.IGNORE_LABEL,
            confidence=0.0,
            traversability=0.0,
            state=CellState.UNKNOWN,
            band_name=band_name
        )

    def get_cell_at_xy(self, x: float, y: float) -> Optional[GridCell25D]:
        """Queries the observed 2.5D cell at real-world coordinates (x, y)."""
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        r = math.sqrt(x * x + y * y)
        band = distance_to_band(r, bands=self.bands)
        if band is None:
            return None
        ix, iy = xy_to_cell(x, y, band.voxel_size)
        return self.get_cell(band.name, ix, iy)

    def insert_cell(self, cell: GridCell25D):
        """Inserts or overwrites a cell in the grid map."""
        key = (cell.band_name, cell.ix, cell.iy)
        if self._cells_dict is not None:
            self._cells_dict[key] = cell
        else:
            self._custom_cells[key] = cell

    def to_dataframe(self) -> pd.DataFrame:
        """Exports all observed cells into a tabular pandas DataFrame using high-speed vectorization."""
        if len(self._ix) == 0 and len(self._custom_cells) == 0:
            return pd.DataFrame(columns=[
                "band_name", "ix", "iy", "resolution", "min_x", "max_x", "min_y", "max_y",
                "center_x", "center_y", "point_count", "elevation_mean", "elevation_min",
                "elevation_max", "semantic_class", "confidence", "traversability", "state"
            ])

        if len(self._custom_cells) == 0 and len(self._ix) > 0:
            min_x = self._ix * self._res
            max_x = min_x + self._res
            min_y = self._iy * self._res
            max_y = min_y + self._res
            return pd.DataFrame({
                "band_name": self._bands,
                "ix": self._ix,
                "iy": self._iy,
                "resolution": self._res,
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
                "center_x": (min_x + max_x) / 2.0,
                "center_y": (min_y + max_y) / 2.0,
                "point_count": self._counts,
                "elevation_mean": self._mean_z,
                "elevation_min": self._min_z,
                "elevation_max": self._max_z,
                "semantic_class": self._classes,
                "confidence": self._conf,
                "traversability": self._trav,
                "state": "OCCUPIED"
            })

        self._populate_cells_dict()
        rows = []
        for c in self._cells_dict.values():
            min_x, max_x, min_y, max_y = c.bounds
            rows.append({
                "band_name": c.band_name,
                "ix": c.ix,
                "iy": c.iy,
                "resolution": c.resolution,
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
                "center_x": (min_x + max_x) / 2.0,
                "center_y": (min_y + max_y) / 2.0,
                "point_count": c.point_count,
                "elevation_mean": c.elevation_mean,
                "elevation_min": c.elevation_min,
                "elevation_max": c.elevation_max,
                "semantic_class": c.semantic_class,
                "confidence": c.confidence,
                "traversability": c.traversability,
                "state": c.state.name
            })
        return pd.DataFrame(rows)


class FoveatedGrid25D:
    """
    Optimized Phase-4 & Phase-6 Foveated 2.5D Grid Builder.
    Supports high-speed pure C++ pybind11 execution with transparent Python fallback:
      1. Elevation aggregation: mean(z), min(z), max(z)
      2. Deterministic obstacle-preserving semantic priority aggregation:
         dynamic_object (3) > static_obstacle (2) > non_drivable (1) > drivable (0) > ignore (255)
      3. Traversability estimation and confidence propagation.
    """
    def __init__(
        self,
        bands: Optional[List[FoveationBand]] = None,
        max_range: float = 100.0,
        use_cpp: bool = True
    ):
        self.bands = list(bands) if bands is not None else list(DEFAULT_FROZEN_BANDS)
        self.max_range = float(max_range)
        self.use_cpp = use_cpp and HAS_CPP_GRID
        self._cpp_engine = foveated_grid_cpp.FoveatedGridEngine() if HAS_CPP_GRID else None

    def build_grid(
        self,
        points: np.ndarray,
        labels: Optional[np.ndarray] = None,
        confidences: Optional[np.ndarray] = None,
        frame_id: str = "000000",
        timestamp: float = 0.0,
        sequence_id: str = "00"
    ) -> GridMap25D:
        """
        Constructs a GridMap25D from raw or preprocessed LiDAR arrays with high-speed vectorization.
        points: float32 [N, 4] -> (x, y, z, intensity)
        labels: int/uint32 [N] -> super-classes
        confidences: float32 [N] -> prediction confidence
        """
        if points is None or len(points) == 0:
            return GridMap25D(
                bands=self.bands,
                frame_id=frame_id,
                timestamp=timestamp,
                sequence_id=sequence_id
            )

        if self.use_cpp and self._cpp_engine is not None:
            lbls_in = labels.astype(np.int64) if labels is not None else None
            confs_in = confidences.astype(np.float32) if confidences is not None else None
            pts_in = points.astype(np.float32)
            res_dict = self._cpp_engine.build_grid_numpy(pts_in, lbls_in, confs_in)
            if res_dict["num_cells"] == 0:
                return GridMap25D(
                    bands=self.bands,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    sequence_id=sequence_id
                )
            return GridMap25D(
                bands=self.bands,
                frame_id=frame_id,
                timestamp=timestamp,
                sequence_id=sequence_id,
                bands_arr=np.array(res_dict["bands"], dtype=object),
                ix_arr=res_dict["ix"],
                iy_arr=res_dict["iy"],
                res_arr=res_dict["resolution"],
                counts_arr=res_dict["point_count"],
                mean_z_arr=res_dict["elevation_mean"],
                min_z_arr=res_dict["elevation_min"],
                max_z_arr=res_dict["elevation_max"],
                classes_arr=res_dict["semantic_class"],
                conf_arr=res_dict["confidence"],
                trav_arr=res_dict["traversability"],
                semantic_counts_list=res_dict.get("semantic_counts", None)
            )

        N = len(points)
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        r = np.sqrt(x * x + y * y)

        lbls = labels if labels is not None else np.full(N, SuperClass.IGNORE_LABEL, dtype=np.int64)
        confs = confidences if confidences is not None else np.ones(N, dtype=np.float32)

        # Semantic priority weights for aggregation:
        # dynamic (4) > static (3) > non-drivable (2) > drivable (1) > ignore (0)
        p_weights = np.zeros(256, dtype=np.int32)
        p_weights[SuperClass.DYNAMIC_OBJECT] = 4
        p_weights[SuperClass.STATIC_OBSTACLE] = 3
        p_weights[SuperClass.NON_DRIVABLE_TERRAIN] = 2
        p_weights[SuperClass.DRIVABLE_TERRAIN] = 1
        p_weights[SuperClass.IGNORE_LABEL] = 0

        all_bands = []
        all_ix = []
        all_iy = []
        all_res = []
        all_counts = []
        all_mean_z = []
        all_min_z = []
        all_max_z = []
        all_classes = []
        all_conf = []
        all_trav = []
        all_semantic_counts = []

        OFFSET = 50000  # Coordinate hashing bias

        # Process each band independently with vectorized spatial binning
        for band in self.bands:
            mask = (r >= band.min_range) & (r < band.max_range) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            if not np.any(mask):
                continue

            bx = x[mask]
            by = y[mask]
            bz = z[mask]
            bl = lbls[mask]
            bc = confs[mask]
            res = band.voxel_size

            # Mathematical floor for 2D XY cell indexing
            b_ix = np.floor(bx / res).astype(np.int64)
            b_iy = np.floor(by / res).astype(np.int64)

            # 1D 64-bit integer hash encoding (replaces slow 2D unique sort)
            keys_1d = ((b_ix + OFFSET) << 32) | ((b_iy + OFFSET) & 0xFFFFFFFF)
            unique_keys_1d, inverse_idx = np.unique(keys_1d, return_inverse=True)
            num_cells = len(unique_keys_1d)

            # Decode unique coordinates
            u_ix = (unique_keys_1d >> 32) - OFFSET
            u_iy = (unique_keys_1d & 0xFFFFFFFF) - OFFSET

            # 1. Point counts per cell
            counts = np.bincount(inverse_idx, minlength=num_cells)

            # 2. Elevation Mean
            sum_z = np.bincount(inverse_idx, weights=bz, minlength=num_cells)
            mean_z = sum_z / counts

            # 3. Elevation Min & Max (using np.minimum.at / np.maximum.at)
            min_z = np.full(num_cells, np.inf, dtype=np.float32)
            max_z = np.full(num_cells, -np.inf, dtype=np.float32)
            np.minimum.at(min_z, inverse_idx, bz)
            np.maximum.at(max_z, inverse_idx, bz)

            # 4. Confidence Mean
            sum_c = np.bincount(inverse_idx, weights=bc, minlength=num_cells)
            mean_c = sum_c / counts

            # 5. Semantic distribution counts per cell
            c0 = np.bincount(inverse_idx[bl == 0], minlength=num_cells) if np.any(bl == 0) else np.zeros(num_cells, dtype=np.int64)
            c1 = np.bincount(inverse_idx[bl == 1], minlength=num_cells) if np.any(bl == 1) else np.zeros(num_cells, dtype=np.int64)
            c2 = np.bincount(inverse_idx[bl == 2], minlength=num_cells) if np.any(bl == 2) else np.zeros(num_cells, dtype=np.int64)
            c3 = np.bincount(inverse_idx[bl == 3], minlength=num_cells) if np.any(bl == 3) else np.zeros(num_cells, dtype=np.int64)
            c255 = np.bincount(inverse_idx[bl == 255], minlength=num_cells) if np.any(bl == 255) else np.zeros(num_cells, dtype=np.int64)
            for ci in range(num_cells):
                cd = {0: int(c0[ci]), 1: int(c1[ci]), 2: int(c2[ci]), 3: int(c3[ci])}
                if c255[ci] > 0:
                    cd[255] = int(c255[ci])
                all_semantic_counts.append(cd)

            # 6. Obstacle-preserving semantic class aggregation
            ranks = p_weights[np.clip(bl, 0, 255)]
            sort_order = np.lexsort((-ranks, inverse_idx))
            _, first_idx = np.unique(inverse_idx[sort_order], return_index=True)
            best_idx = sort_order[first_idx]
            agg_labels = bl[best_idx]

            # 7. Vectorized traversability mapping
            trav = np.zeros(num_cells, dtype=np.float32)
            trav[agg_labels == SuperClass.DRIVABLE_TERRAIN] = 1.0
            trav[agg_labels == SuperClass.NON_DRIVABLE_TERRAIN] = 0.2

            all_bands.append(np.full(num_cells, band.name))
            all_ix.append(u_ix)
            all_iy.append(u_iy)
            all_res.append(np.full(num_cells, res, dtype=np.float32))
            all_counts.append(counts)
            all_mean_z.append(mean_z)
            all_min_z.append(min_z)
            all_max_z.append(max_z)
            all_classes.append(agg_labels)
            all_conf.append(mean_c)
            all_trav.append(trav)

        if len(all_bands) == 0:
            return GridMap25D(
                bands=self.bands,
                frame_id=frame_id,
                timestamp=timestamp,
                sequence_id=sequence_id
            )

        return GridMap25D(
            bands=self.bands,
            frame_id=frame_id,
            timestamp=timestamp,
            sequence_id=sequence_id,
            bands_arr=np.concatenate(all_bands),
            ix_arr=np.concatenate(all_ix),
            iy_arr=np.concatenate(all_iy),
            res_arr=np.concatenate(all_res),
            counts_arr=np.concatenate(all_counts),
            mean_z_arr=np.concatenate(all_mean_z),
            min_z_arr=np.concatenate(all_min_z),
            max_z_arr=np.concatenate(all_max_z),
            classes_arr=np.concatenate(all_classes),
            conf_arr=np.concatenate(all_conf),
            trav_arr=np.concatenate(all_trav),
            semantic_counts_list=all_semantic_counts
        )

