"""Authoritative SemanticPOSS Label Remapper (Phase 11.1).

Maps raw SemanticPOSS 40-beam label IDs to the frozen project ontology:
  0   -> drivable_terrain       (ground: 22)
  1   -> non_drivable_terrain   (other static: 19, unknown ground: 20)
  2   -> static_obstacle        (trunk: 8, plants: 9, signs: 10/11/12/18, pole: 13, trashcan: 14, building: 15, cone: 16, fence: 17)
  3   -> dynamic_object         (people: 4/5, rider: 6, car: 7, bike: 21)
  255 -> ignore                 (unlabeled: 0, outlier: 1)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import numpy as np
import yaml

from ml.data.dataset import LiDARError


class SemanticPOSSMappingError(LiDARError, ValueError):
    """Raised when an error occurs during SemanticPOSS label remapping."""
    pass


# Frozen SIH Ontology Constants
SIH_DRIVABLE_TERRAIN = 0
SIH_NON_DRIVABLE_TERRAIN = 1
SIH_STATIC_OBSTACLE = 2
SIH_DYNAMIC_OBJECT = 3
SIH_IGNORE = 255
VALID_SIH_CLASSES: Set[int] = {0, 1, 2, 3, 255}

# Official SemanticPOSS Raw -> SIH Mapping
SEMANTICPOSS_RAW_TO_SIH: Dict[int, int] = {
    # Class 0: drivable_terrain
    22: 0,   # ground

    # Class 1: non_drivable_terrain
    19: 1,   # other static / terrain
    20: 1,   # unknown ground

    # Class 2: static_obstacle
    8: 2,    # trunk
    9: 2,    # plants / vegetation
    10: 2,   # traffic sign 1
    11: 2,   # traffic sign 2
    12: 2,   # traffic sign 3
    13: 2,   # pole
    14: 2,   # trashcan
    15: 2,   # building
    16: 2,   # cone / stone
    17: 2,   # fence
    18: 2,   # traffic sign 4

    # Class 3: dynamic_object
    4: 3,    # people (standing)
    5: 3,    # people (walking)
    6: 3,    # rider
    7: 3,    # car
    21: 3,   # bike / bicycle

    # Class 255: ignore
    0: 255,  # unlabeled
    1: 255,  # outlier
}

SEMANTICPOSS_RAW_NAMES: Dict[int, str] = {
    0: "unlabeled",
    1: "unlabeled_outlier",
    4: "people",
    5: "people",
    6: "rider",
    7: "car",
    8: "trunk",
    9: "plants",
    10: "traffic sign 1",
    11: "traffic sign 2",
    12: "traffic sign 3",
    13: "pole",
    14: "trashcan",
    15: "building",
    16: "cone/stone",
    17: "fence",
    18: "traffic sign 4",
    19: "other static",
    20: "unknown_20",
    21: "bike",
    22: "ground",
}


class SemanticPOSSLabelRemapper:
    """Vectorized O(N) remapper specifically dedicated to SemanticPOSS data."""

    def __init__(
        self,
        custom_mapping: Optional[Dict[int, int]] = None,
        unmapped_policy: str = "ignore",
        max_raw_id: int = 65535,
    ):
        """Initialize remapper with fast lookup table."""
        self.unmapped_policy = unmapped_policy.lower()
        if self.unmapped_policy not in ("ignore", "error"):
            raise SemanticPOSSMappingError(
                f"Invalid unmapped_policy ''{unmapped_policy}''. Supported: ''ignore'', ''error''."
            )

        self.mapping = dict(custom_mapping) if custom_mapping is not None else dict(SEMANTICPOSS_RAW_TO_SIH)
        self.max_raw_id = max_raw_id

        default_val = SIH_IGNORE if self.unmapped_policy == "ignore" else 65535
        self._lut = np.full(self.max_raw_id + 1, default_val, dtype=np.uint16)

        for raw_id, sih_id in self.mapping.items():
            if not (0 <= raw_id <= self.max_raw_id):
                raise SemanticPOSSMappingError(f"Raw label {raw_id} exceeds bounds {self.max_raw_id}")
            if sih_id not in VALID_SIH_CLASSES:
                raise SemanticPOSSMappingError(f"Target SIH class {sih_id} not in {VALID_SIH_CLASSES}")
            self._lut[raw_id] = sih_id

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "SemanticPOSSLabelRemapper":
        """Load configuration from a YAML file."""
        path = Path(yaml_path)
        if not path.is_file():
            raise SemanticPOSSMappingError(f"Mapping configuration file not found: {path.resolve()}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        raw_map = data.get("semanticposs_raw_to_sih", data.get("raw_to_sih", {}))
        int_map = {int(k): int(v) for k, v in raw_map.items()} if raw_map else None
        policy = data.get("unmapped_policy", "ignore")
        return cls(custom_mapping=int_map, unmapped_policy=policy)

    def remap(self, raw_labels: np.ndarray) -> np.ndarray:
        """Remap 1D array of raw SemanticPOSS label IDs to SIH classes {0, 1, 2, 3, 255}."""
        if raw_labels.size == 0:
            return np.empty((0,), dtype=np.uint8)

        clean_raw = raw_labels.astype(np.uint32) & 0xFFFF

        if np.any(clean_raw > self.max_raw_id):
            out_of_bounds = clean_raw[clean_raw > self.max_raw_id]
            raise SemanticPOSSMappingError(
                f"Raw label values {np.unique(out_of_bounds).tolist()} exceed max ID {self.max_raw_id}"
            )

        mapped = self._lut[clean_raw]

        if self.unmapped_policy == "error" and np.any(mapped == 65535):
            unmapped = np.unique(clean_raw[mapped == 65535]).tolist()
            raise SemanticPOSSMappingError(f"Encountered unmapped raw SemanticPOSS label IDs: {unmapped}")

        return mapped.astype(np.uint8)

    def audit(self, raw_labels: np.ndarray) -> Dict[str, Any]:
        """Perform remapping and generate comprehensive audit report."""
        if raw_labels.size == 0:
            return {"total_points": 0, "class_counts": {}, "class_percentages": {}, "passed": True}

        clean_raw = raw_labels.astype(np.uint32) & 0xFFFF
        mapped = self.remap(raw_labels)

        u_raw, c_raw = np.unique(clean_raw, return_counts=True)
        raw_dist = {int(k): int(v) for k, v in zip(u_raw, c_raw)}

        u_sih, c_sih = np.unique(mapped, return_counts=True)
        sih_dist = {int(k): int(v) for k, v in zip(u_sih, c_sih)}

        total = raw_labels.shape[0]
        sih_pcts = {k: (v / total) * 100.0 for k, v in sih_dist.items()}

        return {
            "total_points": total,
            "raw_distribution": raw_dist,
            "sih_distribution": sih_dist,
            "sih_percentages": sih_pcts,
            "passed": True,
        }
