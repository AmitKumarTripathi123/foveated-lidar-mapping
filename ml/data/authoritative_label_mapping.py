"""Authoritative Semantic Label Remapping Engine for Multi-Dataset Support (Phase 11).

Supports:
  1. SemanticKITTI (default for local dataset sequence 00)
  2. SemanticPOSS (supported for external dataset acquisition)

Project Ontology (Frozen):
  0   -> drivable_terrain       (Roads, parking, drivable ground)
  1   -> non_drivable_terrain   (Sidewalks, terrain, rough ground)
  2   -> static_obstacle        (Buildings, fences, poles, vegetation, structures)
  3   -> dynamic_object         (Cars, trucks, buses, cyclists, pedestrians)
  255 -> ignore                 (Unlabeled noise, outliers, unsupported classes)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import numpy as np
import yaml

from ml.data.dataset import LiDARError


class AuthoritativeMappingError(LiDARError, ValueError):
    """Raised when an error occurs during authoritative label remapping."""
    pass


# Target Ontology Constants
SIH_DRIVABLE_TERRAIN = 0
SIH_NON_DRIVABLE_TERRAIN = 1
SIH_STATIC_OBSTACLE = 2
SIH_DYNAMIC_OBJECT = 3
SIH_IGNORE = 255
VALID_SIH_CLASSES: Set[int] = {0, 1, 2, 3, 255}

SIH_CLASS_NAMES: Dict[int, str] = {
    0: "drivable_terrain",
    1: "non_drivable_terrain",
    2: "static_obstacle",
    3: "dynamic_object",
    255: "ignore",
}

# SemanticKITTI Standard Raw -> SIH Mapping
SEMANTICKITTI_TO_SIH: Dict[int, int] = {
    40: 0, 44: 0, 60: 0,                           # drivable
    48: 1, 49: 1, 72: 1,                           # non-drivable
    50: 2, 51: 2, 52: 2, 70: 2, 71: 2, 80: 2, 81: 2, 99: 2, # static obstacle
    10: 3, 11: 3, 13: 3, 15: 3, 16: 3, 18: 3, 20: 3, 30: 3, 31: 3, 32: 3, # dynamic
    252: 3, 253: 3, 254: 3, 255: 3, 256: 3, 257: 3, 258: 3, 259: 3,       # dynamic moving
    0: 255, 1: 255,                                 # ignore
}

# SemanticPOSS Standard Raw -> SIH Mapping
SEMANTICPOSS_TO_SIH: Dict[int, int] = {
    22: 0,                                         # ground -> drivable
    19: 1, 20: 1,                                  # static other / ground -> non-drivable
    8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2, 17: 2, 18: 2, # static obstacle
    4: 3, 5: 3, 6: 3, 7: 3, 21: 3,                 # people, rider, car, bike -> dynamic
    0: 255, 1: 255,                                 # ignore
}


class AuthoritativeLabelRemapper:
    """Vectorized, deterministic O(N) remapper supporting SemanticKITTI and SemanticPOSS."""

    def __init__(
        self,
        dataset_name: str = "SemanticKITTI",
        custom_mapping: Optional[Dict[int, int]] = None,
        unmapped_policy: str = "ignore",
        max_raw_id: int = 65535,
    ):
        """Initialize authoritative remapper with fast lookup table."""
        self.dataset_name = dataset_name
        self.unmapped_policy = unmapped_policy.lower()
        if self.unmapped_policy not in ("ignore", "error"):
            raise AuthoritativeMappingError(
                f"Invalid unmapped_policy ''{unmapped_policy}''. Supported: ''ignore'', ''error''."
            )

        self.max_raw_id = max_raw_id

        if custom_mapping is not None:
            self.mapping = dict(custom_mapping)
        elif dataset_name.lower() in ("semantickitti", "kitti", "default"):
            self.mapping = dict(SEMANTICKITTI_TO_SIH)
        elif dataset_name.lower() in ("semanticposs", "poss"):
            self.mapping = dict(SEMANTICPOSS_TO_SIH)
        else:
            raise AuthoritativeMappingError(f"Unknown dataset_name ''{dataset_name}''.")

        default_val = SIH_IGNORE if self.unmapped_policy == "ignore" else 65535
        self._lut = np.full(self.max_raw_id + 1, default_val, dtype=np.uint16)

        for raw_id, sih_id in self.mapping.items():
            if not (0 <= raw_id <= self.max_raw_id):
                raise AuthoritativeMappingError(f"Raw label {raw_id} exceeds bounds {self.max_raw_id}")
            if sih_id not in VALID_SIH_CLASSES:
                raise AuthoritativeMappingError(f"Target SIH class {sih_id} not in {VALID_SIH_CLASSES}")
            self._lut[raw_id] = sih_id

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "AuthoritativeLabelRemapper":
        """Load configuration from authoritative YAML config file."""
        path = Path(yaml_path)
        if not path.is_file():
            raise AuthoritativeMappingError(f"Config file not found: {path.resolve()}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        ds_name = data.get("dataset_name", "SemanticKITTI")
        policy = data.get("unmapped_policy", "ignore")

        if ds_name.lower() in ("semantickitti", "kitti") and "semantickitti_raw_to_sih" in data:
            raw_map = {int(k): int(v) for k, v in data["semantickitti_raw_to_sih"].items()}
        elif ds_name.lower() in ("semanticposs", "poss") and "semanticposs_raw_to_sih" in data:
            raw_map = {int(k): int(v) for k, v in data["semanticposs_raw_to_sih"].items()}
        elif "raw_to_sih" in data:
            raw_map = {int(k): int(v) for k, v in data["raw_to_sih"].items()}
        else:
            raw_map = None

        return cls(dataset_name=ds_name, custom_mapping=raw_map, unmapped_policy=policy)

    def remap(self, raw_labels: np.ndarray) -> np.ndarray:
        """Remap 1D array of raw label IDs to SIH classes {0, 1, 2, 3, 255}."""
        if raw_labels.size == 0:
            return np.empty((0,), dtype=np.uint8)

        clean_raw = raw_labels.astype(np.uint32) & 0xFFFF

        if np.any(clean_raw > self.max_raw_id):
            out_of_bounds = clean_raw[clean_raw > self.max_raw_id]
            raise AuthoritativeMappingError(
                f"Raw label values {np.unique(out_of_bounds).tolist()} exceed max ID {self.max_raw_id}"
            )

        mapped = self._lut[clean_raw]

        if self.unmapped_policy == "error" and np.any(mapped == 65535):
            unmapped = np.unique(clean_raw[mapped == 65535]).tolist()
            raise AuthoritativeMappingError(f"Encountered unmapped raw label IDs: {unmapped}")

        return mapped.astype(np.uint8)

    def audit(self, raw_labels: np.ndarray) -> Dict[str, Any]:
        """Audit raw label distribution and mapping percentages."""
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
            "dataset_name": self.dataset_name,
            "total_points": total,
            "raw_distribution": raw_dist,
            "sih_distribution": sih_dist,
            "sih_percentages": sih_pcts,
            "passed": True,
        }
