"""SIH Four-Class Semantic Ontology and Vectorized Label Remapper (Phase 3).

Frozen Project Ontology:
    0   -> drivable_terrain       (Roads, parking, lane markings)
    1   -> non_drivable_terrain   (Sidewalks, terrain, grass, rough ground)
    2   -> static_obstacle        (Buildings, fences, poles, vegetation, signs)
    3   -> dynamic_object         (Cars, trucks, buses, motorcycles, pedestrians)
    255 -> ignore                 (Unlabeled noise, outliers, unsupported classes)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import yaml

from ml.data.dataset import LiDARError


class LabelMappingError(LiDARError, ValueError):
    """Raised when an error occurs during semantic label remapping or configuration."""
    pass


# Frozen SIH Ontology Constants
SIH_DRIVABLE_TERRAIN: int = 0
SIH_NON_DRIVABLE_TERRAIN: int = 1
SIH_STATIC_OBSTACLE: int = 2
SIH_DYNAMIC_OBJECT: int = 3
SIH_IGNORE: int = 255

SIH_CLASS_NAMES: Dict[int, str] = {
    0: "drivable_terrain",
    1: "non_drivable_terrain",
    2: "static_obstacle",
    3: "dynamic_object",
    255: "ignore",
}

VALID_SIH_IDS: Set[int] = {0, 1, 2, 3, 255}

# Default verified SemanticKITTI raw ID -> SIH Class mapping
DEFAULT_RAW_TO_SIH: Dict[int, int] = {
    # Class 0: drivable_terrain
    40: 0,   # road
    44: 0,   # parking
    60: 0,   # lane-marking

    # Class 1: non_drivable_terrain
    48: 1,   # sidewalk
    49: 1,   # other-ground
    72: 1,   # terrain

    # Class 2: static_obstacle
    50: 2,   # building
    51: 2,   # fence
    52: 2,   # other-structure
    70: 2,   # vegetation
    71: 2,   # trunk
    80: 2,   # pole
    81: 2,   # traffic-sign
    99: 2,   # other-object

    # Class 3: dynamic_object
    10: 3,   # car
    11: 3,   # bicycle
    13: 3,   # bus
    15: 3,   # motorcycle
    16: 3,   # on-rails
    18: 3,   # truck
    20: 3,   # other-vehicle
    30: 3,   # person
    31: 3,   # bicyclist
    32: 3,   # motorcyclist
    252: 3,  # moving-car
    253: 3,  # moving-bicyclist
    254: 3,  # moving-pedestrian
    255: 3,  # moving-motorcyclist
    256: 3,  # moving-on-rails
    257: 3,  # moving-bus
    258: 3,  # moving-truck
    259: 3,  # moving-other-vehicle

    # Class 255: ignore
    0: 255,  # unlabeled / noise
    1: 255,  # outlier
}

DEFAULT_RAW_CLASS_NAMES: Dict[int, str] = {
    0: "unlabeled",
    1: "outlier",
    10: "car",
    11: "bicycle",
    13: "bus",
    15: "motorcycle",
    16: "on-rails",
    18: "truck",
    20: "other-vehicle",
    30: "person",
    31: "bicyclist",
    32: "motorcyclist",
    40: "road",
    44: "parking",
    48: "sidewalk",
    49: "other-ground",
    50: "building",
    51: "fence",
    52: "other-structure",
    60: "lane-marking",
    70: "vegetation",
    71: "trunk",
    72: "terrain",
    80: "pole",
    81: "traffic-sign",
    99: "other-object",
    252: "moving-car",
    253: "moving-bicyclist",
    254: "moving-pedestrian",
    255: "moving-motorcyclist",
    256: "moving-on-rails",
    257: "moving-bus",
    258: "moving-truck",
    259: "moving-other-vehicle",
}


@dataclass
class LabelDistributionItem:
    """Frequency and percentage metric for an individual label class."""
    class_id: int
    class_name: str
    count: int
    percentage: float


@dataclass
class RawLabelAuditItem:
    """Audit entry mapping a raw dataset label to its target SIH class."""
    raw_id: int
    raw_name: str
    sih_id: int
    sih_name: str
    count: int
    percentage: float


@dataclass
class LabelRemapReport:
    """Comprehensive audit report comparing raw dataset labels and remapped SIH classes."""
    total_points: int
    raw_unique_count: int
    sih_unique_count: int
    raw_distribution: List[LabelDistributionItem]
    sih_distribution: List[LabelDistributionItem]
    audit_table: List[RawLabelAuditItem]
    mapped_count: int
    mapped_percentage: float
    ignored_count: int
    ignored_percentage: float
    unmapped_ids: List[int]
    passed: bool


def validate_mapped_labels(mapped_labels: np.ndarray) -> bool:
    """Strictly validate that all mapped label values belong to {0, 1, 2, 3, 255}.

    Args:
        mapped_labels: 1D array of remapped label IDs.

    Returns:
        bool: True if valid.

    Raises:
        LabelMappingError: If any value is outside {0, 1, 2, 3, 255}.
    """
    if mapped_labels.size == 0:
        return True

    unique_vals = set(np.unique(mapped_labels))
    invalid_vals = unique_vals - VALID_SIH_IDS

    if invalid_vals:
        raise LabelMappingError(
            f"Invalid SIH label IDs detected: {invalid_vals}. "
            f"All labels must strictly belong to {VALID_SIH_IDS}."
        )

    return True


class SemanticLabelRemapper:
    """High-performance vectorized label remapper for SIH 4-Class ontology."""

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], Path, str]] = None,
        raw_to_sih: Optional[Dict[int, int]] = None,
        ignore_id: int = SIH_IGNORE,
    ):
        """Initialize label remapper.

        Args:
            config: Optional dict or path to YAML config file.
            raw_to_sih: Optional explicit raw ID -> SIH ID mapping dictionary.
            ignore_id: Target class ID for ignored/unmapped points (default: 255).
        """
        self.ignore_id = ignore_id
        self.raw_to_sih: Dict[int, int] = dict(DEFAULT_RAW_TO_SIH)
        self.raw_class_names: Dict[int, str] = dict(DEFAULT_RAW_CLASS_NAMES)
        self.sih_class_names: Dict[int, str] = dict(SIH_CLASS_NAMES)

        if config is not None:
            self._load_config(config)

        if raw_to_sih is not None:
            self.raw_to_sih.update(raw_to_sih)

        self._validate_configuration()
        self._build_lookup_table()

    def _load_config(self, config: Union[Dict[str, Any], Path, str]) -> None:
        """Load mapping definitions from dictionary or YAML file."""
        if isinstance(config, (str, Path)):
            path = Path(config)
            if not path.is_file():
                raise LabelMappingError(f"Configuration file not found: {path.resolve()}")
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        elif isinstance(config, dict):
            data = config
        else:
            raise TypeError(f"Invalid config type: {type(config)}")

        # Parse ontology
        ont_data = data.get("ontology", {})
        if "ignore_id" in ont_data:
            self.ignore_id = int(ont_data["ignore_id"])

        # Parse SIH classes
        classes_data = data.get("classes", {})
        for cid_str, info in classes_data.items():
            cid = int(cid_str)
            name = info.get("name") if isinstance(info, dict) else str(info)
            self.sih_class_names[cid] = name

        # Parse raw_to_sih mappings
        raw_map = data.get("raw_to_sih", {})
        for raw_id_str, target_id in raw_map.items():
            self.raw_to_sih[int(raw_id_str)] = int(target_id)

        # Parse raw class names
        raw_names = data.get("raw_class_names", {})
        for raw_id_str, name in raw_names.items():
            self.raw_class_names[int(raw_id_str)] = str(name)

    def _validate_configuration(self) -> None:
        """Verify that configuration defines all 4 SIH classes + ignore and valid mappings."""
        if self.ignore_id != SIH_IGNORE:
            raise LabelMappingError(
                f"Invalid ignore ID: {self.ignore_id}. Must be {SIH_IGNORE}."
            )

        for raw_id, target_id in self.raw_to_sih.items():
            if target_id not in VALID_SIH_IDS:
                raise LabelMappingError(
                    f"Raw label ID {raw_id} mapped to invalid SIH class {target_id}. "
                    f"Must be one of {VALID_SIH_IDS}."
                )

    def _build_lookup_table(self) -> None:
        """Construct fast O(1) vectorized NumPy lookup table."""
        max_raw_id = max(self.raw_to_sih.keys()) if self.raw_to_sih else 259
        max_size = max(max_raw_id + 1, 300)

        # Initialize LUT with default ignore_id (255)
        self._lut = np.full(max_size, self.ignore_id, dtype=np.uint8)

        for raw_id, target_id in self.raw_to_sih.items():
            if 0 <= raw_id < max_size:
                self._lut[raw_id] = target_id

    def remap(self, raw_labels: np.ndarray) -> np.ndarray:
        """Remap a 1D array of raw SemanticKITTI label IDs to SIH classes {0, 1, 2, 3, 255}.

        Args:
            raw_labels: 1D array of raw label IDs with integer dtype.

        Returns:
            np.ndarray: 1D array of shape (N,) with dtype uint8 containing remapped SIH classes.

        Raises:
            LabelMappingError: If input is not 1D or output validation fails.
        """
        if raw_labels.ndim != 1:
            raise LabelMappingError(
                f"Expected 1D label array, got shape {raw_labels.shape}"
            )

        if raw_labels.size == 0:
            return np.empty(0, dtype=np.uint8)

        # Fast vectorized lookup
        lut_size = len(self._lut)
        in_bounds = (raw_labels >= 0) & (raw_labels < lut_size)

        mapped = np.full(raw_labels.shape, self.ignore_id, dtype=np.uint8)
        mapped[in_bounds] = self._lut[raw_labels[in_bounds]]

        # Strict validation
        validate_mapped_labels(mapped)

        return mapped

    def audit(
        self,
        raw_labels: np.ndarray,
        mapped_labels: Optional[np.ndarray] = None,
    ) -> LabelRemapReport:
        """Perform comprehensive statistical audit of raw labels vs remapped SIH classes.

        Args:
            raw_labels: 1D array of raw dataset labels.
            mapped_labels: Optional pre-computed mapped labels. If None, remap is executed.

        Returns:
            LabelRemapReport: Detailed audit report.
        """
        if mapped_labels is None:
            mapped_labels = self.remap(raw_labels)

        total_points = len(raw_labels)
        if total_points == 0:
            return LabelRemapReport(
                total_points=0,
                raw_unique_count=0,
                sih_unique_count=0,
                raw_distribution=[],
                sih_distribution=[],
                audit_table=[],
                mapped_count=0,
                mapped_percentage=0.0,
                ignored_count=0,
                ignored_percentage=0.0,
                unmapped_ids=[],
                passed=True,
            )

        # Raw distribution
        raw_unique, raw_counts = np.unique(raw_labels, return_counts=True)
        raw_distribution: List[LabelDistributionItem] = []
        for rid, count in zip(raw_unique, raw_counts):
            r_name = self.raw_class_names.get(int(rid), f"raw_class_{rid}")
            raw_distribution.append(
                LabelDistributionItem(
                    class_id=int(rid),
                    class_name=r_name,
                    count=int(count),
                    percentage=(float(count) / float(total_points)) * 100.0,
                )
            )

        # SIH distribution
        sih_unique, sih_counts = np.unique(mapped_labels, return_counts=True)
        sih_distribution: List[LabelDistributionItem] = []
        for sid, count in zip(sih_unique, sih_counts):
            s_name = self.sih_class_names.get(int(sid), f"sih_class_{sid}")
            sih_distribution.append(
                LabelDistributionItem(
                    class_id=int(sid),
                    class_name=s_name,
                    count=int(count),
                    percentage=(float(count) / float(total_points)) * 100.0,
                )
            )

        # Audit mapping table
        audit_table: List[RawLabelAuditItem] = []
        unmapped_ids: List[int] = []
        ignored_count = 0
        mapped_count = 0

        for rid, count in zip(raw_unique, raw_counts):
            rid_int = int(rid)
            r_name = self.raw_class_names.get(rid_int, f"raw_class_{rid_int}")
            is_configured = rid_int in self.raw_to_sih

            if not is_configured:
                unmapped_ids.append(rid_int)

            target_sih = self.raw_to_sih.get(rid_int, self.ignore_id)
            target_name = self.sih_class_names.get(target_sih, f"sih_class_{target_sih}")

            pct = (float(count) / float(total_points)) * 100.0
            audit_table.append(
                RawLabelAuditItem(
                    raw_id=rid_int,
                    raw_name=r_name,
                    sih_id=target_sih,
                    sih_name=target_name,
                    count=int(count),
                    percentage=pct,
                )
            )

            if target_sih == self.ignore_id:
                ignored_count += int(count)
            else:
                mapped_count += int(count)

        ignored_pct = (float(ignored_count) / float(total_points)) * 100.0
        mapped_pct = (float(mapped_count) / float(total_points)) * 100.0

        passed = (
            len(raw_labels) == len(mapped_labels)
            and set(np.unique(mapped_labels)).issubset(VALID_SIH_IDS)
        )

        return LabelRemapReport(
            total_points=total_points,
            raw_unique_count=len(raw_unique),
            sih_unique_count=len(sih_unique),
            raw_distribution=raw_distribution,
            sih_distribution=sih_distribution,
            audit_table=audit_table,
            mapped_count=mapped_count,
            mapped_percentage=mapped_pct,
            ignored_count=ignored_count,
            ignored_percentage=ignored_pct,
            unmapped_ids=unmapped_ids,
            passed=passed,
        )
