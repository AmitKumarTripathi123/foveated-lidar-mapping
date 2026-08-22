"""
Semantic Label Mapping and Validation Module.
Transforms dataset-specific raw classes into standardized navigation super-classes:
0: drivable_terrain, 1: non_drivable_terrain, 2: static_obstacle, 3: dynamic_object, 255: ignore.
Detects missing, unexpected, and undefined class mappings with mandatory human warnings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import yaml
import numpy as np

from src.types import SuperClass, PointCloudFrame


@dataclass
class RawLabelStats:
    raw_class_id: int
    point_count: int
    percentage: float
    mapped_super_class: int
    super_class_name: str
    is_known: bool


@dataclass
class SuperClassDistribution:
    super_class_id: int
    super_class_name: str
    point_count: int
    percentage: float


@dataclass
class LabelValidationReport:
    total_labeled_points: int
    raw_label_histogram: List[RawLabelStats]
    super_class_distribution: List[SuperClassDistribution]
    unknown_raw_labels: List[int]
    unmapped_point_count: int
    unmapped_point_percentage: float
    class_imbalance_ratio: float  # max_class_count / min_class_count (non-zero)
    warnings: List[str] = field(default_factory=list)
    dataset_name: str = "Unknown"


class LabelMapper:
    """
    Maps raw LiDAR semantic labels into the 5-class navigation contract.
    """

    def __init__(self, mapping_config_path: Optional[Union[str, Path]] = None, dataset_type: str = "semantickitti"):
        self.mapping: Dict[int, int] = {}
        self.super_class_defs: Dict[int, str] = {
            SuperClass.DRIVABLE_TERRAIN: "drivable_terrain",
            SuperClass.NON_DRIVABLE_TERRAIN: "non_drivable_terrain",
            SuperClass.STATIC_OBSTACLE: "static_obstacle",
            SuperClass.DYNAMIC_OBJECT: "dynamic_object",
            SuperClass.IGNORE_LABEL: "ignore"
        }
        self.dataset_name = dataset_type
        self.has_mapping_warning = False
        self.mapping_warning_msg = ""

        if mapping_config_path and Path(mapping_config_path).exists():
            self.load_mapping(mapping_config_path)
        else:
            self._load_builtin_defaults(dataset_type)

    def load_mapping(self, config_path: Union[str, Path]):
        """Loads mapping configuration from YAML."""
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        self.dataset_name = data.get("dataset", "custom")
        raw_map = data.get("mapping", {})
        # Convert keys and values to int
        self.mapping = {int(k): int(v) for k, v in raw_map.items()}

        if data.get("super_class_definitions"):
            self.super_class_defs = {int(k): str(v) for k, v in data["super_class_definitions"].items()}

        if not data.get("mapping_complete", True) or "warning" in data:
            self.has_mapping_warning = True
            self.mapping_warning_msg = data.get(
                "warning",
                "WARNING: non_drivable_terrain mapping is undefined/incomplete. Human confirmation required."
            )

    def _load_builtin_defaults(self, dataset_type: str):
        """Builtin defaults for SemanticKITTI and SemanticPOSS."""
        if dataset_type.lower() == "semanticposs":
            self.dataset_name = "SemanticPOSS"
            self.has_mapping_warning = True
            self.mapping_warning_msg = (
                "WARNING: non_drivable_terrain mapping is undefined/incomplete. Human confirmation required."
            )
            # Default SemanticPOSS raw to superclass
            self.mapping = {
                0: 255, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3,
                9: 2, 10: 2, 11: 2, 13: 2, 14: 2, 15: 2, 16: 2, 17: 2, 18: 2,
                19: 1, 20: 255, 21: 0, 22: 255
            }
        else:
            # SemanticKITTI default
            self.dataset_name = "SemanticKITTI"
            self.mapping = {
                0: 255, 1: 255, 10: 3, 11: 3, 13: 3, 15: 3, 16: 3, 18: 3, 20: 3,
                30: 3, 31: 3, 32: 3, 40: 0, 44: 0, 48: 1, 49: 1, 50: 2, 51: 2,
                52: 2, 60: 0, 70: 2, 71: 2, 72: 1, 80: 2, 81: 2, 99: 255,
                252: 3, 253: 3, 254: 3, 255: 3, 256: 3, 257: 3, 258: 3, 259: 3
            }

    @property
    def mapping_complete(self) -> bool:
        """Returns True if mapping is verified and complete, False if provisional/requires review."""
        return not self.has_mapping_warning

    @property
    def warning_message(self) -> Optional[str]:
        """Returns the mapping warning message if present."""
        return self.mapping_warning_msg if self.has_mapping_warning else None

    def map_single_label(self, raw_label: int) -> int:
        """Maps a single integer raw label to its target super-class ID."""
        return self.mapping.get(int(raw_label), SuperClass.IGNORE_LABEL)

    def map_labels(self, raw_labels: np.ndarray) -> np.ndarray:
        """
        Maps raw labels array to super-classes: [0, 1, 2, 3, 255].
        Unknown labels are safely mapped to IGNORE_LABEL (255).
        Returns a new array.
        """
        if raw_labels is None or len(raw_labels) == 0:
            return np.empty((0,), dtype=np.uint32)

        mapped = np.full(raw_labels.shape, SuperClass.IGNORE_LABEL, dtype=np.uint32)
        for raw_cls, super_cls in self.mapping.items():
            mask = (raw_labels == raw_cls)
            if np.any(mask):
                mapped[mask] = super_cls
        return mapped

    def map_frame(self, frame: PointCloudFrame) -> PointCloudFrame:
        """Returns a new PointCloudFrame with super-class mapped labels."""
        new_frame = frame.copy()
        new_frame.labels = self.map_labels(frame.labels)
        new_frame.metadata["raw_labels"] = frame.labels
        new_frame.metadata["dataset_mapping"] = self.dataset_name
        return new_frame

    def analyze_and_validate(self, raw_labels: np.ndarray) -> LabelValidationReport:
        """
        Generates full label histogram, super-class distribution, checks unknown labels,
        and computes class imbalance.
        """
        if raw_labels is None or len(raw_labels) == 0:
            return LabelValidationReport(0, [], [], [], 0, 0.0, 1.0, ["Empty label array"], self.dataset_name)

        total = len(raw_labels)
        unique_raw, counts = np.unique(raw_labels, return_counts=True)

        raw_histogram: List[RawLabelStats] = []
        unknown_labels: List[int] = []
        unmapped_count = 0

        # Histogram
        for raw_id, count in zip(unique_raw, counts):
            raw_id_int = int(raw_id)
            count_int = int(count)
            pct = round((count_int / total) * 100.0, 4)
            is_known = raw_id_int in self.mapping
            mapped_cls = self.mapping.get(raw_id_int, int(SuperClass.IGNORE_LABEL))
            cls_name = self.super_class_defs.get(mapped_cls, f"unknown_{mapped_cls}")

            if not is_known:
                unknown_labels.append(raw_id_int)
                unmapped_count += count_int

            raw_histogram.append(RawLabelStats(
                raw_class_id=raw_id_int,
                point_count=count_int,
                percentage=pct,
                mapped_super_class=mapped_cls,
                super_class_name=cls_name,
                is_known=is_known
            ))

        # Super-class aggregation
        mapped_labels = self.map_labels(raw_labels)
        unique_super, super_counts = np.unique(mapped_labels, return_counts=True)
        super_dist: List[SuperClassDistribution] = []
        for s_cls, s_cnt in zip(unique_super, super_counts):
            s_cls_int = int(s_cls)
            s_cnt_int = int(s_cnt)
            super_dist.append(SuperClassDistribution(
                super_class_id=s_cls_int,
                super_class_name=self.super_class_defs.get(s_cls_int, f"super_{s_cls_int}"),
                point_count=s_cnt_int,
                percentage=round((s_cnt_int / total) * 100.0, 4)
            ))

        # Class imbalance
        non_zero_counts = [s.point_count for s in super_dist if s.super_class_id != SuperClass.IGNORE_LABEL and s.point_count > 0]
        imbalance = (max(non_zero_counts) / min(non_zero_counts)) if len(non_zero_counts) > 1 else 1.0

        warnings = []
        if self.has_mapping_warning:
            warnings.append(self.mapping_warning_msg)

        if unknown_labels:
            warnings.append(
                f"Detected {len(unknown_labels)} unknown raw labels: {unknown_labels} "
                f"({unmapped_count} points, {round((unmapped_count/total)*100, 2)}% mapped to IGNORE_LABEL)."
            )

        if imbalance > 50.0:
            warnings.append(
                f"Severe class imbalance detected (ratio {round(imbalance, 1)}:1). "
                f"Dynamic objects / small obstacles constitute a minority of points."
            )

        return LabelValidationReport(
            total_labeled_points=total,
            raw_label_histogram=raw_histogram,
            super_class_distribution=super_dist,
            unknown_raw_labels=unknown_labels,
            unmapped_point_count=unmapped_count,
            unmapped_point_percentage=round((unmapped_count / total) * 100.0, 4) if total > 0 else 0.0,
            class_imbalance_ratio=round(imbalance, 2),
            warnings=warnings,
            dataset_name=self.dataset_name
        )
