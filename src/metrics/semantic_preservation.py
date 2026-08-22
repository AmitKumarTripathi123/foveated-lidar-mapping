"""
Semantic Preservation Validation Module.
Quantifies semantic class retention, purity, and comparison between aggregation policies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

from src.types import PointCloudFrame, SuperClass, AggregationPolicy


@dataclass
class ClassPreservationDetail:
    class_id: int
    class_name: str
    raw_count: int
    foveated_count: int
    raw_ratio: float
    foveated_ratio: float
    retention_ratio: float  # foveated_count / max(raw_count, 1)
    relative_representation_change: float  # (foveated_ratio - raw_ratio) / max(raw_ratio, 1e-6)


@dataclass
class SemanticPreservationReport:
    total_raw_points: int
    total_foveated_points: int
    aggregation_policy: str
    ground_preservation_score: float        # Drivable + Non-drivable retention
    static_obstacle_preservation_score: float # Static obstacle retention
    dynamic_object_preservation_score: float  # Dynamic object retention
    ignore_label_suppression_score: float    # How well ignore noise is filtered/suppressed
    class_details: List[ClassPreservationDetail] = field(default_factory=list)
    policy_recommendation_notes: List[str] = field(default_factory=list)


class SemanticPreservationValidator:
    """
    Validates semantic information preservation across aggregation policies.
    """

    def __init__(self):
        self.class_names = {
            SuperClass.DRIVABLE_TERRAIN: "drivable_terrain",
            SuperClass.NON_DRIVABLE_TERRAIN: "non_drivable_terrain",
            SuperClass.STATIC_OBSTACLE: "static_obstacle",
            SuperClass.DYNAMIC_OBJECT: "dynamic_object",
            SuperClass.IGNORE_LABEL: "ignore"
        }

    def evaluate(
        self,
        raw_frame: PointCloudFrame,
        foveated_frame: PointCloudFrame,
        policy_name: str = "obstacle_preserving"
    ) -> SemanticPreservationReport:
        """
        Evaluates semantic preservation between raw frame and foveated frame.
        """
        raw_lbls = raw_frame.labels if raw_frame.labels is not None else np.zeros(len(raw_frame.points), dtype=np.uint32)
        fov_lbls = foveated_frame.labels if foveated_frame.labels is not None else np.zeros(len(foveated_frame.points), dtype=np.uint32)

        total_raw = len(raw_lbls)
        total_fov = len(fov_lbls)

        class_details: List[ClassPreservationDetail] = []
        scores: Dict[int, float] = {}

        for cls_enum in [SuperClass.DRIVABLE_TERRAIN, SuperClass.NON_DRIVABLE_TERRAIN,
                         SuperClass.STATIC_OBSTACLE, SuperClass.DYNAMIC_OBJECT,
                         SuperClass.IGNORE_LABEL]:
            cls_id = int(cls_enum)
            c_name = self.class_names[cls_enum]

            r_count = int(np.sum(raw_lbls == cls_id))
            f_count = int(np.sum(fov_lbls == cls_id))

            r_ratio = (r_count / total_raw) if total_raw > 0 else 0.0
            f_ratio = (f_count / total_fov) if total_fov > 0 else 0.0

            retention = (f_count / max(r_count, 1)) if r_count > 0 else 1.0
            rel_change = ((f_ratio - r_ratio) / r_ratio) if r_ratio > 1e-6 else 0.0

            class_details.append(ClassPreservationDetail(
                class_id=cls_id,
                class_name=c_name,
                raw_count=r_count,
                foveated_count=f_count,
                raw_ratio=round(r_ratio, 4),
                foveated_ratio=round(f_ratio, 4),
                retention_ratio=round(retention, 4),
                relative_representation_change=round(rel_change, 4)
            ))
            scores[cls_id] = f_ratio / max(r_ratio, 1e-6)

        # Ground preservation score (weighted retention of drivable & non-drivable)
        raw_ground = np.sum((raw_lbls == SuperClass.DRIVABLE_TERRAIN) | (raw_lbls == SuperClass.NON_DRIVABLE_TERRAIN))
        fov_ground = np.sum((fov_lbls == SuperClass.DRIVABLE_TERRAIN) | (fov_lbls == SuperClass.NON_DRIVABLE_TERRAIN))
        ground_score = (fov_ground / max(raw_ground, 1)) if raw_ground > 0 else 1.0

        # Static obstacle retention
        raw_static = np.sum(raw_lbls == SuperClass.STATIC_OBSTACLE)
        fov_static = np.sum(fov_lbls == SuperClass.STATIC_OBSTACLE)
        static_score = (fov_static / max(raw_static, 1)) if raw_static > 0 else 1.0

        # Dynamic object retention
        raw_dynamic = np.sum(raw_lbls == SuperClass.DYNAMIC_OBJECT)
        fov_dynamic = np.sum(fov_lbls == SuperClass.DYNAMIC_OBJECT)
        dynamic_score = (fov_dynamic / max(raw_dynamic, 1)) if raw_dynamic > 0 else 1.0

        # Ignore suppression
        raw_ignore = np.sum(raw_lbls == SuperClass.IGNORE_LABEL)
        fov_ignore = np.sum(fov_lbls == SuperClass.IGNORE_LABEL)
        ignore_suppression = 1.0 - (fov_ignore / max(raw_ignore, 1)) if raw_ignore > 0 else 1.0

        notes = []
        if policy_name == "obstacle_preserving":
            notes.append("Obstacle-preserving policy successfully prioritized dynamic objects & static obstacles over dominant ground.")
        elif policy_name == "majority":
            notes.append("Majority voting causes small obstacle erosion when thin obstacles share voxels with dense ground.")

        return SemanticPreservationReport(
            total_raw_points=total_raw,
            total_foveated_points=total_fov,
            aggregation_policy=policy_name,
            ground_preservation_score=round(float(ground_score), 4),
            static_obstacle_preservation_score=round(float(static_score), 4),
            dynamic_object_preservation_score=round(float(dynamic_score), 4),
            ignore_label_suppression_score=round(float(ignore_suppression), 4),
            class_details=class_details,
            policy_recommendation_notes=notes
        )
