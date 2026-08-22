"""LiDAR dataset, preprocessing, foveated adapter, label mapping, and frame discovery."""

from ml.data.dataset import (
    load_point_cloud,
    load_labels,
    validate_data_integrity,
    validate_point_label_alignment,
    LidarDataset,
    lidar_collate_fn,
)
from ml.data.preprocessing import (
    PreprocessingConfig,
    filter_invalid_points,
    apply_range_filter,
    sample_points,
    handle_coordinates,
    handle_intensity,
    preprocess_point_cloud,
    LidarPreprocessor,
)

# Aliases
filter_range = apply_range_filter
normalize_intensity = handle_intensity
normalize_coordinates = handle_coordinates

from ml.data.amit_adapter import (
    FoveatedVoxelSampler,
    FoveatedSamplingReport,
    FoveatedZoneStats,
    voxel_grid_downsample,
)
from ml.data.foveated_dataset import (
    FoveatedLidarDataset,
    normalize_point_count,
)
from ml.data.label_mapping import (
    SemanticLabelRemapper,
    validate_mapped_labels,
    LabelMappingError,
    SIH_DRIVABLE_TERRAIN,
    SIH_NON_DRIVABLE_TERRAIN,
    SIH_STATIC_OBSTACLE,
    SIH_DYNAMIC_OBJECT,
    SIH_IGNORE,
    VALID_SIH_IDS,
    SIH_CLASS_NAMES,
    DEFAULT_RAW_TO_SIH,
)
from ml.data.authoritative_label_mapping import (
    AuthoritativeLabelRemapper,
    AuthoritativeMappingError,
    SEMANTICKITTI_TO_SIH,
    SEMANTICPOSS_TO_SIH,
)
from ml.data.semanticposs_label_mapping import (
    SemanticPOSSLabelRemapper,
    SemanticPOSSMappingError,
    SEMANTICPOSS_RAW_TO_SIH,
)
from ml.data.manifest import (
    discover_dataset,
    audit_dataset,
)
from ml.data.frame_discovery import (
    FrameRecord,
    discover_frames,
    audit_discovered_frames,
)

__all__ = [
    "load_point_cloud",
    "load_labels",
    "validate_data_integrity",
    "validate_point_label_alignment",
    "LidarDataset",
    "lidar_collate_fn",
    "PreprocessingConfig",
    "filter_invalid_points",
    "apply_range_filter",
    "filter_range",
    "sample_points",
    "handle_coordinates",
    "normalize_coordinates",
    "handle_intensity",
    "normalize_intensity",
    "preprocess_point_cloud",
    "LidarPreprocessor",
    "FoveatedVoxelSampler",
    "FoveatedSamplingReport",
    "FoveatedZoneStats",
    "voxel_grid_downsample",
    "FoveatedLidarDataset",
    "normalize_point_count",
    "SemanticLabelRemapper",
    "validate_mapped_labels",
    "LabelMappingError",
    "SIH_DRIVABLE_TERRAIN",
    "SIH_NON_DRIVABLE_TERRAIN",
    "SIH_STATIC_OBSTACLE",
    "SIH_DYNAMIC_OBJECT",
    "SIH_IGNORE",
    "VALID_SIH_IDS",
    "SIH_CLASS_NAMES",
    "DEFAULT_RAW_TO_SIH",
    "AuthoritativeLabelRemapper",
    "AuthoritativeMappingError",
    "SEMANTICKITTI_TO_SIH",
    "SEMANTICPOSS_TO_SIH",
    "SemanticPOSSLabelRemapper",
    "SemanticPOSSMappingError",
    "SEMANTICPOSS_RAW_TO_SIH",
    "discover_dataset",
    "audit_dataset",
    "FrameRecord",
    "discover_frames",
    "audit_discovered_frames",
]
