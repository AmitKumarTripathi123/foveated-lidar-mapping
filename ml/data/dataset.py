"""LiDAR Dataset Loader, Validation Engine, and PyTorch Dataset Abstraction.

Coordinate Convention:
    X -> forward / ahead (+X = vehicle front, -X = vehicle rear)
    Y -> left / port     (+Y = vehicle left,  -Y = vehicle right)
    Z -> upward / height (+Z = upward,        -Z = below sensor / ground direction)
    Origin (0, 0, 0)     = LiDAR sensor center

Point Representation:
    Shape: (N, 4)
    Channels: [x, y, z, intensity]
    Dtype: np.float32

Label Representation:
    Shape: (N,)
    Raw dtype: np.uint32
    Semantic extraction: raw_labels & 0xFFFF (lower 16 bits)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

# Optional PyTorch import
try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    TorchDataset = object  # type: ignore


class LiDARError(Exception):
    """Base exception for LiDAR data processing errors."""
    pass


class LiDARFileNotFoundError(LiDARError, FileNotFoundError):
    """Raised when a requested LiDAR scan or label file is not found."""
    pass


class LiDARFormatError(LiDARError, ValueError):
    """Raised when a LiDAR scan or label file contains invalid data or corrupt layout."""
    pass


class LiDARAlignmentError(LiDARError, ValueError):
    """Raised when point count does not match label count."""
    pass


class LiDARDataValidationError(LiDARError, ValueError):
    """Raised when point cloud contains invalid values such as NaNs or Infs."""
    pass


@dataclass
class CoordinateStats:
    """Statistical summary for a single coordinate axis."""
    min: float
    max: float
    mean: float
    std: float


@dataclass
class IntensityStats:
    """Statistical summary for point reflectance intensity."""
    min: float
    max: float
    mean: float
    std: float


@dataclass
class PointCloudStats:
    """Comprehensive point cloud statistics."""
    num_points: int
    x: CoordinateStats
    y: CoordinateStats
    z: CoordinateStats
    intensity: IntensityStats
    has_nan: bool
    has_inf: bool


@dataclass
class LabelDistributionEntry:
    """Frequency and percentage for a specific semantic label."""
    label_id: int
    count: int
    percentage: float


@dataclass
class ValidationReport:
    """Comprehensive data quality and validation report."""
    scan_path: Path
    label_path: Optional[Path]
    scan_exists: bool
    label_exists: bool
    scan_readable: bool
    label_readable: bool
    point_dtype: str
    label_dtype: Optional[str]
    points_shape: Tuple[int, ...]
    labels_shape: Optional[Tuple[int, ...]]
    num_points: int
    num_labels: Optional[int]
    alignment_pass: bool
    nan_check_pass: bool
    inf_check_pass: bool
    stats: Optional[PointCloudStats]
    label_distribution: List[LabelDistributionEntry] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    passed: bool = False


def load_point_cloud(bin_path: Union[str, Path]) -> np.ndarray:
    """Load a raw LiDAR scan from a binary (.bin) file.

    Args:
        bin_path: Path to the .bin file containing float32 points [x, y, z, intensity].

    Returns:
        np.ndarray: Array of shape (N, 4) with dtype np.float32.

    Raises:
        LiDARFileNotFoundError: If the file does not exist.
        LiDARFormatError: If file size is not a multiple of 4 float32 values (16 bytes).
        LiDARError: If the file cannot be read.
    """
    path = Path(bin_path)
    if not path.is_file():
        raise LiDARFileNotFoundError(f"LiDAR file not found: {path.resolve()}")

    try:
        raw = np.fromfile(str(path), dtype=np.float32)
    except Exception as exc:
        raise LiDARError(f"Failed to read LiDAR file '{path}': {exc}") from exc

    if raw.size == 0:
        return np.empty((0, 4), dtype=np.float32)

    if raw.size % 4 != 0:
        raise LiDARFormatError(
            f"Invalid LiDAR file '{path.name}': raw size {raw.size} float32 values "
            f"({raw.nbytes} bytes) is not divisible by 4. "
            f"Expected groups of 4 float32 values [x, y, z, intensity]."
        )

    points = raw.reshape(-1, 4)
    return points


def load_labels(label_path: Union[str, Path]) -> np.ndarray:
    """Load and decode semantic labels from a .label file.

    In SemanticKITTI format, labels are stored as 32-bit unsigned integers (uint32).
    The lower 16 bits represent the semantic label ID, and the upper 16 bits
    represent the instance ID.

    Args:
        label_path: Path to the .label file containing uint32 entries.

    Returns:
        np.ndarray: 1D array of shape (N,) containing extracted semantic label IDs (uint32).

    Raises:
        LiDARFileNotFoundError: If the file does not exist.
        LiDARFormatError: If labels cannot be parsed or are malformed.
        LiDARError: If the file cannot be read.
    """
    path = Path(label_path)
    if not path.is_file():
        raise LiDARFileNotFoundError(f"Label file not found: {path.resolve()}")

    try:
        raw_labels = np.fromfile(str(path), dtype=np.uint32)
    except Exception as exc:
        raise LiDARError(f"Failed to read label file '{path}': {exc}") from exc

    # Extract lower 16 bits for semantic label
    semantic_labels = raw_labels & 0xFFFF

    if semantic_labels.ndim != 1:
        raise LiDARFormatError(
            f"Expected 1D semantic label array, but got shape {semantic_labels.shape}"
        )

    return semantic_labels


def validate_point_label_alignment(points: np.ndarray, labels: np.ndarray) -> bool:
    """Strictly validate point-label count alignment.

    Args:
        points: Point cloud array of shape (N, 4).
        labels: Label array of shape (M,).

    Returns:
        bool: True if counts match exactly.

    Raises:
        LiDARAlignmentError: If points.shape[0] != labels.shape[0].
    """
    num_points = points.shape[0]
    num_labels = labels.shape[0]

    if num_points != num_labels:
        raise LiDARAlignmentError(
            f"Point-label count mismatch:\n"
            f"  Points: {num_points}\n"
            f"  Labels: {num_labels}\n"
            f"Strict alignment failed. No points or labels will be silently truncated or padded."
        )

    return True


def validate_data_integrity(
    points: np.ndarray, labels: Optional[np.ndarray] = None
) -> Dict[str, bool]:
    """Check point cloud and labels for invalid numerical values (NaN, Inf).

    Args:
        points: Point cloud array of shape (N, 4).
        labels: Optional label array.

    Returns:
        Dict[str, bool]: Results dictionary with 'has_nan' and 'has_inf'.
    """
    has_nan = bool(np.isnan(points).any())
    has_inf = bool(np.isinf(points).any())

    return {
        "has_nan": has_nan,
        "has_inf": has_inf,
        "valid": not (has_nan or has_inf),
    }


def compute_point_cloud_stats(points: np.ndarray) -> PointCloudStats:
    """Compute coordinate and intensity statistics for a point cloud.

    Args:
        points: Point cloud array of shape (N, 4) with columns [x, y, z, intensity].

    Returns:
        PointCloudStats: Comprehensive coordinate and intensity statistics.
    """
    num_points = points.shape[0]
    if num_points == 0:
        empty_coord = CoordinateStats(0.0, 0.0, 0.0, 0.0)
        empty_int = IntensityStats(0.0, 0.0, 0.0, 0.0)
        return PointCloudStats(
            num_points=0,
            x=empty_coord,
            y=empty_coord,
            z=empty_coord,
            intensity=empty_int,
            has_nan=False,
            has_inf=False,
        )

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    intensity = points[:, 3]

    integrity = validate_data_integrity(points)

    return PointCloudStats(
        num_points=num_points,
        x=CoordinateStats(
            min=float(np.min(x)),
            max=float(np.max(x)),
            mean=float(np.mean(x)),
            std=float(np.std(x)),
        ),
        y=CoordinateStats(
            min=float(np.min(y)),
            max=float(np.max(y)),
            mean=float(np.mean(y)),
            std=float(np.std(y)),
        ),
        z=CoordinateStats(
            min=float(np.min(z)),
            max=float(np.max(z)),
            mean=float(np.mean(z)),
            std=float(np.std(z)),
        ),
        intensity=IntensityStats(
            min=float(np.min(intensity)),
            max=float(np.max(intensity)),
            mean=float(np.mean(intensity)),
            std=float(np.std(intensity)),
        ),
        has_nan=integrity["has_nan"],
        has_inf=integrity["has_inf"],
    )


def compute_label_distribution(labels: np.ndarray) -> List[LabelDistributionEntry]:
    """Compute frequency and percentage distribution of semantic labels.

    Args:
        labels: 1D array of semantic label IDs.

    Returns:
        List[LabelDistributionEntry]: Ordered list of label frequency entries.
    """
    total = len(labels)
    if total == 0:
        return []

    unique_labels, counts = np.unique(labels, return_counts=True)
    distribution = []
    for label_id, count in zip(unique_labels, counts):
        percentage = (float(count) / float(total)) * 100.0
        distribution.append(
            LabelDistributionEntry(
                label_id=int(label_id),
                count=int(count),
                percentage=percentage,
            )
        )
    return distribution


def validate_dataset_pair(
    bin_path: Union[str, Path], label_path: Optional[Union[str, Path]] = None
) -> ValidationReport:
    """Perform a full validation workflow on a LiDAR scan and optional label file.

    Args:
        bin_path: Path to the .bin point cloud file.
        label_path: Optional path to the .label semantic label file.

    Returns:
        ValidationReport: Structured validation results.
    """
    b_path = Path(bin_path)
    l_path = Path(label_path) if label_path is not None else None

    errors: List[str] = []
    scan_exists = b_path.is_file()
    label_exists = l_path.is_file() if l_path is not None else False

    if not scan_exists:
        errors.append(f"Point cloud file not found: {b_path.resolve()}")

    if l_path is not None and not label_exists:
        errors.append(f"Label file not found: {l_path.resolve()}")

    points: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    scan_readable = False
    label_readable = False

    if scan_exists:
        try:
            points = load_point_cloud(b_path)
            scan_readable = True
        except Exception as exc:
            errors.append(f"Failed to load point cloud: {exc}")

    if l_path is not None and label_exists:
        try:
            labels = load_labels(l_path)
            label_readable = True
        except Exception as exc:
            errors.append(f"Failed to load labels: {exc}")

    points_shape = points.shape if points is not None else (0,)
    labels_shape = labels.shape if labels is not None else None
    num_points = points.shape[0] if points is not None else 0
    num_labels = labels.shape[0] if labels is not None else None

    # Alignment check
    alignment_pass = False
    if points is not None and labels is not None:
        try:
            validate_point_label_alignment(points, labels)
            alignment_pass = True
        except LiDARAlignmentError as exc:
            errors.append(str(exc))
    elif points is not None and labels is None:
        alignment_pass = True  # Point cloud only scan

    # Integrity & statistics
    stats: Optional[PointCloudStats] = None
    nan_check_pass = False
    inf_check_pass = False

    if points is not None:
        stats = compute_point_cloud_stats(points)
        nan_check_pass = not stats.has_nan
        inf_check_pass = not stats.has_inf
        if not nan_check_pass:
            errors.append("Point cloud contains NaN values.")
        if not inf_check_pass:
            errors.append("Point cloud contains Inf values.")

    # Label distribution
    label_dist: List[LabelDistributionEntry] = []
    if labels is not None:
        label_dist = compute_label_distribution(labels)

    overall_pass = (
        scan_exists
        and scan_readable
        and (l_path is None or (label_exists and label_readable and alignment_pass))
        and nan_check_pass
        and inf_check_pass
        and len(errors) == 0
    )

    return ValidationReport(
        scan_path=b_path,
        label_path=l_path,
        scan_exists=scan_exists,
        label_exists=label_exists,
        scan_readable=scan_readable,
        label_readable=label_readable,
        point_dtype=str(points.dtype) if points is not None else "unknown",
        label_dtype=str(labels.dtype) if labels is not None else None,
        points_shape=points_shape,
        labels_shape=labels_shape,
        num_points=num_points,
        num_labels=num_labels,
        alignment_pass=alignment_pass,
        nan_check_pass=nan_check_pass,
        inf_check_pass=inf_check_pass,
        stats=stats,
        label_distribution=label_dist,
        errors=errors,
        passed=overall_pass,
    )


class LidarDataset(TorchDataset):
    """PyTorch-compatible Lazy-Loading LiDAR Dataset for SemanticKITTI sequences.

    Discovers matching velodyne/*.bin and labels/*.label files across sequences.
    Performs on-demand scan loading, preprocessing, and optional SIH label remapping
    without caching millions of points in memory.
    """

    def __init__(
        self,
        root: Union[str, Path] = "dataset",
        split: str = "train",
        sequences: Optional[List[str]] = None,
        preprocessor: Optional[Callable] = None,
        label_remapper: Optional[Callable] = None,
        to_tensor: bool = False,
        require_labels: bool = True,
    ):
        """Initialize LidarDataset.

        Args:
            root: Root path to dataset (containing sequences/ directory).
            split: Dataset split identifier ('train', 'val', 'test').
            sequences: Optional explicit list of sequence IDs (e.g. ['00', '01']).
            preprocessor: Optional callable (e.g., LidarPreprocessor instance).
            label_remapper: Optional callable (e.g., SemanticLabelRemapper instance).
            to_tensor: Whether __getitem__ converts outputs to PyTorch tensors.
            require_labels: If True, only includes frames that have matching label files.
        """
        self.root = Path(root)
        self.split = split
        self.preprocessor = preprocessor
        self.label_remapper = label_remapper
        self.to_tensor = to_tensor
        self.require_labels = require_labels

        self.seq_dir = self.root / "sequences" if (self.root / "sequences").is_dir() else self.root
        if not self.seq_dir.is_dir():
            raise LiDARFileNotFoundError(f"Dataset sequences directory not found at: {self.seq_dir.resolve()}")

        # Determine sequences to include
        if sequences is not None:
            self.sequence_ids = [str(s).zfill(2) for s in sequences]
        else:
            # Default split mapping
            if split == "train":
                self.sequence_ids = ["00"]
            elif split in ("val", "validation"):
                self.sequence_ids = ["00"]
            elif split == "test":
                self.sequence_ids = ["00"]
            else:
                self.sequence_ids = ["00"]

        # Discover matching files lazily
        self.samples: List[Dict[str, Any]] = []
        self._index_dataset()

    def _index_dataset(self) -> None:
        """Discover matching .bin and .label pairs across specified sequences."""
        self.samples.clear()

        for seq_id in self.sequence_ids:
            seq_path = self.seq_dir / seq_id
            if not seq_path.is_dir():
                continue

            velo_dir = seq_path / "velodyne"
            labels_dir = seq_path / "labels"

            if not velo_dir.is_dir():
                continue

            bin_files = sorted(velo_dir.glob("*.bin"))
            for bin_file in bin_files:
                frame_stem = bin_file.stem
                label_file = labels_dir / f"{frame_stem}.label"

                if self.require_labels and not label_file.is_file():
                    continue

                self.samples.append({
                    "sequence": seq_id,
                    "frame": frame_stem,
                    "point_path": bin_file,
                    "label_path": label_file if label_file.is_file() else None,
                })

    def __len__(self) -> int:
        """Return total number of discovered frames."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load and return a single preprocessed and optionally remapped sample.

        Args:
            idx: Sample index.

        Returns:
            Dict containing:
                - 'points': np.ndarray or torch.FloatTensor of shape (N, 4)
                - 'labels': np.ndarray or torch.LongTensor of shape (N,)
                - 'metadata': dict of frame information and preprocessing metrics
        """
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of bounds for dataset with {len(self.samples)} samples.")

        sample_info = self.samples[idx]
        points = load_point_cloud(sample_info["point_path"])

        labels: Optional[np.ndarray] = None
        if sample_info["label_path"] is not None:
            labels = load_labels(sample_info["label_path"])
            validate_point_label_alignment(points, labels)

        metadata = {
            "sequence": sample_info["sequence"],
            "frame": sample_info["frame"],
            "scan_path": str(sample_info["point_path"]),
            "label_path": str(sample_info["label_path"]) if sample_info["label_path"] else None,
        }

        # Apply preprocessor if provided
        if self.preprocessor is not None:
            processed = self.preprocessor(points, labels, metadata=metadata)
            points = processed.points
            labels = processed.labels
            metadata = processed.metadata

        # Apply SIH label remapping if provided
        if self.label_remapper is not None and labels is not None:
            if hasattr(self.label_remapper, "remap"):
                labels = self.label_remapper.remap(labels)
            elif callable(self.label_remapper):
                labels = self.label_remapper(labels)
            validate_point_label_alignment(points, labels)

        # Convert to PyTorch tensors if requested
        if self.to_tensor:
            if not _HAS_TORCH:
                raise RuntimeError("PyTorch is not installed. Cannot convert to PyTorch tensors.")
            points_tensor = torch.from_numpy(points.copy()).float()
            labels_tensor = (
                torch.from_numpy(labels.astype(np.int64).copy()).long()
                if labels is not None
                else None
            )
            return {
                "points": points_tensor,
                "labels": labels_tensor,
                "metadata": metadata,
            }

        return {
            "points": points,
            "labels": labels,
            "metadata": metadata,
        }


def lidar_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collation function for PyTorch DataLoader.

    Safely handles both uniform (fixed sampled) and variable-sized point clouds:
      - If all samples have identical point count N, stacks points into (B, N, 4)
        and labels into (B, N).
      - If samples have variable point counts, collates into lists of tensors
        without crashing or applying silent padding.

    Args:
        batch: List of dictionaries returned by LidarDataset.__getitem__.

    Returns:
        Dict with collated 'points', 'labels', and 'metadata'.
    """
    if not batch:
        return {}

    has_labels = batch[0]["labels"] is not None
    is_tensor = _HAS_TORCH and isinstance(batch[0]["points"], torch.Tensor)

    # Check if all point counts match
    point_counts = [
        b["points"].shape[0] if not is_tensor else b["points"].size(0)
        for b in batch
    ]
    all_same_size = len(set(point_counts)) == 1

    metadata_list = [b["metadata"] for b in batch]

    if all_same_size and is_tensor:
        points_batch = torch.stack([b["points"] for b in batch], dim=0)
        labels_batch = (
            torch.stack([b["labels"] for b in batch], dim=0) if has_labels else None
        )
    elif all_same_size and not is_tensor:
        points_batch = np.stack([b["points"] for b in batch], axis=0)
        labels_batch = (
            np.stack([b["labels"] for b in batch], axis=0) if has_labels else None
        )
    else:
        # Variable point counts: return list of tensors/arrays
        points_batch = [b["points"] for b in batch]  # type: ignore
        labels_batch = [b["labels"] for b in batch] if has_labels else None  # type: ignore

    return {
        "points": points_batch,
        "labels": labels_batch,
        "metadata": metadata_list,
    }
