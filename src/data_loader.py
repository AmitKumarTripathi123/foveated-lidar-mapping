"""
LiDAR Dataset Loader Module.
Supports SemanticKITTI format (.bin, .label), SemanticPOSS format, numpy arrays, and structured frame directories.
Enforces strict label/point count matching and configurable invalid frame handling.
"""

import os
import glob
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Dict, Any, Union
import numpy as np

from src.types import PointCloudFrame, ValidationPolicy


class LiDARDataLoader:
    """
    Configurable LiDAR Data Loader with strict validation of point-to-label alignment.
    """

    def __init__(
        self,
        dataset_path: Optional[Union[str, Path]] = None,
        sequence_id: str = "00",
        validation_policy: ValidationPolicy = ValidationPolicy.SKIP_AND_WARN,
        scan_extension: str = ".bin",
        label_extension: str = ".label",
    ):
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.sequence_id = str(sequence_id)
        self.validation_policy = validation_policy
        self.scan_extension = scan_extension
        self.label_extension = label_extension
        self.invalid_frames: List[Dict[str, Any]] = []

    def discover_frames(self) -> List[Tuple[Path, Optional[Path]]]:
        """
        Discovers point cloud files and matching label files in the sequence.
        Returns list of (scan_path, label_path) tuples.
        """
        if not self.dataset_path or not self.dataset_path.exists():
            return []

        # Check standard SemanticKITTI hierarchy: dataset_path/sequences/{seq}/velodyne/
        seq_dir = self.dataset_path / "sequences" / self.sequence_id
        if not seq_dir.exists():
            seq_dir = self.dataset_path / self.sequence_id
        if not seq_dir.exists():
            seq_dir = self.dataset_path

        velodyne_dir = seq_dir / "velodyne"
        if not velodyne_dir.exists():
            velodyne_dir = seq_dir / "scans"
        if not velodyne_dir.exists():
            velodyne_dir = seq_dir

        labels_dir = seq_dir / "labels"
        if not labels_dir.exists():
            labels_dir = seq_dir

        scan_files = sorted(velodyne_dir.glob(f"*{self.scan_extension}"))
        frames = []
        for scan_file in scan_files:
            stem = scan_file.stem
            label_file = labels_dir / f"{stem}{self.label_extension}"
            if not label_file.exists():
                # Check directly in label dir with same stem
                label_candidates = list(labels_dir.glob(f"{stem}.*"))
                label_file = label_candidates[0] if label_candidates else None
            frames.append((scan_file, label_file if (label_file and label_file.exists()) else None))

        return frames

    def load_frame(
        self,
        scan_path: Union[str, Path],
        label_path: Optional[Union[str, Path]] = None,
        frame_id: Optional[str] = None,
        timestamp: float = 0.0
    ) -> PointCloudFrame:
        """
        Loads a single LiDAR frame and validates point/label length equality.
        """
        scan_path = Path(scan_path)
        if frame_id is None:
            frame_id = scan_path.stem

        if not scan_path.exists():
            raise FileNotFoundError(f"Scan file does not exist: {scan_path}")

        if scan_path.suffix == ".bin":
            file_bytes = os.path.getsize(str(scan_path))
            if file_bytes % 16 != 0:
                msg = f"Binary scan file size ({file_bytes} bytes) is not divisible by 16 (4 float32s per point)."
                return self._handle_invalid_frame(frame_id, str(scan_path), msg)
            raw_scan = np.fromfile(str(scan_path), dtype=np.float32)
            points = raw_scan.reshape(-1, 4) if raw_scan.size > 0 else np.empty((0, 4), dtype=np.float32)

        elif scan_path.suffix in [".npy", ".npz"]:
            data = np.load(str(scan_path))
            if isinstance(data, np.lib.npyio.NpzFile):
                points = data["points"].astype(np.float32)
            else:
                points = data.astype(np.float32)
            if points.ndim != 2 or points.shape[1] < 4:
                msg = f"Invalid numpy shape {points.shape}, expected (N, 4)."
                return self._handle_invalid_frame(frame_id, str(scan_path), msg)
            points = points[:, :4]
        else:
            raise ValueError(f"Unsupported scan file format: {scan_path.suffix}")

        num_points = len(points)

        # 2. Load labels if present
        if label_path and Path(label_path).exists():
            label_p = Path(label_path)
            if label_p.suffix == ".label":
                # SemanticKITTI format: uint32 (lower 16 bits = semantic class, upper 16 = instance)
                raw_labels = np.fromfile(str(label_p), dtype=np.uint32)
                labels = raw_labels & 0xFFFF
            elif label_p.suffix in [".npy", ".npz"]:
                data = np.load(str(label_p))
                labels = data["labels"] if isinstance(data, np.lib.npyio.NpzFile) else data
                labels = labels.astype(np.uint32)
            else:
                raw_labels = np.fromfile(str(label_p), dtype=np.uint16)
                labels = raw_labels.astype(np.uint32)
        else:
            # Default to 0 / unlabeled if no labels file provided
            labels = np.zeros(num_points, dtype=np.uint32)

        num_labels = len(labels)

        # 3. Verify number_of_points == number_of_labels
        if num_points != num_labels:
            mismatch_msg = (
                f"Point/label count mismatch in frame {frame_id}: "
                f"{num_points} points vs {num_labels} labels."
            )
            return self._handle_invalid_frame(
                frame_id=frame_id,
                scan_path=str(scan_path),
                reason=mismatch_msg,
                points=points,
                labels=labels
            )

        return PointCloudFrame(
            points=points.astype(np.float32),
            labels=labels.astype(np.uint32),
            frame_id=frame_id,
            timestamp=timestamp,
            sequence_id=self.sequence_id,
            is_valid=True,
            metadata={"scan_path": str(scan_path), "label_path": str(label_path) if label_path else None}
        )

    def _handle_invalid_frame(
        self,
        frame_id: str,
        scan_path: str,
        reason: str,
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None
    ) -> PointCloudFrame:
        """Enforces validation policy on invalid frame."""
        record = {"frame_id": frame_id, "scan_path": scan_path, "reason": reason}
        self.invalid_frames.append(record)

        if self.validation_policy == ValidationPolicy.STRICT_STOP:
            raise ValueError(f"[STRICT_STOP] Invalid frame detected: {reason}")

        # Return frame marked invalid
        return PointCloudFrame(
            points=points if points is not None else np.empty((0, 4), dtype=np.float32),
            labels=labels if labels is not None else np.empty((0,), dtype=np.uint32),
            frame_id=frame_id,
            sequence_id=self.sequence_id,
            is_valid=False,
            validation_notes=[reason],
            metadata={"scan_path": scan_path, "error": reason}
        )

    def iterate_frames(self) -> Iterator[PointCloudFrame]:
        """Yields PointCloudFrame for all discovered files in sequence."""
        frame_pairs = self.discover_frames()
        for idx, (scan_p, label_p) in enumerate(frame_pairs):
            frame = self.load_frame(scan_p, label_p, timestamp=float(idx) * 0.1)
            if not frame.is_valid and self.validation_policy == ValidationPolicy.ISOLATE:
                continue
            yield frame
