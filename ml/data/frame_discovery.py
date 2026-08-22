"""Authoritative Frame Discovery & Dataset Integrity Audit Engine (Phase 7.1).

Discovers all sequence folders, velodyne point clouds, and label files
across local or external dataset roots, providing standardized FrameRecord DTOs
and comprehensive multi-frame integrity verification.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ml.data.dataset import load_point_cloud, load_labels, validate_data_integrity


@dataclass
class FrameRecord:
    """Standardized metadata and path DTO for a single LiDAR frame."""
    sequence_id: str
    frame_id: str
    point_cloud_path: str
    label_path: Optional[str]
    has_label: bool = True
    is_matched: bool = True
    point_count: Optional[int] = None
    label_count: Optional[int] = None
    finite_status: bool = True
    alignment_status: bool = True


def discover_frames(
    dataset_root: Optional[Union[str, Path]] = None,
    allow_external: bool = True,
) -> List[FrameRecord]:
    """Discover all valid LiDAR frames across all sequences under dataset_root.

    Args:
        dataset_root: Path to dataset root directory. If None, checks DATASET_ROOT env or defaults to "dataset".
        allow_external: If True, supports external paths outside the repository.

    Returns:
        List[FrameRecord]: Sorted list of discovered frame records.
    """
    if dataset_root is None:
        dataset_root = os.environ.get("DATASET_ROOT", "dataset")

    root = Path(dataset_root)
    if not root.is_dir():
        return []

    seq_dir = root / "sequences" if (root / "sequences").is_dir() else root
    cwd = Path.cwd()

    records: List[FrameRecord] = []

    for s_path in sorted(seq_dir.iterdir()):
        if not s_path.is_dir():
            continue

        seq_id = s_path.name
        velo_dir = s_path / "velodyne"
        label_dir = s_path / "labels"

        if not velo_dir.is_dir():
            continue

        bin_files = sorted(velo_dir.glob("*.bin"))
        for b_file in bin_files:
            frame_stem = b_file.stem
            l_file = label_dir / f"{frame_stem}.label"
            has_label = l_file.is_file()

            # Generate portable relative path if within cwd, else normalized path
            try:
                rel_b = b_file.resolve().relative_to(cwd.resolve()).as_posix()
            except ValueError:
                rel_b = b_file.as_posix()

            if has_label:
                try:
                    rel_l = l_file.resolve().relative_to(cwd.resolve()).as_posix()
                except ValueError:
                    rel_l = l_file.as_posix()
            else:
                rel_l = None

            record = FrameRecord(
                sequence_id=seq_id,
                frame_id=frame_stem,
                point_cloud_path=rel_b,
                label_path=rel_l,
                has_label=has_label,
                is_matched=has_label,
            )
            records.append(record)

    return sorted(records, key=lambda r: (r.sequence_id, r.frame_id))


def audit_discovered_frames(records: List[FrameRecord]) -> Dict[str, Any]:
    """Perform deep numerical and semantic audit of all discovered frame records.

    Args:
        records: List of FrameRecord objects from discover_frames.

    Returns:
        Dict: Comprehensive audit dictionary containing summary and per-frame metrics.
    """
    total_points = 0
    total_labels = 0
    matched_count = 0
    unmatched_count = 0
    corrupted_count = 0
    global_raw_classes: Dict[int, int] = {}
    frame_details: List[Dict[str, Any]] = []

    for rec in records:
        b_path = Path(rec.point_cloud_path)
        l_path = Path(rec.label_path) if rec.label_path else None

        try:
            pts = load_point_cloud(b_path)
            num_pts = pts.shape[0]
            integrity = validate_data_integrity(pts)
            rec.point_count = num_pts
            rec.finite_status = not (integrity["has_nan"] or integrity["has_inf"])
        except Exception:
            corrupted_count += 1
            rec.finite_status = False
            continue

        num_lbls = None
        raw_dist: Dict[int, int] = {}
        if l_path and l_path.is_file():
            try:
                lbls = load_labels(l_path)
                num_lbls = len(lbls)
                rec.label_count = num_lbls
                u_lbls, counts = np.unique(lbls, return_counts=True)
                for u, c in zip(u_lbls, counts):
                    raw_dist[int(u)] = int(c)
                    global_raw_classes[int(u)] = global_raw_classes.get(int(u), 0) + int(c)
            except Exception:
                corrupted_count += 1
                continue

        aligned = (num_lbls is not None) and (num_pts == num_lbls)
        rec.alignment_status = aligned

        if rec.has_label and aligned:
            matched_count += 1
            total_labels += (num_lbls or 0)
        else:
            unmatched_count += 1

        total_points += num_pts

        frame_details.append({
            "sequence": rec.sequence_id,
            "frame": rec.frame_id,
            "points": num_pts,
            "labels": num_lbls,
            "aligned": aligned,
            "finite": rec.finite_status,
            "raw_classes": raw_dist,
        })

    sequences = sorted(list({r.sequence_id for r in records}))

    return {
        "total_frames_discovered": len(records),
        "total_sequences": len(sequences),
        "sequences": sequences,
        "matched_pairs": matched_count,
        "unmatched_frames": unmatched_count,
        "corrupted_frames": corrupted_count,
        "total_points": total_points,
        "total_labels": total_labels,
        "global_raw_classes": global_raw_classes,
        "frames": frame_details,
    }
