"""
Universal Dataset Discovery, Verification, and Manifest Engine for SemanticPOSS.
Fixes the 1-frame bottleneck by:
  1. Dynamically scanning and indexing all sequences (00, 01, 02, 03, 04, 05).
  2. Bypassing stale 1-frame cache folders.
  3. Stem-matching all .bin and .label pairs across 2,988 frames.
  4. Automatically generating disjoint train (00, 01, 03, 04, 05) and validation (02) splits.
"""

import os
import sys
import glob
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

DEFAULT_SEARCH_PATHS = [
    os.environ.get("DATASET_ROOT", ""),
    "dataset",
    "data/semanticposs_sequence",
    "data/full_semanticposs",
    os.path.expanduser("~/Downloads/SemanticPOSS_dataset"),
    os.path.expanduser("~/Downloads/dataset"),
    "/Users/ankurtiwari/Downloads/SemanticPOSS_dataset",
]

EXPECTED_SEQUENCES = ["00", "01", "02", "03", "04", "05"]
TRAIN_SEQUENCES = ["00", "01", "03", "04", "05"]
VAL_SEQUENCES = ["02"]


def discover_dataset_root(custom_path: Optional[str] = None) -> Optional[Path]:
    """Finds the most complete dataset root directory."""
    candidate_paths = []
    if custom_path:
        candidate_paths.append(Path(custom_path))
    for p in DEFAULT_SEARCH_PATHS:
        if p:
            candidate_paths.append(Path(p))

    best_root = None
    max_frames_found = 0

    for root in candidate_paths:
        if not root.exists():
            continue
        
        # Check direct sequences or nested sequences folder
        seq_root = root / "sequences" if (root / "sequences").exists() else root
        
        frames_count = 0
        for seq in EXPECTED_SEQUENCES:
            s_dir = seq_root / seq
            if s_dir.exists():
                bins = list((s_dir / "velodyne").glob("*.bin")) if (s_dir / "velodyne").exists() else list(s_dir.glob("*.bin"))
                frames_count += len(bins)

        if frames_count > max_frames_found:
            max_frames_found = frames_count
            best_root = root

    return best_root


def build_full_manifest(dataset_root: Optional[str] = None) -> Dict[str, Any]:
    """
    Builds complete train and validation manifests from all discovered sequences.
    """
    root = discover_dataset_root(dataset_root)
    if root is None:
        return {
            "status": "NOT_FOUND",
            "total_frames": 0,
            "train": [],
            "val": [],
            "sequence_breakdown": {}
        }

    seq_root = root / "sequences" if (root / "sequences").exists() else root

    train_pairs: List[Tuple[str, str]] = []
    val_pairs: List[Tuple[str, str]] = []
    seq_breakdown: Dict[str, Dict[str, int]] = {}

    for seq in EXPECTED_SEQUENCES:
        s_dir = seq_root / seq
        if not s_dir.exists():
            seq_breakdown[seq] = {"bins": 0, "labels": 0, "matched": 0}
            continue

        velo_dir = s_dir / "velodyne" if (s_dir / "velodyne").exists() else s_dir
        lbl_dir = s_dir / "labels" if (s_dir / "labels").exists() else s_dir

        bins = {p.stem: str(p) for p in velo_dir.glob("*.bin")}
        lbls = {p.stem: str(p) for p in lbl_dir.glob("*.label")}
        common_stems = sorted(set(bins.keys()) & set(lbls.keys()))

        seq_breakdown[seq] = {
            "bins": len(bins),
            "labels": len(lbls),
            "matched": len(common_stems)
        }

        matched_pairs = [(bins[stem], lbls[stem]) for stem in common_stems]

        if seq in VAL_SEQUENCES:
            val_pairs.extend(matched_pairs)
        else:
            train_pairs.extend(matched_pairs)

    total = len(train_pairs) + len(val_pairs)

    manifest = {
        "status": "DISCOVERED",
        "dataset_root": str(root),
        "total_frames": total,
        "train_frames": len(train_pairs),
        "val_frames": len(val_pairs),
        "sequence_breakdown": seq_breakdown,
        "train_pairs": train_pairs,
        "val_pairs": val_pairs
    }

    # Save manifest file
    with open("data_manifest.json", "w") as f:
        json.dump({
            "dataset_root": str(root),
            "total_frames": total,
            "train": [p[0] for p in train_pairs],
            "val": [p[0] for p in val_pairs],
            "train_pairs": train_pairs,
            "val_pairs": val_pairs,
            "sequence_breakdown": seq_breakdown
        }, f, indent=2)

    return manifest


if __name__ == "__main__":
    print("=" * 80)
    print("  SEMANTICPOSS DATASET DISCOVERY & MANIFEST GENERATOR")
    print("=" * 80)
    manifest = build_full_manifest()
    print(f"Status:        {manifest['status']}")
    print(f"Dataset Root:  {manifest.get('dataset_root', 'None')}")
    print(f"Total Frames:  {manifest['total_frames']}")
    print(f"Train Frames:  {manifest['train_frames']}")
    print(f"Val Frames:    {manifest['val_frames']}")
    print("\nSequence Breakdown:")
    for seq, d in manifest["sequence_breakdown"].items():
        print(f"  Sequence {seq}: {d['matched']} matched pairs (Bins: {d['bins']}, Labels: {d['labels']})")
    print("=" * 80)
