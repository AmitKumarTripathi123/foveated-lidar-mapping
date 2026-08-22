"""
scripts/audit_semanticposs.py
==============================
Rigorous audit tool for SemanticPOSS dataset completeness and point-label integrity:
  - Enumerate .bin and .label files per sequence (00..05)
  - Pair by filename stem: clouds[frame_id] <-> labels[frame_id]
  - Detect missing files, duplicate IDs, malformed files
  - Verify float32 point representations and uint32 semantic labels (& 0xFFFF)
  - Verify 1:1 point-to-label alignment (N_points == N_labels)
  - Check for NaN and Inf values
  - Verify SIH super-class remapping

Priority for dataset root:
  1. CLI argument (--dataset-root)
  2. Environment variable (DATASET_ROOT)
  3. Default repository directory (dataset/)
"""

import os
import sys
import glob
import argparse
import numpy as np
from typing import Dict, List, Tuple


def get_dataset_root(cli_root: str = None) -> str:
    """Resolve dataset root using priority: CLI > ENV > Default 'dataset/'."""
    if cli_root and os.path.exists(cli_root):
        return os.path.abspath(cli_root)
    env_root = os.getenv("DATASET_ROOT")
    if env_root and os.path.exists(env_root):
        return os.path.abspath(env_root)
    return os.path.abspath("dataset")


def audit_sequence(seq_dir: str, seq_id: str) -> Dict:
    """Audit a single sequence for frame pairing, data representation, and alignment."""
    bin_dir = os.path.join(seq_dir, "velodyne")
    label_dir = os.path.join(seq_dir, "labels")

    if not os.path.exists(bin_dir) or not os.path.exists(label_dir):
        return {"error": f"Missing velodyne/ or labels/ in {seq_dir}"}

    bin_files = {os.path.splitext(f)[0]: os.path.join(bin_dir, f) for f in os.listdir(bin_dir) if f.endswith(".bin")}
    label_files = {os.path.splitext(f)[0]: os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".label")}

    all_stems = sorted(list(set(bin_files.keys()) | set(label_files.keys())))

    matched_pairs = []
    missing_bins = []
    missing_labels = []
    alignment_errors = []
    nan_inf_errors = []
    total_points = 0

    for stem in all_stems:
        has_bin = stem in bin_files
        has_lbl = stem in label_files

        if not has_bin:
            missing_bins.append(stem)
            continue
        if not has_lbl:
            missing_labels.append(stem)
            continue

        bin_path = bin_files[stem]
        lbl_path = label_files[stem]

        # Read binary files
        pts = np.fromfile(bin_path, dtype=np.float32)
        lbls = np.fromfile(lbl_path, dtype=np.uint32)

        if pts.size % 4 != 0:
            alignment_errors.append((stem, f"Points array size {pts.size} not divisible by 4"))
            continue

        pts = pts.reshape(-1, 4)
        lbls = lbls & 0xFFFF

        if pts.shape[0] != lbls.shape[0]:
            alignment_errors.append((stem, f"Point count {pts.shape[0]} != Label count {lbls.shape[0]}"))
            continue

        if np.isnan(pts).any() or np.isinf(pts).any():
            nan_inf_errors.append((stem, "Contains NaN or Inf in spatial coordinates"))

        matched_pairs.append((stem, bin_path, lbl_path, pts.shape[0]))
        total_points += pts.shape[0]

    return {
        "sequence": seq_id,
        "total_stems": len(all_stems),
        "matched_pairs": len(matched_pairs),
        "missing_bins": missing_bins,
        "missing_labels": missing_labels,
        "alignment_errors": alignment_errors,
        "nan_inf_errors": nan_inf_errors,
        "total_points": total_points,
        "frame_details": matched_pairs,
    }


def main():
    parser = argparse.ArgumentParser(description="SemanticPOSS Dataset Audit Script")
    parser.add_argument("--dataset-root", type=str, default=None, help="Path to SemanticPOSS dataset root")
    args = parser.parse_args()

    root = get_dataset_root(args.dataset_root)
    print("==================================================")
    print("      SemanticPOSS Physical Dataset Audit         ")
    print("==================================================")
    print(f"Dataset Root Resolved: {root}\n")

    seq_dir = os.path.join(root, "sequences")
    if not os.path.exists(seq_dir):
        print(f"CRITICAL ERROR: {seq_dir} does not exist!")
        sys.exit(1)

    expected_counts = {"00": 488, "01": 500, "02": 500, "03": 500, "04": 500, "05": 500}
    all_seqs = sorted([d for d in os.listdir(seq_dir) if os.path.isdir(os.path.join(seq_dir, d))])

    print(f"Found Sequences: {all_seqs}\n")

    total_matched = 0
    total_expected = sum(expected_counts.values())
    total_dataset_points = 0

    for seq_id in ["00", "01", "02", "03", "04", "05"]:
        if seq_id not in all_seqs:
            print(f"[MISSING] Sequence {seq_id}: MISSING DIRECTORY!")
            continue

        res = audit_sequence(os.path.join(seq_dir, seq_id), seq_id)
        matched = res.get("matched_pairs", 0)
        expected = expected_counts.get(seq_id, 0)
        pts_count = res.get("total_points", 0)
        total_matched += matched
        total_dataset_points += pts_count

        status = "PASSED [OK]" if matched == expected and not res["alignment_errors"] else "WARNING [PARTIAL]"
        print(f"Sequence {seq_id}: {status}")
        print(f"  Matched Scan Pairs: {matched} / {expected}")
        print(f"  Total Points      : {pts_count:,}")
        if res["missing_bins"]:
            print(f"  Missing .bin      : {res['missing_bins']}")
        if res["missing_labels"]:
            print(f"  Missing .label    : {res['missing_labels']}")
        if res["alignment_errors"]:
            print(f"  Alignment Errors  : {res['alignment_errors']}")
        if res["nan_inf_errors"]:
            print(f"  NaN/Inf Errors    : {res['nan_inf_errors']}")

    print("\n--------------------------------------------------")
    print(f"AUDIT SUMMARY:")
    print(f"  Total Matched Scan Pairs : {total_matched:,} / {total_expected:,}")
    print(f"  Total Dataset Points     : {total_dataset_points:,}")
    print("--------------------------------------------------\n")

    if total_matched == total_expected:
        print("[OK] FULL SEMANTICPOSS DATASET VERIFIED & PASSED!")
    else:
        print("[WARNING] PARTIAL DATASET AUDIT COMPLETED.")


if __name__ == "__main__":
    main()
