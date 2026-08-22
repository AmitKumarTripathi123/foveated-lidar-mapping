#!/usr/bin/env python3
"""scripts/audit_semanticposs.py

Comprehensive forensic diagnostic and audit tool for SemanticPOSS datasets.
Performs recursive discovery, sequence validation, stem-based frame pairing,
file integrity checks, and point-label alignment verification.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


def audit_dataset(
    root_path: Path,
    expected_sequences: Optional[List[str]] = None,
    expected_frames: int = 2988,
) -> Dict[str, any]:
    """Perform a deep forensic audit of the specified dataset directory."""
    if expected_sequences is None:
        expected_sequences = ["00", "01", "02", "03", "04", "05"]

    root = Path(root_path).resolve()
    results = {
        "root": str(root),
        "root_exists": root.exists(),
        "expected_sequences": expected_sequences,
        "expected_frames": expected_frames,
        "sequences_found": [],
        "sequence_details": {},
        "total_clouds": 0,
        "total_labels": 0,
        "total_matched": 0,
        "total_missing_labels": 0,
        "total_missing_clouds": 0,
        "empty_files": [],
        "malformed_files": [],
        "point_label_mismatches": [],
        "status": "FAIL",
    }

    if not root.exists():
        return results

    # Check potential sequence base directories
    seq_root = root / "sequences"
    base_dir = seq_root if seq_root.exists() else root

    # Discover sequence directories
    discovered_seq_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    discovered_seq_names = [d.name for d in discovered_seq_dirs]
    results["sequences_found"] = discovered_seq_names

    for seq in expected_sequences:
        seq_dir = base_dir / seq
        seq_info = {
            "exists": seq_dir.exists(),
            "cloud_count": 0,
            "label_count": 0,
            "matched_count": 0,
            "missing_labels": 0,
            "missing_clouds": 0,
            "point_stats": {},
        }

        if seq_dir.exists():
            # Scan velodyne / scans / root
            velo_dir = seq_dir / "velodyne"
            if not velo_dir.exists():
                velo_dir = seq_dir / "scans"
            if not velo_dir.exists():
                velo_dir = seq_dir

            # Scan labels / root
            lbl_dir = seq_dir / "labels"
            if not lbl_dir.exists():
                lbl_dir = seq_dir

            # Discover bin files
            cloud_files = {}
            for ext in ("*.bin", "*.BIN"):
                for p in velo_dir.glob(ext):
                    cloud_files[p.stem] = p

            # Discover label files
            label_files = {}
            for ext in ("*.label", "*.LABEL"):
                for p in lbl_dir.glob(ext):
                    label_files[p.stem] = p

            matched_stems = sorted(set(cloud_files.keys()) & set(label_files.keys()))
            unmatched_clouds = sorted(set(cloud_files.keys()) - set(label_files.keys()))
            unmatched_labels = sorted(set(label_files.keys()) - set(cloud_files.keys()))

            seq_info["cloud_count"] = len(cloud_files)
            seq_info["label_count"] = len(label_files)
            seq_info["matched_count"] = len(matched_stems)
            seq_info["missing_labels"] = len(unmatched_clouds)
            seq_info["missing_clouds"] = len(unmatched_labels)

            results["total_clouds"] += len(cloud_files)
            results["total_labels"] += len(label_files)
            results["total_matched"] += len(matched_stems)
            results["total_missing_labels"] += len(unmatched_clouds)
            results["total_missing_clouds"] += len(unmatched_labels)

            # Sample audit on matched frames for file integrity
            for stem in matched_stems[:10]:
                c_path = cloud_files[stem]
                l_path = label_files[stem]

                # Check empty
                if c_path.stat().st_size == 0:
                    results["empty_files"].append(str(c_path))
                    continue
                if l_path.stat().st_size == 0:
                    results["empty_files"].append(str(l_path))
                    continue

                # Check malformed
                if c_path.stat().st_size % 16 != 0:  # 4 * 4 bytes
                    results["malformed_files"].append(str(c_path))
                    continue
                if l_path.stat().st_size % 4 != 0:   # 4 bytes per label
                    results["malformed_files"].append(str(l_path))
                    continue

                # Point-label count equality
                n_pts = c_path.stat().st_size // 16
                n_lbls = l_path.stat().st_size // 4
                if n_pts != n_lbls:
                    results["point_label_mismatches"].append({
                        "sequence": seq,
                        "frame": stem,
                        "points": n_pts,
                        "labels": n_lbls,
                    })

        results["sequence_details"][seq] = seq_info

    # Determine overall audit status
    all_seqs_present = all(seq in results["sequences_found"] for seq in expected_sequences)
    frame_count_matched = (results["total_matched"] == expected_frames)
    zero_errors = (
        len(results["empty_files"]) == 0
        and len(results["malformed_files"]) == 0
        and len(results["point_label_mismatches"]) == 0
    )

    if all_seqs_present and frame_count_matched and zero_errors:
        results["status"] = "PASS"
    else:
        results["status"] = "FAIL"

    return results


def print_audit_report(res: Dict[str, any]) -> None:
    """Print standard formatted audit report to stdout."""
    print("SemanticPOSS DATASET AUDIT")
    print("==========================")
    print(f"\nROOT:\n{res['root']}")
    print("\nSEQUENCES FOUND:")
    if res["sequences_found"]:
        for s in res["sequences_found"]:
            print(f"  {s}")
    else:
        print("  (None found)")

    print("\nSEQUENCE BREAKDOWN")
    print("------------------")
    for seq, detail in res["sequence_details"].items():
        print(f"SEQUENCE {seq}")
        print(f"  Exists:         {'YES' if detail['exists'] else 'NO'}")
        print(f"  Point clouds:   {detail['cloud_count']}")
        print(f"  Labels:         {detail['label_count']}")
        print(f"  Matched:        {detail['matched_count']}")
        print(f"  Missing labels: {detail['missing_labels']}")
        print(f"  Missing clouds: {detail['missing_clouds']}")
        print()

    print("TOTAL")
    print("-----")
    print(f"Point clouds:   {res['total_clouds']}")
    print(f"Labels:         {res['total_labels']}")
    print(f"Matched pairs:  {res['total_matched']}")
    print()
    print(f"Expected sequences: {len(res['expected_sequences'])} ({', '.join(res['expected_sequences'])})")
    print(f"Detected sequences: {len(res['sequences_found'])} ({', '.join(res['sequences_found'])})")
    print()
    print(f"Expected frames: {res['expected_frames']}")
    print(f"Detected frames: {res['total_matched']}")
    print()
    print(f"STATUS:\n{res['status']}")
    print("==========================\n")


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Audit SemanticPOSS Dataset.")
    parser.add_argument(
        "--root",
        type=str,
        default=os.environ.get("DATASET_ROOT", "dataset"),
        help="Path to SemanticPOSS dataset root directory.",
    )
    parser.add_argument(
        "--expected-sequences",
        nargs="+",
        default=["00", "01", "02", "03", "04", "05"],
        help="List of expected sequence IDs.",
    )
    parser.add_argument(
        "--expected-frames",
        type=int,
        default=2988,
        help="Expected total number of matched frames.",
    )

    args = parser.parse_args()
    results = audit_dataset(
        root_path=Path(args.root),
        expected_sequences=args.expected_sequences,
        expected_frames=args.expected_frames,
    )
    print_audit_report(results)

    return 0 if results["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
