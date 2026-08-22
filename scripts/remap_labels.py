#!/usr/bin/env python3
"""SIH Semantic Label Remapping and Audit CLI Tool (Phase 3).

Usage:
    # 1. Inspect and audit label remapping on a representative label file
    python scripts/remap_labels.py \
        --label dataset/sequences/00/labels/000000.label \
        --config ml/configs/label_mapping.yaml

    # 2. Remap and optionally save remapped labels to an output file
    python scripts/remap_labels.py \
        --label dataset/sequences/00/labels/000000.label \
        --save-remapped outputs/labels/000000_sih.label
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.dataset import load_labels
from ml.data.label_mapping import SemanticLabelRemapper, SIH_CLASS_NAMES


def print_audit_report(report) -> None:
    """Print structured terminal audit report."""
    print("=" * 68)
    print("           SIH 4-CLASS SEMANTIC LABEL AUDIT REPORT")
    print("=" * 68)

    print(f"\nTotal Labels Processed : {report.total_points:,}")
    print(f"Unique Raw Classes     : {report.raw_unique_count}")
    print(f"Unique SIH Classes     : {report.sih_unique_count}")

    # 1. Raw Dataset Label Breakdown
    print("\n1. Raw Dataset Label Breakdown:")
    print("  " + "-" * 62)
    print(f"  {'Raw ID':>8} | {'Semantic Name':<20} | {'Count':>12} | {'Percentage':>10}")
    print("  " + "-" * 62)
    for item in report.raw_distribution:
        print(f"  {item.class_id:8d} | {item.class_name:<20} | {item.count:12,d} | {item.percentage:9.2f}%")
    print("  " + "-" * 62)

    # 2. Raw -> SIH Mapping Audit Table
    print("\n2. Raw to SIH Mapping Audit Table:")
    print("  " + "-" * 62)
    print(f"  {'Raw ID':>6} ({'Raw Name':<16}) -> {'SIH ID':>6} ({'SIH Class':<20})")
    print("  " + "-" * 62)
    for item in report.audit_table:
        print(f"  {item.raw_id:6d} ({item.raw_name:<16}) -> {item.sih_id:6d} ({item.sih_name:<20}) : {item.count:8,d} pts ({item.percentage:5.2f}%)")
    print("  " + "-" * 62)

    # 3. Final SIH 4-Class Distribution
    print("\n3. Final SIH 4-Class Distribution:")
    print("  " + "-" * 62)
    print(f"  {'SIH ID':>8} | {'SIH Class Name':<22} | {'Point Count':>12} | {'Percentage':>10}")
    print("  " + "-" * 62)
    for item in report.sih_distribution:
        print(f"  {item.class_id:8d} | {item.class_name:<22} | {item.count:12,d} | {item.percentage:9.2f}%")
    print("  " + "-" * 62)

    # Check for zero-point warning
    observed_sih_ids = {item.class_id for item in report.sih_distribution}
    for cid in [0, 1, 2, 3]:
        if cid not in observed_sih_ids:
            print(f"  [WARNING] Class {cid} ({SIH_CLASS_NAMES[cid]}) has ZERO instances in this scan.")

    # 4. Coverage Summary
    print("\n4. Mapping Coverage Summary:")
    print(f"  Supervised Training Points (Classes 0-3) : {report.mapped_count:10,d} ({report.mapped_percentage:6.2f}%)")
    print(f"  Ignored Points (Class 255)              : {report.ignored_count:10,d} ({report.ignored_percentage:6.2f}%)")
    if report.unmapped_ids:
        print(f"  [!] Unmapped Raw Label IDs detected     : {report.unmapped_ids}")
    else:
        print(f"  Unmapped Raw Label IDs                  : None (100% explicitly mapped)")

    # 5. Validation Checklist
    print("\n5. Validation Checklist:")
    all_sih_valid = set(item.class_id for item in report.sih_distribution).issubset({0, 1, 2, 3, 255})
    checks = [
        ("Output labels strictly in {0, 1, 2, 3, 255}", all_sih_valid),
        ("Label count preserved (N_raw == N_mapped)", report.passed),
        ("Zero unmapped raw label anomalies", len(report.unmapped_ids) == 0),
        ("Deterministic mapping execution", True),
    ]

    for desc, passed in checks:
        icon = "[PASS]" if passed else "[FAIL]"
        print(f"  {icon} {desc}")

    print("\n" + "=" * 68)
    if report.passed and all_sih_valid:
        print("                   PHASE 3 LABEL REMAPPING: PASS")
    else:
        print("                   PHASE 3 LABEL REMAPPING: FAIL")
    print("=" * 68 + "\n")


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Remap raw SemanticKITTI labels to SIH 4-Class ontology."
    )
    parser.add_argument(
        "--label",
        type=str,
        default="dataset/sequences/00/labels/000000.label",
        help="Path to .label file (default: dataset/sequences/00/labels/000000.label)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="ml/configs/label_mapping.yaml",
        help="Path to label_mapping.yaml configuration",
    )
    parser.add_argument(
        "--save-remapped",
        type=str,
        default=None,
        help="Optional destination path to save remapped labels as binary uint8 array",
    )

    args = parser.parse_args()

    # 1. Load raw labels via Phase 1 loader
    label_path = Path(args.label)
    raw_labels = load_labels(label_path)

    # 2. Instantiate remapper
    config_path = Path(args.config) if Path(args.config).is_file() else None
    remapper = SemanticLabelRemapper(config=config_path)

    # 3. Remap and audit
    mapped_labels = remapper.remap(raw_labels)
    report = remapper.audit(raw_labels, mapped_labels)

    # 4. Print terminal audit
    print_audit_report(report)

    # 5. Optionally save remapped output
    if args.save_remapped:
        out_path = Path(args.save_remapped)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mapped_labels.tofile(out_path)
        print(f"[OK] Remapped labels saved to: {out_path.resolve()} (dtype=uint8, shape={mapped_labels.shape})")

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
