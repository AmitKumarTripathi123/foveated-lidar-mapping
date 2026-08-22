#!/usr/bin/env python3
"""Dataset Manifest and Integrity Audit Generation CLI Tool (Master Task).

Usage:
    python scripts/generate_manifest.py --dataset-root dataset
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.manifest import discover_dataset, audit_dataset


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Generate dataset manifest and integrity audit.")
    parser.add_argument("--dataset-root", type=str, default="dataset", help="Root dataset directory")
    parser.add_argument("--manifest-out", type=str, default="data_manifest.json", help="Path to data_manifest.json")
    parser.add_argument("--audit-json", type=str, default="dataset_audit.json", help="Path to dataset_audit.json")
    parser.add_argument("--audit-md", type=str, default="dataset_audit.md", help="Path to dataset_audit.md")

    args = parser.parse_args()

    # 1. Discover all scans
    manifest = discover_dataset(args.dataset_root)
    with open(args.manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[OK] Generated manifest: {args.manifest_out}")

    # 2. Audit dataset
    audit = audit_dataset(manifest, args.audit_json, args.audit_md)
    print(f"[OK] Generated audit JSON: {args.audit_json}")
    print(f"[OK] Generated audit Markdown: {args.audit_md}")

    print("\n" + "=" * 68)
    print("                     DATASET MANIFEST & AUDIT REPORT")
    print("=" * 68)
    print(f"Total Discovered Scans : {audit['total_unique_frames']}")
    print(f"Total Points Audited   : {audit['total_points']:,}")
    print(f"Sequences Available    : {audit['discovered_sequences']}")
    print(f"Train Frame Entries    : {audit['train_frames_count']}")
    print(f"Val Frame Entries      : {audit['val_frames_count']}")
    print(f"Corrupted / Failed     : {audit['corrupted_count']}")
    print("=" * 68 + "\n")

    return 0 if audit["corrupted_count"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
