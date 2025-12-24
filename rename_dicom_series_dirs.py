#!/usr/bin/env python3

import argparse
import os
import re
import sys
from typing import Dict, List, Tuple

try:
    import pydicom
    from pydicom.errors import InvalidDicomError
except Exception:
    print("ERROR: pydicom is required. Install with: pip install pydicom", file=sys.stderr)
    raise


def find_series_dirs(root: str) -> List[str]:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Not a directory: {root}")
    return sorted([os.path.join(root, name) for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))])


def find_first_dicom_file(directory: str) -> str:
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isfile(path) and (name.lower().endswith('.dcm') or '.' not in name):
            return path
    return ""


def sanitize_name(name: str) -> str:
    # Replace separators with underscore, collapse repeats, trim
    name = name.strip().replace(' ', '_').replace('/', '-').replace('\\', '-')
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name[:80] if len(name) > 80 else name


def propose_names(root: str) -> List[Tuple[str, str]]:
    proposals: List[Tuple[str, str]] = []
    existing: set = set()
    used_targets: Dict[str, int] = {}

    for series_dir in find_series_dirs(root):
        first = find_first_dicom_file(series_dir)
        if not first:
            continue
        try:
            ds = pydicom.dcmread(first, stop_before_pixels=True, force=True)
        except (InvalidDicomError, Exception):
            continue

        base_target = None
        # Prefer ProtocolName if available; otherwise fallback to SeriesDescription
        proto = getattr(ds, 'ProtocolName', None)
        if isinstance(proto, str) and proto.strip():
            base_target = proto
        else:
            series_desc = getattr(ds, 'SeriesDescription', None)
            if isinstance(series_desc, str) and series_desc.strip():
                base_target = series_desc
        if not base_target:
            # As a last resort, use the current directory name
            base_target = os.path.basename(series_dir)

        target = sanitize_name(base_target)
        if not target:
            target = os.path.basename(series_dir)

        # Ensure uniqueness within the root by appending counters if needed
        key = target.lower()
        if key in used_targets or target in existing:
            used_targets[key] = used_targets.get(key, 1) + 1
            target = f"{target}_{used_targets[key]}"
        else:
            used_targets[key] = 1

        existing.add(target)
        proposals.append((series_dir, os.path.join(root, target)))

    return proposals


def apply_moves(moves: List[Tuple[str, str]]) -> None:
    for src, dst in moves:
        if os.path.abspath(src) == os.path.abspath(dst):
            continue
        # If destination exists, add numeric suffix
        if os.path.exists(dst):
            base = dst
            i = 2
            while os.path.exists(f"{base}_{i}"):
                i += 1
            dst = f"{base}_{i}"
        os.rename(src, dst)
        print(f"RENAMED: {src} -> {dst}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rename DICOM series directories based on ProtocolName or SeriesDescription.")
    parser.add_argument("root", help="Root path containing series subdirectories (e.g., .../DICOM/AF2447CF)")
    parser.add_argument("--apply", action="store_true", help="Perform the renames. Without this, only print the proposed mapping.")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    proposals = propose_names(root)

    if not proposals:
        print("No series directories with DICOM files found.")
        return 0

    print("Proposed renames:")
    for src, dst in proposals:
        print(f"{src} -> {dst}")

    if args.apply:
        apply_moves(proposals)
        print("Done.")
    else:
        print("Dry-run complete. Re-run with --apply to perform the renames.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())






