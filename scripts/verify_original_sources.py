#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rebuttal_common import sha256_file


def load_manifest(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"Invalid source manifest line {line_number}")
        entries.append((parts[0].lower(), parts[1].strip().replace("\\", "/")))
    if not entries:
        raise ValueError("Original source manifest is empty.")
    return entries


def verify(repo_root: Path, manifest_path: Path) -> int:
    failures: list[str] = []
    for expected, relative in load_manifest(manifest_path):
        path = (repo_root / relative).resolve()
        if repo_root not in path.parents:
            failures.append(f"unsafe path: {relative}")
        elif not path.is_file():
            failures.append(f"missing: {relative}")
        elif sha256_file(path) != expected:
            failures.append(f"modified: {relative}")
    if failures:
        raise RuntimeError("Original v6 source verification failed: " + "; ".join(failures))
    return len(load_manifest(manifest_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify copied original-v6 core source hashes.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--manifest", default="ORIGINAL_V6_SHA256SUMS")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    count = verify(repo_root, repo_root / args.manifest)
    print(f"original_v6_source_files_verified={count}")


if __name__ == "__main__":
    main()
