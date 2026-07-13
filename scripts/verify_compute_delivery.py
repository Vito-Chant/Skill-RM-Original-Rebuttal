#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any

from package_compute_comparison import ANALYSIS_ARCHIVE, METHODS, SUMMARY_ARCHIVE
from rebuttal_common import BENCHMARKS, assert_text_sanitized, sha256_bytes, sha256_file


def inspect_archive(path: Path, *, kind: str) -> dict[str, Any]:
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate members in {path.name}")
        for member in members:
            pure = Path(member.name)
            if pure.is_absolute() or ".." in pure.parts or not member.isfile():
                raise ValueError(f"Unsafe or non-file member: {member.name}")
        manifest_file = archive.extractfile("ARCHIVE_MANIFEST.json")
        if manifest_file is None:
            raise ValueError(f"Missing archive manifest in {path.name}")
        manifest = json.loads(manifest_file.read().decode("utf-8"))
        if manifest.get("archive_kind") != kind:
            raise ValueError(f"Wrong archive kind in {path.name}")
        declared = manifest.get("files") or {}
        if set(declared) != set(names) - {"ARCHIVE_MANIFEST.json"}:
            raise ValueError(f"Archive member/manifest mismatch in {path.name}")
        for name, metadata in declared.items():
            handle = archive.extractfile(name)
            if handle is None:
                raise ValueError(f"Could not read {name}")
            payload = handle.read()
            if sha256_bytes(payload) != metadata.get("staged_sha256"):
                raise ValueError(f"Hash mismatch for {name}")
            if len(payload) != metadata.get("staged_bytes"):
                raise ValueError(f"Byte-count mismatch for {name}")
            text = payload.decode("utf-8")
            assert_text_sanitized(name, text)
            if name.endswith(".jsonl"):
                rows = sum(1 for line in text.splitlines() if line.strip())
                if rows != metadata.get("jsonl_rows"):
                    raise ValueError(f"JSONL row-count mismatch for {name}")
        if kind == "compute-summary" and any(
            "predictions.jsonl" in name or "traces.jsonl" in name or "raw_output" in name for name in names
        ):
            raise ValueError("Compute summary archive contains per-sample outputs")
        if kind == "compute-analysis":
            for method in METHODS:
                for benchmark in BENCHMARKS:
                    prefix = f"analysis/runs/{method}/{benchmark}/"
                    for required in ("metrics.json", "predictions.jsonl"):
                        if prefix + required not in names:
                            raise ValueError(f"Compute analysis archive missing {prefix + required}")
                    trace_name = prefix + "traces.jsonl"
                    if method == "skill_rm" and trace_name not in names:
                        raise ValueError(f"Compute analysis archive missing {trace_name}")
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path), "members": len(names)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify matched-compute delivery archives.")
    parser.add_argument("--delivery-dir", required=True)
    args = parser.parse_args()
    delivery = Path(args.delivery_dir).resolve()
    expected = {SUMMARY_ARCHIVE: "compute-summary", ANALYSIS_ARCHIVE: "compute-analysis"}
    checksums = {
        parts[1]: parts[0]
        for line in (delivery / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        if len(parts := line.split()) == 2
    }
    results = []
    for name, kind in expected.items():
        result = inspect_archive(delivery / name, kind=kind)
        if checksums.get(name) != result["sha256"]:
            raise ValueError(f"Outer checksum mismatch for {name}")
        results.append(result)
    print(json.dumps({"verified": True, "archives": results}, indent=2))


if __name__ == "__main__":
    main()
