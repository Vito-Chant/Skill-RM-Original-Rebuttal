#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path
from typing import Any

from package_original_v6_rebuttal import ANALYSIS_ARCHIVE, SUMMARY_ARCHIVE
from rebuttal_common import assert_text_sanitized, sha256_bytes, sha256_file


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
        if "ARCHIVE_MANIFEST.json" not in names:
            raise ValueError(f"Missing archive manifest in {path.name}")
        manifest_file = archive.extractfile("ARCHIVE_MANIFEST.json")
        if manifest_file is None:
            raise ValueError("Could not read archive manifest")
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
            if name.endswith(".jsonl") and sum(1 for line in text.splitlines() if line.strip()) != metadata.get("jsonl_rows"):
                raise ValueError(f"JSONL row-count mismatch for {name}")
        if kind == "summary" and any("predictions.jsonl" in name or "traces.jsonl" in name for name in names):
            raise ValueError("Summary archive contains raw predictions or traces")
        if kind == "analysis":
            for method in ("full_fair", "subset_a", "subset_b"):
                for benchmark in ("rewardbench2", "rmbench", "judgebench"):
                    prefix = f"analysis/runs/{method}/{benchmark}/"
                    for required in ("metrics.json", "predictions.jsonl", "traces.jsonl"):
                        if prefix + required not in names:
                            raise ValueError(f"Analysis archive missing {prefix + required}")
            for required in (
                "analysis/audit/audit_input.jsonl",
                "analysis/audit/audit_key.jsonl",
                "analysis/audit/audit_results.jsonl",
                "analysis/audit/audit_summary.json",
            ):
                if required not in names:
                    raise ValueError(f"Analysis archive missing {required}")
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path), "members": len(names)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify both private rebuttal delivery archives.")
    parser.add_argument("--delivery-dir", required=True)
    args = parser.parse_args()
    delivery = Path(args.delivery_dir).resolve()
    expected = {
        SUMMARY_ARCHIVE: "summary",
        ANALYSIS_ARCHIVE: "analysis",
    }
    checksum_path = delivery / "SHA256SUMS"
    checksums = {
        parts[1]: parts[0]
        for line in checksum_path.read_text(encoding="utf-8").splitlines()
        if len(parts := line.split()) == 2
    }
    results = []
    for name, kind in expected.items():
        path = delivery / name
        result = inspect_archive(path, kind=kind)
        if checksums.get(name) != result["sha256"]:
            raise ValueError(f"Outer checksum mismatch for {name}")
        results.append(result)
    print(json.dumps({"verified": True, "archives": results}, indent=2))


if __name__ == "__main__":
    main()
