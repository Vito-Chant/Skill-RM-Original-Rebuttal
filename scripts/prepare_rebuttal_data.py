#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from rebuttal_common import sha256_file


OPENRS_COMMIT = "4c7f22b570753630e91d8450205d33f1580a00df"
FILES = {
    "rewardbench_v2/rewardbench_v2.jsonl": {
        "rows": 1865,
        "sha256": "032e9a51f0782ce01179e11201b4ef9667948af3601746a00fd999760f5007c8",
    },
    "judgebench/gpt.jsonl": {
        "rows": 350,
        "sha256": "0585f737413c90fb7e3e2a756ead4c0d238dbdc4d7339ce0a0a8b5dab3cbdfec",
    },
    "judgebench/claude.jsonl": {
        "rows": 270,
        "sha256": "0960f3754780a2f14e0f2dc43c1638af54a1c2fef268904baf8bbb8ef4d8fb33",
    },
    "rmbench/rmbench.jsonl": {
        "rows": 1327,
        "sha256": "15607b1895106754cef98e52637135712074b02435e012a984bf911dc5b2a139",
    },
}


def validate_jsonl(path: Path, *, expected_rows: int, expected_sha256: str) -> None:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL in {path.name}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"Expected a JSON object in {path.name}:{line_number}")
            count += 1
    if count != expected_rows:
        raise RuntimeError(f"Unexpected row count for {path.name}: expected {expected_rows}, got {count}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_sha256:
        raise RuntimeError(f"SHA256 mismatch for {path.name}: expected {expected_sha256}, got {actual_hash}")


def prepare(root: Path, *, validate_only: bool = False) -> None:
    for relative, spec in FILES.items():
        target = root / relative
        if not target.is_file() or target.stat().st_size == 0:
            if validate_only:
                raise FileNotFoundError(f"Missing pinned OpenRS file: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".part")
            url = f"https://raw.githubusercontent.com/Qwen-Applications/OpenRS/{OPENRS_COMMIT}/data/{relative}"
            print(f"Downloading pinned OpenRS data: {relative}")
            urllib.request.urlretrieve(url, temporary)
            validate_jsonl(temporary, expected_rows=spec["rows"], expected_sha256=spec["sha256"])
            temporary.replace(target)
        validate_jsonl(target, expected_rows=spec["rows"], expected_sha256=spec["sha256"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the pinned OpenRS data used by the rebuttal runs.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.output).expanduser().resolve()
    prepare(root, validate_only=args.validate_only)
    print("Pinned OpenRS data validated.")


if __name__ == "__main__":
    main()
