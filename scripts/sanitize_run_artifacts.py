#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rebuttal_common import assert_text_sanitized, sanitize_text, sanitize_value


TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".log", ".txt", ".yaml", ".yml"}


def sanitize_file(path: Path) -> None:
    if path.suffix == ".json":
        value: Any = json.loads(path.read_text(encoding="utf-8"))
        rendered = json.dumps(sanitize_value(value), indent=2, ensure_ascii=False) + "\n"
    elif path.suffix == ".jsonl":
        lines: list[str] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            lines.append(json.dumps(sanitize_value(value), ensure_ascii=False))
        rendered = "\n".join(lines) + ("\n" if lines else "")
    else:
        rendered = sanitize_text(path.read_text(encoding="utf-8", errors="replace"))
    assert_text_sanitized(path.name, rendered)
    path.write_text(rendered, encoding="utf-8", newline="\n")


def sanitize_tree(root: Path) -> int:
    if not root.is_dir():
        raise FileNotFoundError(root)
    count = 0
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            sanitize_file(path)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove network, path, and credential metadata from a run tree.")
    parser.add_argument("path")
    args = parser.parse_args()
    root = Path(args.path).resolve()
    print(f"sanitized_files={sanitize_tree(root)}")


if __name__ == "__main__":
    main()
