#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Append one endpoint-free experiment attempt record.")
    parser.add_argument("--path", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--started-at", required=True)
    parser.add_argument("--finished-at", required=True)
    parser.add_argument("--duration-sec", required=True, type=float)
    parser.add_argument("--status", required=True, choices=["completed", "failed"])
    parser.add_argument("--exit-code", required=True, type=int)
    args = parser.parse_args()
    record = {
        "method": args.method,
        "benchmark": args.benchmark,
        "module": args.module,
        "config": args.config,
        "output": args.output,
        "started_at": args.started_at,
        "finished_at": args.finished_at,
        "duration_sec": args.duration_sec,
        "status": args.status,
        "exit_code": args.exit_code,
    }
    path = Path(args.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
