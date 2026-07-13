#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from package_original_v6_rebuttal import add_source, approved_run_files, build_archive
from rebuttal_common import BENCHMARKS, sha256_file, write_json


METHODS = ("direct", "skill_rm")
SUMMARY_ARCHIVE = "skillrm_compute_summary_qwen35_27b.tar.gz"
ANALYSIS_ARCHIVE = "skillrm_compute_full_analysis_qwen35_27b.tar.gz"


def validate_prerequisites(run_root: Path) -> dict[str, Any]:
    summary_path = run_root / "compute_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("all_runs_complete"):
        raise ValueError("Cannot package incomplete compute runs or incomplete usage.")
    for method in METHODS:
        for benchmark in BENCHMARKS:
            run_dir = run_root / method / benchmark
            for required in ("metrics.json", "predictions.jsonl"):
                if not (run_dir / required).is_file():
                    raise FileNotFoundError(run_dir / required)
            if method == "skill_rm" and not (run_dir / "traces.jsonl").is_file():
                raise FileNotFoundError(run_dir / "traces.jsonl")
    return summary


def summary_sources(run_root: Path) -> list[tuple[Path, str]]:
    entries: list[tuple[Path, str]] = []
    add_source(entries, run_root / "compute_summary.json", "summary/compute_summary.json")
    add_source(entries, run_root / "compute_summary.md", "summary/compute_summary.md")
    add_source(entries, run_root / "run_attempts.jsonl", "summary/run_attempts.jsonl")
    for method in METHODS:
        for benchmark in BENCHMARKS:
            prefix = f"summary/runs/{method}/{benchmark}"
            run_dir = run_root / method / benchmark
            add_source(entries, run_dir / "metrics.json", f"{prefix}/metrics.json")
            add_source(entries, run_dir / "config_resolved.json", f"{prefix}/config_resolved.json", required=False)
    return entries


def analysis_sources(run_root: Path) -> list[tuple[Path, str]]:
    entries = summary_sources(run_root)
    for log in sorted((run_root / "logs").glob("*.log")):
        add_source(entries, log, f"analysis/logs/{log.name}")
    for method in METHODS:
        for benchmark in BENCHMARKS:
            run_dir = run_root / method / benchmark
            for source in approved_run_files(run_dir):
                relative = source.relative_to(run_dir).as_posix()
                add_source(entries, source, f"analysis/runs/{method}/{benchmark}/{relative}")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sanitized matched-compute delivery archives.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--delivery-dir")
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    validate_prerequisites(run_root)
    delivery = Path(args.delivery_dir).resolve() if args.delivery_dir else run_root / "delivery"
    delivery.mkdir(parents=True, exist_ok=True)

    summary = build_archive(
        delivery / SUMMARY_ARCHIVE,
        summary_sources(run_root),
        archive_kind="compute-summary",
        staging_parent=delivery,
    )
    analysis = build_archive(
        delivery / ANALYSIS_ARCHIVE,
        analysis_sources(run_root),
        archive_kind="compute-analysis",
        staging_parent=delivery,
    )
    results = [summary, analysis]
    (delivery / "SHA256SUMS").write_text(
        "".join(f"{item['sha256']}  {item['name']}\n" for item in results),
        encoding="utf-8",
        newline="\n",
    )
    write_json(delivery / "delivery_manifest.json", {"archives": results, "send_privately": True})
    for item in results:
        print(f"archive={delivery / item['name']}")
        print(f"bytes={item['bytes']}")
        print(f"sha256={item['sha256']}")


if __name__ == "__main__":
    main()
