#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Iterable

from rebuttal_common import (
    BENCHMARKS,
    METHODS,
    assert_text_sanitized,
    count_nonempty_lines,
    sanitize_text,
    sanitize_value,
    sha256_file,
    write_json,
)


SUMMARY_ARCHIVE = "skillrm_original_v6_rebuttal_summary_qwen35_27b.tar.gz"
ANALYSIS_ARCHIVE = "skillrm_original_v6_rebuttal_analysis_qwen35_27b.tar.gz"
TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".log", ".txt", ".yaml", ".yml"}
FORBIDDEN_MEMBER_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", "server_logs", "server-logs"}


def sanitize_to_file(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(source)
    rows: int | None = None
    if source.suffix == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
        rendered = json.dumps(sanitize_value(value), indent=2, ensure_ascii=False) + "\n"
        assert_text_sanitized(source.name, rendered)
        destination.write_text(rendered, encoding="utf-8", newline="\n")
    elif source.suffix == ".jsonl":
        rows = 0
        with source.open("r", encoding="utf-8") as reader, destination.open("w", encoding="utf-8", newline="\n") as writer:
            for line_number, line in enumerate(reader, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {source}:{line_number}: {exc}") from exc
                rendered = json.dumps(sanitize_value(value), ensure_ascii=False)
                assert_text_sanitized(f"{source.name}:{line_number}", rendered)
                writer.write(rendered + "\n")
                rows += 1
    elif source.suffix.lower() in TEXT_SUFFIXES:
        with source.open("r", encoding="utf-8", errors="replace") as reader, destination.open(
            "w", encoding="utf-8", newline="\n"
        ) as writer:
            for line_number, line in enumerate(reader, start=1):
                rendered = sanitize_text(line)
                assert_text_sanitized(f"{source.name}:{line_number}", rendered)
                writer.write(rendered)
    else:
        raise ValueError(f"Refusing unsupported artifact type: {source}")
    metadata: dict[str, Any] = {
        "source_sha256": source_hash,
        "staged_sha256": sha256_file(destination),
        "staged_bytes": destination.stat().st_size,
    }
    if rows is not None:
        metadata["jsonl_rows"] = rows
    return metadata


def approved_run_files(run_dir: Path) -> list[Path]:
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    files = [path for path in run_dir.rglob("*") if path.is_file()]
    unsupported = [path for path in files if path.suffix.lower() not in TEXT_SUFFIXES]
    if unsupported:
        raise ValueError(f"Unsupported files in run directory: {[path.name for path in unsupported]}")
    return sorted(files)


def add_source(
    entries: list[tuple[Path, str]],
    source: Path,
    member: str,
    *,
    required: bool = True,
) -> None:
    if not source.is_file():
        if required:
            raise FileNotFoundError(source)
        return
    entries.append((source, member))


def summary_sources(repo_root: Path, run_root: Path) -> list[tuple[Path, str]]:
    entries: list[tuple[Path, str]] = []
    add_source(entries, run_root / "rebuttal_summary.json", "summary/rebuttal_summary.json")
    add_source(entries, run_root / "rebuttal_summary.md", "summary/rebuttal_summary.md")
    add_source(entries, run_root / "audit" / "audit_summary.json", "summary/audit_summary.json")
    add_source(entries, run_root / "audit" / "audit_summary.md", "summary/audit_summary.md")
    add_source(entries, run_root / "audit" / "qualitative_cases.json", "summary/qualitative_cases.json")
    add_source(
        entries,
        repo_root / "configs" / "rebuttal_original_v6" / "PREPARATION_REPORT.json",
        "summary/PREPARATION_REPORT.json",
    )
    for subset in ("subset_a", "subset_b"):
        add_source(
            entries,
            repo_root / "skills" / f"reward_judge_fair_50pct_{subset}" / "SUBSET_METADATA.json",
            f"summary/subsets/{subset}/SUBSET_METADATA.json",
        )
    for method in METHODS:
        for benchmark in BENCHMARKS:
            add_source(
                entries,
                run_root / method / benchmark / "metrics.json",
                f"summary/metrics/{method}/{benchmark}/metrics.json",
            )
    return entries


def analysis_sources(repo_root: Path, run_root: Path) -> list[tuple[Path, str]]:
    entries = summary_sources(repo_root, run_root)
    add_source(entries, run_root / "run_attempts.jsonl", "analysis/run_attempts.jsonl")
    for log in sorted((run_root / "logs").glob("*.log")):
        add_source(entries, log, f"analysis/logs/{log.name}")
    for method in METHODS:
        for benchmark in BENCHMARKS:
            run_dir = run_root / method / benchmark
            for source in approved_run_files(run_dir):
                relative = source.relative_to(run_dir).as_posix()
                add_source(entries, source, f"analysis/runs/{method}/{benchmark}/{relative}")
    audit_dir = run_root / "audit"
    for source in sorted(path for path in audit_dir.rglob("*") if path.is_file() and "stale" not in path.parts):
        if source.suffix.lower() not in TEXT_SUFFIXES:
            raise ValueError(f"Unsupported audit artifact: {source.name}")
        add_source(entries, source, f"analysis/audit/{source.relative_to(audit_dir).as_posix()}")
    return entries


def assert_member_safe(member: str) -> None:
    path = Path(member)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive member: {member}")
    lower_parts = {part.lower() for part in path.parts}
    if lower_parts.intersection(FORBIDDEN_MEMBER_PARTS):
        raise ValueError(f"Forbidden archive member: {member}")
    if any(part.lower() == "data" for part in path.parts):
        raise ValueError(f"Benchmark source data must not be archived: {member}")


def build_archive(
    output: Path,
    sources: list[tuple[Path, str]],
    *,
    archive_kind: str,
    staging_parent: Path,
) -> dict[str, Any]:
    seen: set[str] = set()
    with tempfile.TemporaryDirectory(prefix=f"skillrm-{archive_kind}-", dir=staging_parent) as temporary:
        stage = Path(temporary)
        file_manifest: dict[str, Any] = {}
        for source, member in sources:
            assert_member_safe(member)
            if member in seen:
                continue
            seen.add(member)
            destination = stage / member
            file_manifest[member] = sanitize_to_file(source, destination)
        manifest = {
            "archive_version": "skill-rm-original-v6-rebuttal-delivery-v1",
            "archive_kind": archive_kind,
            "sanitized": True,
            "preserves_jsonl_order_and_counts": True,
            "contains_benchmark_source_data": False,
            "contains_server_logs": False,
            "files": dict(sorted(file_manifest.items())),
        }
        manifest_path = stage / "ARCHIVE_MANIFEST.json"
        write_json(manifest_path, manifest)
        assert_text_sanitized("ARCHIVE_MANIFEST.json", manifest_path.read_text(encoding="utf-8"))
        output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output, "w:gz", format=tarfile.PAX_FORMAT) as archive:
            for path in sorted(item for item in stage.rglob("*") if item.is_file()):
                member = path.relative_to(stage).as_posix()
                info = archive.gettarinfo(str(path), arcname=member)
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                info.mode = 0o600
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
    return {
        "name": output.name,
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "members": len(seen) + 1,
    }


def validate_prerequisites(run_root: Path) -> None:
    summary = json.loads((run_root / "rebuttal_summary.json").read_text(encoding="utf-8"))
    if not summary.get("all_runs_complete"):
        raise ValueError("Cannot package incomplete runs.")
    audit_dir = run_root / "audit"
    required = [
        audit_dir / "audit_results.jsonl",
        audit_dir / "audit_results.lock.json",
        audit_dir / "audit_summary.json",
        audit_dir / "qualitative_cases.json",
    ]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"Cannot package before audit completion: {missing}")
    lock = json.loads((audit_dir / "audit_results.lock.json").read_text(encoding="utf-8"))
    if sha256_file(audit_dir / "audit_input.jsonl") != lock.get("input_sha256"):
        raise ValueError("Audit input does not match the locked results.")
    if sha256_file(audit_dir / "audit_results.jsonl") != lock.get("results_sha256"):
        raise ValueError("Audit results changed after locking.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the two sanitized private rebuttal delivery archives.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--delivery-dir")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    run_root = Path(args.run_root).resolve()
    validate_prerequisites(run_root)
    delivery = Path(args.delivery_dir).resolve() if args.delivery_dir else run_root / "delivery"
    delivery.mkdir(parents=True, exist_ok=True)

    summary_result = build_archive(
        delivery / SUMMARY_ARCHIVE,
        summary_sources(repo_root, run_root),
        archive_kind="summary",
        staging_parent=delivery,
    )
    analysis_result = build_archive(
        delivery / ANALYSIS_ARCHIVE,
        analysis_sources(repo_root, run_root),
        archive_kind="analysis",
        staging_parent=delivery,
    )
    results = [summary_result, analysis_result]
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
