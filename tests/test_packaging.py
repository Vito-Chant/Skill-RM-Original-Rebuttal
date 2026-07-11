from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from package_original_v6_rebuttal import build_archive
from rebuttal_common import BENCHMARKS, METHODS, assert_text_sanitized
from verify_delivery import inspect_archive


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_analysis_archive_is_complete_sanitized_and_order_preserving(tmp_path: Path) -> None:
    sources: list[tuple[Path, str]] = []
    for method in METHODS:
        for benchmark in BENCHMARKS:
            base = tmp_path / "source" / method / benchmark
            prefix = f"analysis/runs/{method}/{benchmark}"
            network_value = "http" + "://10.1.2.3:8000/v1"
            sources.append((write(base / "metrics.json", json.dumps({"endpoint": network_value, "completed": 2}) + "\n"), f"{prefix}/metrics.json"))
            sources.append(
                (
                    write(base / "predictions.jsonl", '{"sample_id":"first","raw_output":"C:\\\\company\\\\one"}\n{"sample_id":"second","raw_output":"ok"}\n'),
                    f"{prefix}/predictions.jsonl",
                )
            )
            sources.append((write(base / "traces.jsonl", '{"sample_id":"first","steps":[]}\n{"sample_id":"second","steps":[]}\n'), f"{prefix}/traces.jsonl"))
    audit = tmp_path / "source" / "audit"
    for name in ("audit_input.jsonl", "audit_key.jsonl", "audit_results.jsonl"):
        sources.append((write(audit / name, '{"audit_id":"AUDIT-001"}\n'), f"analysis/audit/{name}"))
    sources.append((write(audit / "audit_summary.json", '{"n":1}\n'), "analysis/audit/audit_summary.json"))
    output = tmp_path / "analysis.tar.gz"
    build_archive(output, sources, archive_kind="analysis", staging_parent=tmp_path)
    result = inspect_archive(output, kind="analysis")
    assert result["members"] == len(sources) + 1
    with tarfile.open(output, "r:gz") as archive:
        payload = archive.extractfile("analysis/runs/full_fair/rewardbench2/predictions.jsonl").read().decode("utf-8")
        assert [json.loads(line)["sample_id"] for line in payload.splitlines()] == ["first", "second"]
        assert "10.1.2.3" not in archive.extractfile("analysis/runs/full_fair/rewardbench2/metrics.json").read().decode("utf-8")
        assert r"C:\\company" not in payload


def test_summary_archive_rejects_raw_prediction_members(tmp_path: Path) -> None:
    source = write(tmp_path / "predictions.jsonl", '{"sample_id":"one"}\n')
    output = tmp_path / "summary.tar.gz"
    build_archive(output, [(source, "summary/predictions.jsonl")], archive_kind="summary", staging_parent=tmp_path)
    with pytest.raises(ValueError, match="raw predictions"):
        inspect_archive(output, kind="summary")


def test_packager_refuses_unsupported_binary(tmp_path: Path) -> None:
    binary = tmp_path / "payload.bin"
    binary.write_bytes(b"http" + b"://10.0.0.1")
    with pytest.raises(ValueError, match="unsupported"):
        build_archive(tmp_path / "bad.tar.gz", [(binary, "analysis/payload.bin")], archive_kind="analysis", staging_parent=tmp_path)


def test_scanner_still_rejects_unsanitized_injected_text() -> None:
    with pytest.raises(ValueError):
        assert_text_sanitized("injected", "password" + "=supersecretvalue")
