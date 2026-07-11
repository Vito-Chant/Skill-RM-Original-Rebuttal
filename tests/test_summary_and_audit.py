from __future__ import annotations

import json
from pathlib import Path

import pytest

import summarize_original_v6_rebuttal as summary_module
from prepare_trace_audit import select_rows, strip_trace_identity
from rebuttal_common import BENCHMARKS, METHODS, sha256_file, write_json, write_jsonl
from summarize_trace_audit import build_summary, validate_results


def metric_for(benchmark: str, accuracy: float, rows: int) -> dict:
    common = {"n": rows, "completed": rows, "missing": 0}
    if benchmark == "rewardbench2":
        return {**common, "official_leaderboard_average": accuracy}
    if benchmark == "rmbench":
        return {**common, "overall": {"win_rate": accuracy}}
    return {**common, "overall": {"acc_rate": accuracy}}


def prediction(sample_id: str, correct: bool, resource: str = "references/generic_principles.md") -> dict:
    return {
        "sample_id": sample_id,
        "trace_id": sample_id,
        "correct": correct,
        "valid": True,
        "predicted_label": "A" if correct else "B",
        "skill_triggered": True,
        "resources_viewed": [resource],
        "resource_view_count": 1,
        "tool_call_count": 2,
        "python_sandbox_call_count": 0,
        "agent_step_count": 2,
        "latency_sec": 1.0,
    }


def trace(sample_id: str, resource: str = "references/generic_principles.md") -> dict:
    return {
        "sample_id": sample_id,
        "steps": [
            {
                "tool_calls": [{"name": "view_resource", "arguments": {"path": resource}}],
                "tool_results": [{"ok": True, "tool": "view_resource", "path": resource}],
            }
        ],
        "final": {"resources_viewed": [resource], "winner": "A", "valid": True},
    }


def test_summary_preserves_duplicate_ids_and_computes_complements(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(summary_module, "EXPECTED_PREDICTION_ROWS", {benchmark: 2 for benchmark in BENCHMARKS})
    monkeypatch.setattr(summary_module, "EXPECTED_TRACE_ROWS", {benchmark: 2 for benchmark in BENCHMARKS})
    accuracies = {"full_fair": 0.8, "subset_a": 0.7, "subset_b": 0.75}
    for method in METHODS:
        for benchmark in BENCHMARKS:
            run = tmp_path / method / benchmark
            run.mkdir(parents=True)
            write_json(run / "metrics.json", metric_for(benchmark, accuracies[method], 2))
            ids = ["duplicate", "duplicate"] if benchmark == "rewardbench2" else ["one", "two"]
            write_jsonl(run / "predictions.jsonl", [prediction(ids[0], True), prediction(ids[1], False)])
            write_jsonl(run / "traces.jsonl", [trace(ids[0]), trace(ids[1])])
    report = summary_module.build_report(tmp_path)
    assert report["all_runs_complete"] is True
    assert report["runs"]["full_fair"]["rewardbench2"]["process"]["duplicate_sample_id_rows"] == 1
    assert report["runs"]["full_fair"]["rewardbench2"]["prediction_rows"] == 2
    assert report["comparisons"]["rewardbench2"]["subset_a_delta"] == pytest.approx(-0.1)
    assert report["overall_complement"]["subset_mean"] == pytest.approx(0.725)


def test_audit_selection_is_balanced_and_identity_is_removed(tmp_path: Path) -> None:
    rows = [prediction(f"correct-{index:02d}", True) for index in range(8)]
    rows += [prediction(f"wrong-{index:02d}", False) for index in range(8)]
    write_jsonl(tmp_path / "predictions.jsonl", rows)
    selected = select_rows(tmp_path / "predictions.jsonl", "rewardbench2")
    assert sum(row["correct"] is True for row in selected) == 5
    assert sum(row["correct"] is False for row in selected) == 5
    cleaned = strip_trace_identity({"sample_id": "secret", "trace_id": "secret", "steps": [], "final": {}})
    assert "sample_id" not in cleaned and "trace_id" not in cleaned


def make_audit_fixture(audit_dir: Path) -> None:
    inputs = []
    keys = []
    results = []
    for index in range(30):
        audit_id = f"AUDIT-{index + 1:03d}"
        benchmark = BENCHMARKS[index // 10]
        correct = index % 10 < 5
        inputs.append(
            {
                "audit_id": audit_id,
                "benchmark": benchmark,
                "visible_messages": [{"role": "user", "content": "visible"}],
                "trace": {"steps": [], "final": {"resources_viewed": ["principle.generic"]}},
                "recorded_final": {"predicted_label": "A"},
            }
        )
        keys.append({"audit_id": audit_id, "benchmark": benchmark, "sample_id": f"hidden-{index}", "correct": correct})
        results.append(
            {
                "audit_id": audit_id,
                "resource_relevance": 4,
                "evidence_faithfulness": 4,
                "verdict_supported": 4,
                "auditability": 4,
                "error_type": "none",
                "brief_reason": "The recorded evidence and verdict are internally consistent.",
            }
        )
    write_jsonl(audit_dir / "audit_input.jsonl", inputs)
    write_jsonl(audit_dir / "audit_key.jsonl", keys)
    write_jsonl(audit_dir / "audit_results.jsonl", results)
    write_json(
        audit_dir / "audit_manifest.json",
        {
            "input_sha256": sha256_file(audit_dir / "audit_input.jsonl"),
            "key_sha256": sha256_file(audit_dir / "audit_key.jsonl"),
        },
    )


def test_audit_validates_before_unblinding_and_writes_bounded_cases(tmp_path: Path) -> None:
    make_audit_fixture(tmp_path)
    summary, cases = build_summary(tmp_path)
    assert summary["overall"]["n"] == 30
    assert summary["design"]["human_evaluation"] is False
    assert summary["faithfulness_among_resource_using_traces"]["n"] == 30
    assert len(cases) == 6
    assert (tmp_path / "audit_results.lock.json").is_file()
    assert all("sample_id" not in case for case in cases)


def test_audit_rejects_stale_input_and_forbidden_result_fields(tmp_path: Path) -> None:
    make_audit_fixture(tmp_path)
    results = list(json.loads(line) for line in (tmp_path / "audit_results.jsonl").read_text().splitlines())
    results[0]["sample_id"] = "leaked"
    write_jsonl(tmp_path / "audit_results.jsonl", results)
    with pytest.raises(ValueError, match="forbidden"):
        validate_results(tmp_path)
    make_audit_fixture(tmp_path)
    with (tmp_path / "audit_input.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="stale"):
        validate_results(tmp_path)
