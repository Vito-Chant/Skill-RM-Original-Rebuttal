from __future__ import annotations

import json
from pathlib import Path

import pytest

import summarize_compute_comparison as compute_summary
from check_endpoints import normalized_usage
from prepare_human_trace_audit import select_cases
from rebuttal_common import BENCHMARKS, write_json, write_jsonl
from skillrm.qwen_baseline import aggregate_usage_fields, normalized_usage_fields
from summarize_human_trace_audit import weighted_kappa


ROOT = Path(__file__).resolve().parents[1]


def test_usage_validation_and_multistep_accumulation() -> None:
    assert normalized_usage({"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}) == {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
    }
    assert normalized_usage({"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 13}) is None
    assert normalized_usage_fields(None, call_succeeded=True)["usage_complete"] is False
    total = aggregate_usage_fields(
        [
            {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12, "usage_complete": True, "llm_call_count": 1, "request_attempt_count": 1, "latency_sec": 0.5},
            {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23, "usage_complete": True, "llm_call_count": 1, "request_attempt_count": 2, "latency_sec": 0.75},
        ]
    )
    assert total == {
        "prompt_tokens": 30,
        "completion_tokens": 5,
        "total_tokens": 35,
        "usage_complete": True,
        "llm_call_count": 2,
        "request_attempt_count": 3,
        "model_wait_sec": 1.25,
    }


def test_compute_runner_declares_exactly_six_jobs() -> None:
    text = (ROOT / "scripts/run_compute_comparison.sh").read_text(encoding="utf-8")
    jobs = [line.strip() for line in text.splitlines() if line.strip().startswith("run_one ")]
    assert jobs == ['run_one direct "$bench"', 'run_one skill_rm "$bench"']
    assert "for bench in rewardbench2 rmbench judgebench" in text
    assert "subset_a" not in "\n".join(jobs)
    assert "subset_b" not in "\n".join(jobs)


def metric(benchmark: str, rows: int, accuracy: float) -> dict:
    common = {"n": rows, "completed": rows, "missing": 0}
    if benchmark == "rewardbench2":
        return {**common, "official_leaderboard_average": accuracy}
    if benchmark == "rmbench":
        return {**common, "overall": {"win_rate": accuracy}}
    return {**common, "overall": {"acc_rate": accuracy}}


def test_compute_summary_requires_exact_usage_and_reports_ratios(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compute_summary, "EXPECTED_PREDICTION_ROWS", {benchmark: 2 for benchmark in BENCHMARKS})
    monkeypatch.setattr(compute_summary, "EXPECTED_TRACE_ROWS", {benchmark: 2 for benchmark in BENCHMARKS})
    attempts = []
    for method in compute_summary.METHODS:
        for benchmark in BENCHMARKS:
            run = tmp_path / method / benchmark
            write_json(run / "metrics.json", metric(benchmark, 2, 0.8 if method == "direct" else 0.85))
            rows = []
            for index in range(2):
                factor = 2 if method == "skill_rm" else 1
                rows.append({"sample_id": str(index), "valid": True, "correct": True, "llm_call_count": factor, "usage_complete": True, "prompt_tokens": 10 * factor, "completion_tokens": 2 * factor, "total_tokens": 12 * factor, "model_wait_sec": 0.5 * factor, "end_to_end_latency_sec": 1.0 * factor})
            write_jsonl(run / "predictions.jsonl", rows)
            if method == "skill_rm":
                write_jsonl(run / "traces.jsonl", [{"sample_id": "0", "steps": []}, {"sample_id": "1", "steps": []}])
            attempts.append({"method": method, "benchmark": benchmark, "status": "completed", "duration_sec": 4.0 * (2 if method == "skill_rm" else 1)})
    write_jsonl(tmp_path / "run_attempts.jsonl", attempts)
    report = compute_summary.build_report(tmp_path)
    assert report["all_runs_complete"] is True
    assert report["comparisons"]["rmbench"]["total_token_ratio"] == 2
    assert report["comparisons"]["judgebench"]["job_wall_clock_ratio"] == 2


def candidate(index: int, group: str, evidence: list[str]) -> dict:
    return {"key": (f"id-{index}", 0), "outcome_group": group, "evidence_types": evidence}


def test_human_selection_is_ten_and_tolerates_sparse_regressions() -> None:
    cases = [candidate(index, "improved", ["python", "resource"] if index < 3 else ["python"]) for index in range(8)]
    cases += [candidate(20 + index, "both_correct", ["resource"] if index < 4 else ["python"]) for index in range(8)]
    cases += [candidate(40, "regressed", ["python"])]
    selected = select_cases(cases, "rewardbench2")
    assert len(selected) == 10
    assert sum("resource" in case["evidence_types"] for case in selected) >= 3
    assert sum("python" in case["evidence_types"] for case in selected) >= 5
    assert any(case["outcome_group"] == "regressed" for case in selected)


def test_weighted_kappa_is_one_for_identical_labels() -> None:
    assert weighted_kappa(["Yes", "Partial", "No", "N/A"], ["Yes", "Partial", "No", "N/A"]) == pytest.approx(1.0)
