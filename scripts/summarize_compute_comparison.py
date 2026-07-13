#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

from rebuttal_common import BENCHMARKS, EXPECTED_PREDICTION_ROWS, EXPECTED_TRACE_ROWS, iter_jsonl, write_json


METHODS = ("direct", "skill_rm")
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def primary_accuracy(benchmark: str, metrics: dict[str, Any]) -> float:
    if benchmark == "rewardbench2":
        value = metrics.get("official_leaderboard_average")
    elif benchmark == "rmbench":
        value = (metrics.get("overall") or {}).get("win_rate")
        if value is None:
            value = ((((metrics.get("rmbench") or {}).get("global") or {}).get("overall") or {}).get("win_rate"))
    else:
        value = (metrics.get("overall") or {}).get("acc_rate")
        if value is None:
            value = ((metrics.get("judgebench") or {}).get("overall") or {}).get("acc_rate")
    if value is None:
        raise KeyError(f"Missing primary accuracy for {benchmark}")
    return float(value)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def latest_attempts(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.is_file():
        return latest
    for row in iter_jsonl(path):
        latest[(str(row.get("method")), str(row.get("benchmark")))] = row
    return latest


def summarize_run(
    run_dir: Path,
    method: str,
    benchmark: str,
    attempt: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    predictions_path = run_dir / "predictions.jsonl"
    required = [metrics_path, predictions_path]
    if method == "skill_rm":
        required.append(run_dir / "traces.jsonl")
    missing_files = [path.name for path in required if not path.is_file()]
    if missing_files:
        return {"complete": False, "missing_files": missing_files}

    metrics = load_json(metrics_path)
    rows = list(iter_jsonl(predictions_path))
    traces = list(iter_jsonl(run_dir / "traces.jsonl")) if method == "skill_rm" else []
    expected_rows = EXPECTED_PREDICTION_ROWS[benchmark]
    invoked = [row for row in rows if int(row.get("llm_call_count") or 0) > 0]
    usage_complete_rows = [row for row in invoked if row.get("usage_complete") is True]
    usage_complete = bool(invoked) and len(usage_complete_rows) == len(invoked)
    token_totals = {
        key: sum(int(row[key]) for row in invoked if isinstance(row.get(key), int)) if usage_complete else None
        for key in TOKEN_FIELDS
    }
    latencies = [
        float(row.get("end_to_end_latency_sec", row.get("latency_sec")))
        for row in invoked
        if isinstance(row.get("end_to_end_latency_sec", row.get("latency_sec")), (int, float))
    ]
    model_wait = [float(row["model_wait_sec"]) for row in invoked if isinstance(row.get("model_wait_sec"), (int, float))]
    calls = sum(int(row.get("llm_call_count") or 0) for row in invoked)
    job_wall = float((attempt or {}).get("duration_sec") or 0.0)
    completed = int(metrics.get("completed") or metrics.get("n") or 0)
    missing = int(metrics.get("missing") or 0)
    trace_complete = method != "skill_rm" or len(traces) == EXPECTED_TRACE_ROWS[benchmark]
    complete = (
        len(rows) == expected_rows
        and completed == expected_rows
        and missing == 0
        and trace_complete
        and usage_complete
        and bool(attempt)
        and attempt.get("status") == "completed"
    )
    return {
        "complete": complete,
        "accuracy": primary_accuracy(benchmark, metrics),
        "benchmark_examples": len(rows),
        "model_invoked_examples": len(invoked),
        "trace_rows": len(traces),
        "expected_trace_rows": EXPECTED_TRACE_ROWS[benchmark] if method == "skill_rm" else 0,
        "usage_complete_rows": len(usage_complete_rows),
        "usage_complete_rate": len(usage_complete_rows) / len(invoked) if invoked else None,
        "llm_calls": calls,
        "llm_calls_per_example": calls / len(rows) if rows else None,
        "llm_calls_per_invoked_example": calls / len(invoked) if invoked else None,
        **{f"total_{key}": value for key, value in token_totals.items()},
        **{
            f"{key}_per_example": (token_totals[key] / len(rows) if token_totals[key] is not None and rows else None)
            for key in TOKEN_FIELDS
        },
        **{
            f"{key}_per_invoked_example": (
                token_totals[key] / len(invoked) if token_totals[key] is not None and invoked else None
            )
            for key in TOKEN_FIELDS
        },
        "mean_end_to_end_latency_sec": mean(latencies) if latencies else None,
        "median_end_to_end_latency_sec": median(latencies) if latencies else None,
        "p95_end_to_end_latency_sec": percentile(latencies, 0.95),
        "mean_model_wait_sec": mean(model_wait) if model_wait else None,
        "job_wall_clock_sec": job_wall,
        "throughput_examples_per_sec": len(rows) / job_wall if job_wall > 0 else None,
        "invalid_rows": sum(row.get("valid") is False for row in rows),
        "request_error_rows": sum(bool(row.get("request_error")) for row in rows),
    }


def safe_ratio(numerator: Any, denominator: Any) -> float | None:
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)) or denominator == 0:
        return None
    return float(numerator) / float(denominator)


def build_report(run_root: Path) -> dict[str, Any]:
    attempts = latest_attempts(run_root / "run_attempts.jsonl")
    runs = {
        method: {
            benchmark: summarize_run(
                run_root / method / benchmark,
                method,
                benchmark,
                attempts.get((method, benchmark)),
            )
            for benchmark in BENCHMARKS
        }
        for method in METHODS
    }
    comparisons: dict[str, Any] = {}
    for benchmark in BENCHMARKS:
        direct = runs["direct"][benchmark]
        skill = runs["skill_rm"][benchmark]
        comparisons[benchmark] = {
            "accuracy_delta_points": (
                100.0 * (skill["accuracy"] - direct["accuracy"])
                if isinstance(skill.get("accuracy"), float) and isinstance(direct.get("accuracy"), float)
                else None
            ),
            "prompt_token_ratio": safe_ratio(skill.get("prompt_tokens_per_example"), direct.get("prompt_tokens_per_example")),
            "completion_token_ratio": safe_ratio(skill.get("completion_tokens_per_example"), direct.get("completion_tokens_per_example")),
            "total_token_ratio": safe_ratio(skill.get("total_tokens_per_example"), direct.get("total_tokens_per_example")),
            "llm_call_ratio": safe_ratio(skill.get("llm_calls_per_example"), direct.get("llm_calls_per_example")),
            "mean_latency_ratio": safe_ratio(skill.get("mean_end_to_end_latency_sec"), direct.get("mean_end_to_end_latency_sec")),
            "job_wall_clock_ratio": safe_ratio(skill.get("job_wall_clock_sec"), direct.get("job_wall_clock_sec")),
        }
    return {
        "report_version": "skill-rm-compute-comparison-v1",
        "expected_jobs": 6,
        "all_runs_complete": all(run.get("complete") for method in runs.values() for run in method.values()),
        "usage_source": "vllm_openai_compatible_response",
        "methods": list(METHODS),
        "benchmarks": list(BENCHMARKS),
        "runs": runs,
        "comparisons": comparisons,
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Qwen3.5-27B Matched Compute Comparison",
        "",
        f"- all six jobs complete with exact server usage: `{report['all_runs_complete']}`",
        "- token source: vLLM OpenAI-compatible non-streaming response usage",
        "",
        "| Benchmark | Method | Acc. | Examples / invoked | Calls/example | Prompt tok/example | Completion tok/example | Total tok/example | Mean / p50 / p95 latency (s) | Job wall (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for benchmark in BENCHMARKS:
        for method in METHODS:
            run = report["runs"][method][benchmark]
            accuracy = run.get("accuracy")
            lines.append(
                f"| {benchmark} | {method} | {fmt(100 * accuracy if isinstance(accuracy, (int, float)) else None, 2)} | "
                f"{run.get('benchmark_examples', 'NA')} / {run.get('model_invoked_examples', 'NA')} | "
                f"{fmt(run.get('llm_calls_per_example'))} | {fmt(run.get('prompt_tokens_per_example'), 1)} | "
                f"{fmt(run.get('completion_tokens_per_example'), 1)} | {fmt(run.get('total_tokens_per_example'), 1)} | "
                f"{fmt(run.get('mean_end_to_end_latency_sec'), 1)} / {fmt(run.get('median_end_to_end_latency_sec'), 1)} / "
                f"{fmt(run.get('p95_end_to_end_latency_sec'), 1)} | {fmt(run.get('job_wall_clock_sec'), 1)} |"
            )
    lines.extend(
        [
            "",
            "| Benchmark | Accuracy delta (points) | Prompt-token ratio | Completion-token ratio | Total-token ratio | Call ratio | Mean-latency ratio | Job-wall ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for benchmark in BENCHMARKS:
        item = report["comparisons"][benchmark]
        lines.append(
            f"| {benchmark} | {fmt(item['accuracy_delta_points'], 2)} | {fmt(item['prompt_token_ratio'])} | "
            f"{fmt(item['completion_token_ratio'])} | {fmt(item['total_token_ratio'])} | "
            f"{fmt(item['llm_call_ratio'])} | {fmt(item['mean_latency_ratio'])} | {fmt(item['job_wall_clock_ratio'])} |"
        )
    lines.extend(
        [
            "",
            "RewardBench2 reports both all benchmark examples and examples that invoked the model so its separate Ties path remains explicit.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize the six matched compute runs.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    report = build_report(run_root)
    write_json(run_root / "compute_summary.json", report)
    (run_root / "compute_summary.md").write_text(markdown(report), encoding="utf-8", newline="\n")
    print(markdown(report))
    if not report["all_runs_complete"] and not args.allow_partial:
        raise SystemExit("One or more compute runs are incomplete or missing exact usage.")


if __name__ == "__main__":
    main()
