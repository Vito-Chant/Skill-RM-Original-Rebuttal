#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from rebuttal_common import (
    BENCHMARKS,
    EXPECTED_PREDICTION_ROWS,
    EXPECTED_TRACE_ROWS,
    METHODS,
    count_nonempty_lines,
    iter_jsonl,
    write_json,
)


RESOURCE_FAMILIES = {
    "rubrics/generic_pairwise.md": "rubric.generic_pairwise",
    "references/generic_principles.md": "principle.generic",
    "references/bias_control.md": "bias_control",
    "references/generic_aggregation.md": "aggregation.generic",
    "references/output_format.md": "output_format",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def primary_accuracy(benchmark: str, metrics: dict[str, Any]) -> float:
    if benchmark == "rewardbench2":
        value = metrics.get("official_leaderboard_average")
    elif benchmark == "rmbench":
        value = (metrics.get("overall") or {}).get("win_rate")
        if value is None:
            value = ((((metrics.get("rmbench") or {}).get("global") or {}).get("overall") or {}).get("win_rate"))
    elif benchmark == "judgebench":
        value = (metrics.get("overall") or {}).get("acc_rate")
        if value is None:
            value = ((metrics.get("judgebench") or {}).get("overall") or {}).get("acc_rate")
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    if value is None:
        raise KeyError(f"Missing primary metric for {benchmark}")
    return float(value)


def numeric_mean(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return mean(values) if values else None


def bool_rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [bool(row[key]) for row in rows if isinstance(row.get(key), bool)]
    return mean(float(value) for value in values) if values else None


def normalize_resource(value: Any) -> str:
    text = str(value).replace("\\", "/").lstrip("./")
    for path, resource_id in RESOURCE_FAMILIES.items():
        if text == path or text.endswith("/" + path) or text == resource_id:
            return resource_id
    return text


def trace_tool_stats(traces: list[dict[str, Any]]) -> dict[str, Any]:
    tool_calls: Counter[str] = Counter()
    successful_views: Counter[str] = Counter()
    failed_views = 0
    nonexistent_views = 0
    traces_with_python = 0
    trace_signatures: Counter[tuple[str, tuple[str, ...]]] = Counter()
    for trace in traces:
        sample_id = str(trace.get("sample_id") or "")
        final_resources = tuple(sorted(normalize_resource(item) for item in ((trace.get("final") or {}).get("resources_viewed") or [])))
        trace_signatures[(sample_id, final_resources)] += 1
        used_python = False
        for step in trace.get("steps") or []:
            calls = step.get("tool_calls") or []
            results = step.get("tool_results") or []
            for index, call in enumerate(calls):
                name = str(call.get("name") or ((call.get("function") or {}).get("name")) or "unknown")
                tool_calls[name] += 1
                if name == "python_sandbox":
                    used_python = True
                if name != "view_resource":
                    continue
                result = results[index] if index < len(results) and isinstance(results[index], dict) else {}
                if result.get("ok"):
                    resource = result.get("path") or result.get("resource_id") or (call.get("arguments") or {}).get("path")
                    successful_views[normalize_resource(resource)] += 1
                else:
                    failed_views += 1
                    if "not found" in str(result.get("error", "")).lower():
                        nonexistent_views += 1
        traces_with_python += int(used_python)
    return {
        "tool_call_frequency": dict(sorted(tool_calls.items())),
        "successful_view_frequency": dict(sorted(successful_views.items())),
        "failed_resource_views": failed_views,
        "nonexistent_resource_views": nonexistent_views,
        "traces_with_python": traces_with_python,
        "trace_signatures": trace_signatures,
    }


def process_statistics(predictions: list[dict[str, Any]], traces: list[dict[str, Any]]) -> dict[str, Any]:
    resources = Counter(
        normalize_resource(resource)
        for row in predictions
        for resource in (row.get("resources_viewed") or [])
    )
    prediction_signatures: Counter[tuple[str, tuple[str, ...]]] = Counter()
    for row in predictions:
        if not row.get("trace_id"):
            continue
        signature = (
            str(row.get("sample_id") or ""),
            tuple(sorted(normalize_resource(item) for item in (row.get("resources_viewed") or []))),
        )
        prediction_signatures[signature] += 1
    tool_stats = trace_tool_stats(traces)
    trace_signatures = tool_stats.pop("trace_signatures")
    unmatched_prediction_claims = sum((prediction_signatures - trace_signatures).values())
    unmatched_trace_finals = sum((trace_signatures - prediction_signatures).values())
    sample_ids = [str(row.get("sample_id")) for row in predictions]
    request_errors = sum(bool(row.get("request_error")) for row in predictions)
    parse_errors = sum(bool(row.get("parse_error")) for row in predictions)
    invalid = sum(row.get("valid") is False for row in predictions)
    return {
        "prediction_rows": len(predictions),
        "trace_rows": len(traces),
        "unique_sample_ids": len(set(sample_ids)),
        "duplicate_sample_id_rows": len(sample_ids) - len(set(sample_ids)),
        "valid_rate": bool_rate(predictions, "valid"),
        "skill_trigger_rate": bool_rate(predictions, "skill_triggered"),
        "resource_view_rate": mean(float(bool(row.get("resources_viewed"))) for row in predictions) if predictions else None,
        "mean_resource_views": numeric_mean(predictions, "resource_view_count"),
        "mean_tool_calls": numeric_mean(predictions, "tool_call_count"),
        "mean_python_calls": numeric_mean(predictions, "python_sandbox_call_count"),
        "mean_agent_steps": numeric_mean(predictions, "agent_step_count"),
        "mean_latency_sec": numeric_mean(predictions, "latency_sec"),
        "resource_frequency": dict(sorted(resources.items())),
        "request_error_rows": request_errors,
        "parse_error_rows": parse_errors,
        "invalid_rows": invalid,
        "provenance": {
            "prediction_trace_resource_mismatches": unmatched_prediction_claims + unmatched_trace_finals,
            "prediction_claims_without_matching_trace": unmatched_prediction_claims,
            "trace_finals_without_matching_prediction": unmatched_trace_finals,
            "verifier_conflict_check": "not_applicable_no_verifier_resource_in_fair_pool",
            **tool_stats,
        },
    }


def summarize_run(run_dir: Path, benchmark: str) -> dict[str, Any]:
    required = [run_dir / "metrics.json", run_dir / "predictions.jsonl", run_dir / "traces.jsonl"]
    missing_files = [path.name for path in required if not path.is_file()]
    if missing_files:
        return {"complete": False, "missing_files": missing_files}
    metrics = load_json(run_dir / "metrics.json")
    predictions = list(iter_jsonl(run_dir / "predictions.jsonl"))
    traces = list(iter_jsonl(run_dir / "traces.jsonl"))
    expected_predictions = EXPECTED_PREDICTION_ROWS[benchmark]
    expected_traces = EXPECTED_TRACE_ROWS[benchmark]
    completed = int(metrics.get("completed") or metrics.get("n") or 0)
    missing = int(metrics.get("missing") or 0)
    accuracy = primary_accuracy(benchmark, metrics)
    complete = (
        completed == expected_predictions
        and missing == 0
        and len(predictions) == expected_predictions
        and len(traces) == expected_traces
    )
    return {
        "complete": complete,
        "expected_predictions": expected_predictions,
        "prediction_rows": len(predictions),
        "expected_traces": expected_traces,
        "trace_rows": len(traces),
        "metrics_completed": completed,
        "metrics_missing": missing,
        "accuracy": accuracy,
        "token_usage": "unavailable_in_original_v6_client",
        "process": process_statistics(predictions, traces),
    }


def summarize_attempts(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False, "attempts": 0}
    rows = list(iter_jsonl(path))
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        latest[(str(row.get("method")), str(row.get("benchmark")))] = row
    completed_durations = [
        float(row.get("duration_sec") or 0)
        for row in latest.values()
        if row.get("status") == "completed"
    ]
    return {
        "available": True,
        "attempts": len(rows),
        "latest_job_status": {
            f"{method}/{benchmark}": {
                "status": row.get("status"),
                "duration_sec": row.get("duration_sec"),
                "exit_code": row.get("exit_code"),
            }
            for (method, benchmark), row in sorted(latest.items())
        },
        "all_attempt_wall_clock_sec": sum(float(row.get("duration_sec") or 0) for row in rows),
        "completed_job_wall_clock_sec": sum(completed_durations),
    }


def build_report(run_root: Path) -> dict[str, Any]:
    runs = {
        method: {
            benchmark: summarize_run(run_root / method / benchmark, benchmark)
            for benchmark in BENCHMARKS
        }
        for method in METHODS
    }
    comparisons: dict[str, Any] = {}
    for benchmark in BENCHMARKS:
        values = [runs[method][benchmark].get("accuracy") for method in METHODS]
        if any(value is None for value in values):
            comparisons[benchmark] = {"complete": False}
            continue
        full, a, b = (float(value) for value in values)
        comparisons[benchmark] = {
            "complete": True,
            "full_fair": full,
            "subset_a": a,
            "subset_b": b,
            "subset_a_delta": a - full,
            "subset_b_delta": b - full,
            "subset_mean": mean((a, b)),
            "subset_min": min(a, b),
            "subset_max": max(a, b),
            "subset_mean_delta": mean((a, b)) - full,
        }
    method_averages: dict[str, float | None] = {}
    for method in METHODS:
        values = [runs[method][benchmark].get("accuracy") for benchmark in BENCHMARKS]
        method_averages[method] = mean(float(value) for value in values) if all(value is not None for value in values) else None
    full_average = method_averages["full_fair"]
    a_average = method_averages["subset_a"]
    b_average = method_averages["subset_b"]
    if None not in (full_average, a_average, b_average):
        full_value, a_value, b_value = float(full_average), float(a_average), float(b_average)
        overall_complement: dict[str, Any] = {
            "complete": True,
            "full_fair": full_value,
            "subset_a": a_value,
            "subset_b": b_value,
            "subset_a_delta": a_value - full_value,
            "subset_b_delta": b_value - full_value,
            "subset_mean": mean((a_value, b_value)),
            "subset_min": min(a_value, b_value),
            "subset_max": max(a_value, b_value),
            "subset_mean_delta": mean((a_value, b_value)) - full_value,
        }
    else:
        overall_complement = {"complete": False}
    repo_root = Path(__file__).resolve().parents[1]
    preparation_path = repo_root / "configs" / "rebuttal_original_v6" / "PREPARATION_REPORT.json"
    preparation = load_json(preparation_path) if preparation_path.is_file() else {}
    return {
        "report_version": "skill-rm-original-v6-rebuttal-summary-v1",
        "all_runs_complete": all(item.get("complete") for method in runs.values() for item in method.values()),
        "expected_jobs": 9,
        "runs": runs,
        "comparisons": comparisons,
        "method_averages": method_averages,
        "overall_complement": overall_complement,
        "run_attempts": summarize_attempts(run_root / "run_attempts.jsonl"),
        "fair_leakage_audit": preparation.get("fair_leakage_audit"),
        "resource_subset_metadata": {
            "seed": 0,
            "subset_a": ["rubric.generic_pairwise", "bias_control"],
            "subset_b": ["principle.generic", "aggregation.generic"],
            "fixed": ["SKILL.md", "output_format", "final_answer", "tool.python_sandbox"],
        },
        "compute_note": "Exact token usage is unavailable because the original v6 client did not retain API usage fields.",
    }


def fmt(value: Any) -> str:
    return "NA" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Original v6 Minimal Rebuttal Results",
        "",
        f"- all 9 runs complete: `{report['all_runs_complete']}`",
        "- token usage: unavailable in the original v6 client",
        "",
        "| Method | RewardBench2 | RM-Bench | JudgeBench | Average | Delta vs full |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    full = report["method_averages"]["full_fair"]
    for method in METHODS:
        values = [report["runs"][method][benchmark]["accuracy"] for benchmark in BENCHMARKS]
        average = report["method_averages"][method]
        delta = None if method == "full_fair" or average is None or full is None else average - full
        lines.append(
            f"| {method} | {fmt(values[0])} | {fmt(values[1])} | {fmt(values[2])} | {fmt(average)} | {fmt(delta)} |"
        )
    overall = report["overall_complement"]
    lines.extend(
        [
            "",
            "## Complementary Half-Pool Summary",
            "",
            f"A/B average mean: {fmt(overall.get('subset_mean'))}; range: "
            f"[{fmt(overall.get('subset_min'))}, {fmt(overall.get('subset_max'))}]; "
            f"mean delta vs full: {fmt(overall.get('subset_mean_delta'))}.",
            "",
            "## Process Statistics",
            "",
            "| Method | Benchmark | Trigger rate | Mean views | Mean tools | Mean Python | Mean steps | Mean latency (s) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in METHODS:
        for benchmark in BENCHMARKS:
            process = report["runs"][method][benchmark].get("process")
            if process is None:
                lines.append(f"| {method} | {benchmark} | NA | NA | NA | NA | NA | NA |")
                continue
            lines.append(
                f"| {method} | {benchmark} | {fmt(process['skill_trigger_rate'])} | "
                f"{fmt(process['mean_resource_views'])} | {fmt(process['mean_tool_calls'])} | "
                f"{fmt(process['mean_python_calls'])} | {fmt(process['mean_agent_steps'])} | "
                f"{fmt(process['mean_latency_sec'])} |"
            )
    lines.extend(
        [
            "",
            "The fair setup static audit uses only generic sample-visible resources and a local Python sandbox; "
            "no benchmark-specific reference, gold-label, ground-truth, or verifier resource is present.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize the nine original-v6 rebuttal runs.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    report = build_report(run_root)
    write_json(run_root / "rebuttal_summary.json", report)
    (run_root / "rebuttal_summary.md").write_text(markdown(report), encoding="utf-8", newline="\n")
    print(markdown(report))
    if not report["all_runs_complete"] and not args.allow_partial:
        raise SystemExit("One or more expected runs are incomplete.")


if __name__ == "__main__":
    main()
