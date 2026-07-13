#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from prepare_human_trace_audit import evidence_types, occurrence_rows, occurrence_traces, outcome_group
from rebuttal_common import BENCHMARKS, write_json


PROCESS_FIELDS = (
    "llm_call_count",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "tool_call_count",
    "python_sandbox_call_count",
    "resource_view_count",
    "agent_step_count",
    "end_to_end_latency_sec",
)


def numeric_mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]
    return mean(values) if values else None


def final_reason(trace: dict[str, Any]) -> str:
    for step in reversed(trace.get("steps") or []):
        calls = step.get("tool_calls") or []
        for call in calls:
            name = str(call.get("name") or ((call.get("function") or {}).get("name")) or "")
            if name != "final_answer":
                continue
            arguments = call.get("arguments") or ((call.get("function") or {}).get("arguments")) or {}
            if isinstance(arguments, dict):
                return str(arguments.get("rationale") or arguments.get("reason") or "")
        content = str(step.get("assistant_content") or step.get("assistant_raw") or "").strip()
        if content:
            return content[:1000]
    return ""


def build_report(run_root: Path) -> dict[str, Any]:
    benchmarks: dict[str, Any] = {}
    case_candidates: list[dict[str, Any]] = []
    for benchmark in BENCHMARKS:
        direct = occurrence_rows(run_root / "direct" / benchmark / "predictions.jsonl")
        skill = occurrence_rows(run_root / "skill_rm" / benchmark / "predictions.jsonl")
        traces = occurrence_traces(run_root / "skill_rm" / benchmark / "traces.jsonl")
        paired: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for key in sorted(set(direct).intersection(skill)):
            group = outcome_group(direct[key], skill[key])
            paired[group].append(skill[key])
            trace = traces.get(key)
            kinds = evidence_types(trace) if trace else set()
            if group == "improved" and kinds and not skill[key].get("request_error") and not skill[key].get("tool_error_count"):
                case_candidates.append(
                    {
                        "benchmark": benchmark,
                        "sample_id": key[0],
                        "occurrence_index": key[1],
                        "evidence_types": sorted(kinds),
                        "direct_label": direct[key].get("predicted_label"),
                        "skill_label": skill[key].get("predicted_label"),
                        "final_reason_excerpt": final_reason(trace),
                    }
                )
        benchmarks[benchmark] = {
            "paired_examples": sum(len(rows) for rows in paired.values()),
            "outcome_counts": {group: len(rows) for group, rows in paired.items()},
            "process_by_outcome": {
                group: {field: numeric_mean(rows, field) for field in PROCESS_FIELDS}
                for group, rows in paired.items()
            },
        }
    return {
        "version": "skill-rm-matched-trace-analysis-v1",
        "benchmarks": benchmarks,
        "representative_case_candidates": case_candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze orchestration and mine improved evidence-bearing cases.")
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    report = build_report(run_root)
    write_json(run_root / "trace_analysis.json", report)
    lines = ["# Matched Orchestration Analysis", ""]
    for benchmark in BENCHMARKS:
        item = report["benchmarks"][benchmark]
        counts = item["outcome_counts"]
        lines.append(
            f"- `{benchmark}`: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        )
    lines.extend(["", f"Evidence-bearing improved case candidates: {len(report['representative_case_candidates'])}", ""])
    (run_root / "trace_analysis.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
