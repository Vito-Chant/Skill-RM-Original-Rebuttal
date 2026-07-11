#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from rebuttal_common import iter_jsonl, sha256_file, write_json


SCORE_FIELDS = ("resource_relevance", "evidence_faithfulness", "verdict_supported", "auditability")
ERROR_TYPES = {
    "none",
    "irrelevant_resource",
    "unsupported_evidence",
    "tool_misuse",
    "reasoning_gap",
    "final_verdict_mismatch",
    "insufficient_trace",
    "other",
}


def validate_results(audit_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads((audit_dir / "audit_manifest.json").read_text(encoding="utf-8"))
    input_path = audit_dir / "audit_input.jsonl"
    key_path = audit_dir / "audit_key.jsonl"
    results_path = audit_dir / "audit_results.jsonl"
    if sha256_file(input_path) != manifest.get("input_sha256"):
        raise ValueError("Audit input changed after sampling; results are stale.")
    if sha256_file(key_path) != manifest.get("key_sha256"):
        raise ValueError("Audit key changed after sampling.")
    inputs = list(iter_jsonl(input_path))
    results = list(iter_jsonl(results_path))
    if len(inputs) != 30 or len(results) != 30:
        raise ValueError(f"Expected 30 inputs and 30 results, got {len(inputs)} and {len(results)}")
    input_ids = [row.get("audit_id") for row in inputs]
    result_ids = [row.get("audit_id") for row in results]
    if len(set(result_ids)) != 30 or set(result_ids) != set(input_ids):
        raise ValueError("Audit result IDs must match all 30 input IDs exactly once.")
    forbidden = {"sample_id", "gold_label", "correct", "official_score", "prompt", "candidate"}
    by_id: dict[str, dict[str, Any]] = {}
    for row in results:
        extra_forbidden = forbidden.intersection(row)
        if extra_forbidden:
            raise ValueError(f"Blinded result includes forbidden fields: {sorted(extra_forbidden)}")
        for field in SCORE_FIELDS:
            if not isinstance(row.get(field), int) or isinstance(row.get(field), bool) or not 1 <= row[field] <= 5:
                raise ValueError(f"{row.get('audit_id')} has invalid {field}")
        if row.get("error_type") not in ERROR_TYPES:
            raise ValueError(f"{row.get('audit_id')} has invalid error_type")
        reason = row.get("brief_reason")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 1200:
            raise ValueError(f"{row.get('audit_id')} has invalid brief_reason")
        by_id[str(row["audit_id"])] = row
    ordered_results = [by_id[str(audit_id)] for audit_id in input_ids]
    return inputs, ordered_results, manifest


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "mean_scores": {field: mean(float(row[field]) for row in rows) for field in SCORE_FIELDS},
        "score_4_or_5_rate": {
            field: mean(float(int(row[field]) >= 4) for row in rows) for field in SCORE_FIELDS
        },
        "error_types": dict(sorted(Counter(str(row["error_type"]) for row in rows).items())),
    }


def build_summary(audit_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    inputs, results, manifest = validate_results(audit_dir)
    input_by_id = {str(row["audit_id"]): row for row in inputs}
    results_hash = sha256_file(audit_dir / "audit_results.jsonl")
    write_json(
        audit_dir / "audit_results.lock.json",
        {
            "input_sha256": manifest["input_sha256"],
            "results_sha256": results_hash,
            "validated_rows": 30,
            "schema_valid": True,
            "blinded_validation_completed_before_key_join": True,
        },
    )

    # The hidden key is opened only after all blinded validation and result hashing above succeed.
    keys = list(iter_jsonl(audit_dir / "audit_key.jsonl"))
    key_by_id = {str(row["audit_id"]): row for row in keys}
    joined: list[dict[str, Any]] = []
    for result in results:
        audit_id = str(result["audit_id"])
        key = key_by_id[audit_id]
        trace_final = ((input_by_id[audit_id].get("trace") or {}).get("final") or {})
        joined.append(
            {
                **result,
                "benchmark": key["benchmark"],
                "correct": bool(key["correct"]),
                "used_resource": bool(trace_final.get("resources_viewed")),
            }
        )
    by_benchmark = {
        benchmark: aggregate([row for row in joined if row["benchmark"] == benchmark])
        for benchmark in sorted({str(row["benchmark"]) for row in joined})
    }
    by_outcome = {
        outcome: aggregate([row for row in joined if row["correct"] is expected])
        for outcome, expected in (("correct", True), ("incorrect", False))
    }
    resource_rows = [row for row in joined if row["used_resource"]]
    summary = {
        "version": "skill-rm-original-v6-model-audit-summary-v1",
        "input_sha256": manifest["input_sha256"],
        "results_sha256": results_hash,
        "design": {
            "auditor": "Codex model-based audit",
            "human_evaluation": False,
            "outcome_stratified": True,
            "method_blinded": False,
            "population_estimate": False,
            "cases": 30,
        },
        "overall": aggregate(joined),
        "by_benchmark": by_benchmark,
        "by_outcome": by_outcome,
        "faithfulness_among_resource_using_traces": (
            {
                "n": len(resource_rows),
                "mean": mean(float(row["evidence_faithfulness"]) for row in resource_rows),
                "score_4_or_5_rate": mean(float(row["evidence_faithfulness"] >= 4) for row in resource_rows),
            }
            if resource_rows
            else {"n": 0, "mean": None, "score_4_or_5_rate": None}
        ),
        "interpretation_boundary": (
            "Outcome-stratified, non-method-blinded model audit of 30 new full-fair traces; "
            "not human evaluation and not an unbiased population estimate."
        ),
    }
    cases: list[dict[str, Any]] = []
    for benchmark in by_benchmark:
        for expected in (True, False):
            pool = [row for row in joined if row["benchmark"] == benchmark and row["correct"] is expected]
            chosen = sorted(
                pool,
                key=lambda row: (
                    row["verdict_supported"],
                    row["evidence_faithfulness"],
                    row["audit_id"],
                ),
            )[0]
            cases.append(
                {
                    "audit_id": chosen["audit_id"],
                    "benchmark": benchmark,
                    "outcome": "correct" if expected else "incorrect",
                    **{field: chosen[field] for field in SCORE_FIELDS},
                    "error_type": chosen["error_type"],
                    "brief_reason": chosen["brief_reason"],
                }
            )
    return summary, cases


def markdown(summary: dict[str, Any], cases: list[dict[str, Any]]) -> str:
    means = summary["overall"]["mean_scores"]
    lines = [
        "# Model-Based Trace Audit",
        "",
        "This is an outcome-stratified, non-method-blinded Codex audit of 30 new full-fair traces. "
        "It is not human evaluation and is not an unbiased population estimate.",
        "",
        "| Dimension | Mean (1-5) | Rate >= 4 |",
        "| --- | ---: | ---: |",
    ]
    for field in SCORE_FIELDS:
        lines.append(
            f"| {field} | {means[field]:.3f} | {summary['overall']['score_4_or_5_rate'][field]:.3f} |"
        )
    conditional = summary["faithfulness_among_resource_using_traces"]
    lines.extend(
        [
            "",
            f"Evidence-faithfulness among resource-using traces: n={conditional['n']}, "
            f"mean={conditional['mean'] if conditional['mean'] is not None else 'NA'}.",
            "",
            "## Bounded Qualitative Cases",
            "",
        ]
    )
    for case in cases:
        lines.append(
            f"- `{case['audit_id']}` ({case['benchmark']}, {case['outcome']}): "
            f"{case['error_type']}; {case['brief_reason']}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate, lock, unblind, and summarize the model trace audit.")
    parser.add_argument("--audit-dir", required=True)
    args = parser.parse_args()
    audit_dir = Path(args.audit_dir).resolve()
    summary, cases = build_summary(audit_dir)
    write_json(audit_dir / "audit_summary.json", summary)
    write_json(audit_dir / "qualitative_cases.json", cases)
    (audit_dir / "audit_summary.md").write_text(markdown(summary, cases), encoding="utf-8", newline="\n")
    print(markdown(summary, cases))


if __name__ == "__main__":
    main()
