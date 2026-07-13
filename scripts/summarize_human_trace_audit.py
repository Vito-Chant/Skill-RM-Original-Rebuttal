#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rebuttal_common import iter_jsonl, sha256_file, write_json


DIMENSIONS = ("action_relevance", "evidence_validity", "evidence_faithfulness", "verdict_support", "auditability")
LABELS = ("No", "Partial", "Yes")
ALLOWED_LABELS = {*LABELS, "N/A"}
ERROR_TYPES = {
    "none",
    "irrelevant_action",
    "invalid_evidence",
    "unsupported_claim",
    "tool_failure",
    "reasoning_gap",
    "insufficient_trace",
    "other",
}


def normalize_label(value: str) -> str:
    text = value.strip().lower()
    mapping = {"yes": "Yes", "partial": "Partial", "no": "No", "n/a": "N/A", "na": "N/A"}
    if text not in mapping:
        raise ValueError(f"Invalid audit label: {value!r}")
    return mapping[text]


def read_form(path: Path, expected_ids: list[str], *, require_complete: bool = True) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        audit_id = str(row.get("audit_id") or "").strip()
        if not audit_id or audit_id in by_id:
            raise ValueError(f"Missing or duplicate audit_id in {path}: {audit_id!r}")
        normalized = {"audit_id": audit_id}
        for dimension in DIMENSIONS:
            raw = str(row.get(dimension) or "").strip()
            if require_complete or raw:
                normalized[dimension] = normalize_label(raw)
            else:
                normalized[dimension] = ""
        error_type = str(row.get("error_type") or "").strip().lower()
        if require_complete and error_type not in ERROR_TYPES:
            raise ValueError(f"Invalid error_type in {path}: {error_type!r}")
        normalized["error_type"] = error_type
        normalized["brief_reason"] = str(row.get("brief_reason") or "").strip()
        if require_complete and not normalized["brief_reason"]:
            raise ValueError(f"Missing brief_reason for {audit_id} in {path}")
        by_id[audit_id] = normalized
    if set(by_id) != set(expected_ids):
        raise ValueError(f"Audit ID mismatch in {path}")
    return by_id


def weighted_kappa(left: list[str], right: list[str]) -> float | None:
    pairs = [(a, b) for a, b in zip(left, right) if a != "N/A" and b != "N/A"]
    if not pairs:
        return None
    index = {label: i for i, label in enumerate(LABELS)}
    max_distance = float((len(LABELS) - 1) ** 2)
    observed = sum(((index[a] - index[b]) ** 2) / max_distance for a, b in pairs) / len(pairs)
    left_counts = Counter(a for a, _ in pairs)
    right_counts = Counter(b for _, b in pairs)
    expected = 0.0
    for a in LABELS:
        for b in LABELS:
            weight = ((index[a] - index[b]) ** 2) / max_distance
            expected += weight * (left_counts[a] / len(pairs)) * (right_counts[b] / len(pairs))
    if expected == 0:
        return 1.0 if observed == 0 else None
    return 1.0 - observed / expected


def agreement(left: dict[str, dict[str, str]], right: dict[str, dict[str, str]], ids: list[str]) -> dict[str, Any]:
    dimensions: dict[str, Any] = {}
    for dimension in DIMENSIONS:
        a = [left[audit_id][dimension] for audit_id in ids]
        b = [right[audit_id][dimension] for audit_id in ids]
        dimensions[dimension] = {
            "n": len(ids),
            "raw_agreement": sum(x == y for x, y in zip(a, b)) / len(ids),
            "ordinal_weighted_kappa": weighted_kappa(a, b),
            "disagreements": sum(x != y for x, y in zip(a, b)),
        }
    errors_a = [left[audit_id]["error_type"] for audit_id in ids]
    errors_b = [right[audit_id]["error_type"] for audit_id in ids]
    return {
        "dimensions": dimensions,
        "error_type_raw_agreement": sum(a == b for a, b in zip(errors_a, errors_b)) / len(ids),
    }


def write_consensus_form(path: Path, left: dict[str, dict[str, str]], right: dict[str, dict[str, str]], ids: list[str]) -> None:
    fields = ["audit_id", *DIMENSIONS, "error_type", "brief_reason"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for audit_id in ids:
            row: dict[str, str] = {"audit_id": audit_id}
            for field in (*DIMENSIONS, "error_type"):
                row[field] = left[audit_id][field] if left[audit_id][field] == right[audit_id][field] else ""
            row["brief_reason"] = left[audit_id]["brief_reason"] if left[audit_id]["brief_reason"] == right[audit_id]["brief_reason"] else ""
            writer.writerow(row)


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a two-author human trace evaluation.")
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--annotator-a", default="annotator_a.csv")
    parser.add_argument("--annotator-b", default="annotator_b.csv")
    parser.add_argument("--consensus", default="consensus.csv")
    args = parser.parse_args()
    audit_dir = Path(args.audit_dir).resolve()
    manifest = json.loads((audit_dir / "manifest.json").read_text(encoding="utf-8"))
    input_path = audit_dir / "audit_input.jsonl"
    key_path = audit_dir / "sealed_outcome_key.jsonl"
    if sha256_file(input_path) != manifest.get("input_sha256") or sha256_file(key_path) != manifest.get("key_sha256"):
        raise ValueError("Human audit input or sealed key changed after preparation.")
    inputs = list(iter_jsonl(input_path))
    ids = [str(row["audit_id"]) for row in inputs]
    left = read_form(audit_dir / args.annotator_a, ids)
    right = read_form(audit_dir / args.annotator_b, ids)
    agreement_summary = agreement(left, right, ids)
    write_json(audit_dir / "agreement_summary.json", agreement_summary)

    consensus_path = audit_dir / args.consensus
    if not consensus_path.is_file():
        write_consensus_form(audit_dir / "consensus_template.csv", left, right, ids)
        print(json.dumps(agreement_summary, indent=2))
        print("Complete consensus_template.csv, save it as consensus.csv, and rerun to unblind and summarize.")
        return

    consensus = read_form(consensus_path, ids)
    keys = {str(row["audit_id"]): row for row in iter_jsonl(key_path)}
    dimension_counts: dict[str, Counter[str]] = {dimension: Counter() for dimension in DIMENSIONS}
    by_outcome: defaultdict[str, Counter[str]] = defaultdict(Counter)
    error_types: Counter[str] = Counter()
    representative: list[dict[str, Any]] = []
    for audit_id in ids:
        row = consensus[audit_id]
        key = keys[audit_id]
        for dimension in DIMENSIONS:
            dimension_counts[dimension][row[dimension]] += 1
        error_types[row["error_type"]] += 1
        by_outcome[str(key["outcome_group"])][row["error_type"]] += 1
        if key["outcome_group"] == "improved" and row["error_type"] == "none" and all(
            row[dimension] == "Yes" for dimension in DIMENSIONS
        ):
            representative.append(
                {
                    "audit_id": audit_id,
                    "benchmark": key["benchmark"],
                    "sample_id": key["sample_id"],
                    "evidence_types": key["evidence_types"],
                    "brief_reason": row["brief_reason"],
                }
            )
    report = {
        "version": "skill-rm-two-author-human-audit-summary-v1",
        "design": {
            "cases": len(ids),
            "annotators": 2,
            "annotator_type": "paper authors",
            "outcome_blinded_during_independent_annotation": True,
            "method_blinded": False,
            "behavior_enriched": True,
            "population_estimate": False,
        },
        "agreement": agreement_summary,
        "consensus_dimension_counts": {key: dict(value) for key, value in dimension_counts.items()},
        "consensus_error_types": dict(error_types),
        "error_types_by_outcome": {key: dict(value) for key, value in by_outcome.items()},
        "representative_improved_candidates": representative,
    }
    write_json(audit_dir / "human_audit_summary.json", report)
    write_json(audit_dir / "representative_case_candidates.json", {"cases": representative})
    lines = [
        "# Two-Author Human Trace Evaluation",
        "",
        "This is a behavior-enriched, outcome-blinded two-author evaluation of 30 traces, not a population estimate.",
        "",
        "| Dimension | Yes | Yes+Partial | Raw agreement | Weighted kappa |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for dimension in DIMENSIONS:
        counts = dimension_counts[dimension]
        item = agreement_summary["dimensions"][dimension]
        lines.append(
            f"| {dimension} | {counts['Yes']}/{len(ids)} | {counts['Yes'] + counts['Partial']}/{len(ids)} | "
            f"{fmt(item['raw_agreement'])} | {fmt(item['ordinal_weighted_kappa'])} |"
        )
    lines.extend(["", "Error types: " + ", ".join(f"{key}={value}" for key, value in sorted(error_types.items())), ""])
    (audit_dir / "human_audit_summary.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
