#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rebuttal_common import BENCHMARKS, iter_jsonl, sha256_bytes, write_json, write_jsonl
from skillrm.openrs_benchmark import (
    format_judgebench_minimal_skill_system_prompt,
    format_legacy_pairwise_skill_system_prompt,
    format_pairwise_user_prompt,
    judgebench_system_prompt_profile,
    load_openrs_tasks,
)
from skillrm.qwen_baseline import (
    format_self_select_skill_system_prompt,
    format_self_select_skill_user_prompt,
    load_config,
    load_official_records,
    load_skill_package,
    official_format_ranking_record,
)


AUDIT_SEED = 0
PER_OUTCOME = 5


def stable_row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    payload = json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return str(row.get("sample_id") or ""), str(row.get("predicted_label") or ""), hashlib.sha256(payload).hexdigest()


def load_prediction_candidates(path: Path) -> list[dict[str, Any]]:
    occurrences: defaultdict[str, int] = defaultdict(int)
    rows: list[dict[str, Any]] = []
    for line_index, row in enumerate(iter_jsonl(path)):
        sample_id = str(row.get("sample_id") or "")
        occurrence = occurrences[sample_id]
        occurrences[sample_id] += 1
        if row.get("valid") is not True or not row.get("trace_id") or not isinstance(row.get("correct"), bool):
            continue
        item = dict(row)
        item["_prediction_line_index"] = line_index
        item["_occurrence_index"] = occurrence
        rows.append(item)
    return rows


def select_rows(path: Path, benchmark: str) -> list[dict[str, Any]]:
    candidates = load_prediction_candidates(path)
    selected: list[dict[str, Any]] = []
    for correct in (True, False):
        outcome_rows = [row for row in candidates if row["correct"] is correct]
        unique: dict[str, dict[str, Any]] = {}
        for row in sorted(outcome_rows, key=stable_row_key):
            unique.setdefault(str(row["sample_id"]), row)
        pool = list(unique.values())
        random.Random(f"{AUDIT_SEED}:{benchmark}:{correct}").shuffle(pool)
        if len(pool) < PER_OUTCOME:
            raise ValueError(f"Not enough {'correct' if correct else 'incorrect'} full-fair cases for {benchmark}")
        selected.extend(pool[:PER_OUTCOME])
    return selected


def trace_by_sample(path: Path) -> dict[str, dict[str, Any]]:
    traces: dict[str, dict[str, Any]] = {}
    for trace in iter_jsonl(path):
        traces.setdefault(str(trace.get("sample_id") or ""), trace)
    return traces


def strip_trace_identity(trace: dict[str, Any]) -> dict[str, Any]:
    cleaned = {key: value for key, value in trace.items() if key not in {"sample_id", "trace_id", "original_sample_id"}}
    return cleaned


def rb2_messages(config: dict[str, Any], sample_id: str) -> list[dict[str, str]]:
    records = load_official_records(
        config["data_source"],
        limit=config.get("limit"),
        include_ties=bool(config.get("include_ties", True)),
    )
    record = next((item for item in records if str(item.get("id")) == sample_id), None)
    if record is None:
        raise KeyError(f"RewardBench2 sample not found: {sample_id}")
    skill_package = load_skill_package(config)
    formatted = official_format_ranking_record(record, seed=int(config.get("seed", 0)))
    return [
        {"role": "system", "content": format_self_select_skill_system_prompt(skill_package, config)},
        {"role": "user", "content": format_self_select_skill_user_prompt(record, formatted)},
    ]


def openrs_messages(config: dict[str, Any], sample_id: str) -> list[dict[str, str]]:
    tasks = {task.task_id: task for task in load_openrs_tasks(config)}
    task = tasks.get(sample_id)
    if task is None:
        raise KeyError(f"OpenRS task not found: {sample_id}")
    skill_package = load_skill_package(config)
    profile = judgebench_system_prompt_profile(task, config) if str(task.benchmark).lower().startswith("judgebench") else None
    use_system = not str(task.benchmark).lower().startswith("judgebench") or profile != "none"
    user_prompt = format_pairwise_user_prompt(task, with_skill_hint=True)
    if not use_system:
        return [{"role": "user", "content": user_prompt}]
    system_prompt = format_self_select_skill_system_prompt(skill_package, config)
    if str(task.benchmark).lower().startswith("judgebench"):
        if profile == "minimal":
            system_prompt = format_judgebench_minimal_skill_system_prompt(skill_package, config)
        elif profile == "legacy":
            system_prompt = format_legacy_pairwise_skill_system_prompt(skill_package, config)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def config_path(repo_root: Path, benchmark: str) -> Path:
    return repo_root / "configs" / "rebuttal_original_v6" / f"{benchmark}_full_fair.yaml"


def build_cases(repo_root: Path, run_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    staged: list[dict[str, Any]] = []
    for benchmark in BENCHMARKS:
        run_dir = run_root / "full_fair" / benchmark
        selected = select_rows(run_dir / "predictions.jsonl", benchmark)
        traces = trace_by_sample(run_dir / "traces.jsonl")
        config = load_config(str(config_path(repo_root, benchmark)))
        for row in selected:
            sample_id = str(row["sample_id"])
            trace = traces.get(sample_id)
            if trace is None:
                raise KeyError(f"Trace missing for selected case: {benchmark}/{sample_id}")
            messages = rb2_messages(config, sample_id) if benchmark == "rewardbench2" else openrs_messages(config, sample_id)
            staged.append(
                {
                    "benchmark": benchmark,
                    "sample_id": sample_id,
                    "occurrence_index": int(row["_occurrence_index"]),
                    "prediction_line_index": int(row["_prediction_line_index"]),
                    "correct": bool(row["correct"]),
                    "official_score": row.get("official_score"),
                    "gold_label": row.get("gold_label") or row.get("chosen_label"),
                    "recorded_final": {
                        "predicted_label": row.get("predicted_label"),
                        "skill_final_verdict": row.get("skill_final_verdict"),
                        "verdict_source": row.get("verdict_source"),
                    },
                    "visible_messages": messages,
                    "trace": strip_trace_identity(trace),
                }
            )
    random.Random(AUDIT_SEED).shuffle(staged)
    inputs: list[dict[str, Any]] = []
    keys: list[dict[str, Any]] = []
    for index, case in enumerate(staged, start=1):
        audit_id = f"AUDIT-{index:03d}"
        inputs.append(
            {
                "audit_id": audit_id,
                "benchmark": case["benchmark"],
                "visible_messages": case["visible_messages"],
                "trace": case["trace"],
                "recorded_final": case["recorded_final"],
            }
        )
        keys.append(
            {
                "audit_id": audit_id,
                "benchmark": case["benchmark"],
                "sample_id": case["sample_id"],
                "occurrence_index": case["occurrence_index"],
                "prediction_line_index": case["prediction_line_index"],
                "correct": case["correct"],
                "official_score": case["official_score"],
                "gold_label": case["gold_label"],
            }
        )
    return inputs, keys


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")


def write_instructions(path: Path, input_sha256: str) -> None:
    path.write_text(
        "\n".join(
            [
                "# Blinded Model-Based Trace Audit",
                "",
                "Audit every row in `audit_input.jsonl` without opening `audit_key.jsonl`.",
                "Assess whether the recorded trace supports its own final verdict. Do not infer or guess the gold answer.",
                "This is a model-based audit, not human evaluation.",
                "",
                f"Input SHA256: `{input_sha256}`",
                "",
                "Write exactly one JSON object per input row to `audit_results.jsonl` with:",
                "",
                "```json",
                '{"audit_id":"AUDIT-001","resource_relevance":1,"evidence_faithfulness":1,"verdict_supported":1,"auditability":1,"error_type":"none","brief_reason":"..."}',
                "```",
                "",
                "All four scores are integers from 1 (poor) to 5 (strong).",
                "`error_type` must be one of: `none`, `irrelevant_resource`, `unsupported_evidence`, `tool_misuse`, "
                "`reasoning_gap`, `final_verdict_mismatch`, `insufficient_trace`, `other`.",
                "Keep `brief_reason` concise and grounded only in the visible messages and trace.",
                "Do not include sample IDs, gold labels, correctness guesses, prompts, or candidate text in the result row.",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a blinded 30-case audit from new full-fair outputs.")
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    summary = json.loads((run_root / "rebuttal_summary.json").read_text(encoding="utf-8"))
    if not summary.get("all_runs_complete"):
        raise ValueError("All nine runs must validate before audit sampling.")

    inputs, keys = build_cases(repo_root, run_root)
    if len(inputs) != 30 or len(keys) != 30:
        raise AssertionError("Expected exactly 30 audit cases.")
    counts = Counter((item["benchmark"], item["correct"]) for item in keys)
    expected = {(benchmark, outcome): 5 for benchmark in BENCHMARKS for outcome in (True, False)}
    if dict(counts) != expected:
        raise AssertionError(f"Outcome stratification failed: {counts}")

    audit_dir = run_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    input_payload = jsonl_bytes(inputs)
    key_payload = jsonl_bytes(keys)
    input_hash = sha256_bytes(input_payload)
    key_hash = sha256_bytes(key_payload)
    manifest_path = audit_dir / "audit_manifest.json"
    if manifest_path.is_file() and (audit_dir / "audit_results.jsonl").is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("input_sha256") != input_hash:
            stale_dir = audit_dir / "stale"
            stale_dir.mkdir(exist_ok=True)
            suffix = str(previous.get("input_sha256") or "unknown")[:12]
            shutil.move(str(audit_dir / "audit_results.jsonl"), str(stale_dir / f"audit_results_{suffix}.jsonl"))

    (audit_dir / "audit_input.jsonl").write_bytes(input_payload)
    (audit_dir / "audit_key.jsonl").write_bytes(key_payload)
    write_instructions(audit_dir / "audit_instructions.md", input_hash)
    write_json(
        manifest_path,
        {
            "version": "skill-rm-original-v6-model-audit-v1",
            "seed": AUDIT_SEED,
            "cases": 30,
            "per_benchmark": 10,
            "per_benchmark_outcome": 5,
            "input_sha256": input_hash,
            "key_sha256": key_hash,
            "sampling": "outcome-stratified from new full_fair valid traced predictions",
            "blinding": "auditor sees method and benchmark but not sample identity, gold, or outcome",
        },
    )
    print(f"audit_cases={len(inputs)}")
    print(f"audit_input_sha256={input_hash}")
    print("Do not inspect audit_key.jsonl before all 30 audit_results.jsonl rows are complete.")


if __name__ == "__main__":
    main()
