#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rebuttal_common import BENCHMARKS, iter_jsonl, sha256_bytes, write_json
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


SEED = 0
OUTCOME_PRIORITIES = {"improved": 5, "both_correct": 3, "regressed": 2}
DIMENSIONS = ("action_relevance", "evidence_validity", "evidence_faithfulness", "verdict_support", "auditability")


def occurrence_rows(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    counts: defaultdict[str, int] = defaultdict(int)
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for line_index, row in enumerate(iter_jsonl(path)):
        sample_id = str(row.get("sample_id") or "")
        occurrence = counts[sample_id]
        counts[sample_id] += 1
        item = dict(row)
        item["_line_index"] = line_index
        item["_occurrence_index"] = occurrence
        result[(sample_id, occurrence)] = item
    return result


def occurrence_traces(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    counts: defaultdict[str, int] = defaultdict(int)
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for trace in iter_jsonl(path):
        sample_id = str(trace.get("sample_id") or "")
        occurrence = counts[sample_id]
        counts[sample_id] += 1
        result[(sample_id, occurrence)] = trace
    return result


def evidence_types(trace: dict[str, Any]) -> set[str]:
    found: set[str] = set()
    for step in trace.get("steps") or []:
        calls = step.get("tool_calls") or []
        results = step.get("tool_results") or []
        for index, call in enumerate(calls):
            name = str(call.get("name") or ((call.get("function") or {}).get("name")) or "")
            if name not in {"python_sandbox", "view_resource"}:
                continue
            result = results[index] if index < len(results) and isinstance(results[index], dict) else {}
            if result.get("ok") is True:
                found.add("python" if name == "python_sandbox" else "resource")
    return found


def outcome_group(direct: dict[str, Any], skill: dict[str, Any]) -> str:
    direct_correct = direct.get("correct") is True
    skill_correct = skill.get("correct") is True
    if skill_correct and not direct_correct:
        return "improved"
    if skill_correct and direct_correct:
        return "both_correct"
    if direct_correct and not skill_correct:
        return "regressed"
    return "both_wrong"


def stable_sort_key(case: dict[str, Any]) -> str:
    payload = json.dumps(case["key"], ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_candidates(run_root: Path, benchmark: str) -> list[dict[str, Any]]:
    direct = occurrence_rows(run_root / "direct" / benchmark / "predictions.jsonl")
    skill = occurrence_rows(run_root / "skill_rm" / benchmark / "predictions.jsonl")
    traces = occurrence_traces(run_root / "skill_rm" / benchmark / "traces.jsonl")
    candidates: list[dict[str, Any]] = []
    for key in sorted(set(direct).intersection(skill).intersection(traces)):
        direct_row = direct[key]
        skill_row = skill[key]
        trace = traces[key]
        kinds = evidence_types(trace)
        if not kinds or skill_row.get("valid") is not True or not isinstance(skill_row.get("correct"), bool):
            continue
        group = outcome_group(direct_row, skill_row)
        if group == "both_wrong":
            continue
        candidates.append(
            {
                "key": key,
                "benchmark": benchmark,
                "outcome_group": group,
                "evidence_types": sorted(kinds),
                "direct": direct_row,
                "skill": skill_row,
                "trace": trace,
            }
        )
    return candidates


def select_cases(candidates: list[dict[str, Any]], benchmark: str) -> list[dict[str, Any]]:
    pools: dict[str, list[dict[str, Any]]] = {}
    for group in OUTCOME_PRIORITIES:
        pool = [case for case in candidates if case["outcome_group"] == group]
        pool.sort(key=stable_sort_key)
        random.Random(f"{SEED}:{benchmark}:{group}").shuffle(pool)
        pools[group] = pool

    selected: list[dict[str, Any]] = []
    for group, target in OUTCOME_PRIORITIES.items():
        selected.extend(pools[group][:target])
    if len(selected) < 10:
        remainder = [case for group in OUTCOME_PRIORITIES for case in pools[group] if case not in selected]
        remainder.sort(key=stable_sort_key)
        selected.extend(remainder[: 10 - len(selected)])
    if len(selected) < 10:
        raise ValueError(f"Not enough {benchmark} evidence-bearing cases: {len(selected)} < 10")

    counts: Counter[str] = Counter()
    for case in selected:
        counts.update(case["evidence_types"])

    def replace_until(predicate: Callable[[dict[str, Any]], bool], field: str, target: int) -> None:
        while counts[field] < target:
            choice = next((case for group in OUTCOME_PRIORITIES for case in pools[group] if case not in selected and predicate(case)), None)
            if choice is None:
                raise ValueError(f"Cannot satisfy {benchmark} {field}>={target}")
            removable = next((case for case in reversed(selected) if field not in case["evidence_types"]), None)
            if removable is None:
                raise ValueError(f"Cannot rebalance {benchmark} for {field}")
            selected.remove(removable)
            counts.subtract(removable["evidence_types"])
            selected.append(choice)
            counts.update(choice["evidence_types"])

    replace_until(lambda case: "resource" in case["evidence_types"], "resource", 3)
    replace_until(lambda case: "python" in case["evidence_types"], "python", 5)
    if len(selected) != 10:
        raise AssertionError(f"Invalid selection for {benchmark}")
    return selected


def rb2_messages(config: dict[str, Any], sample_id: str) -> list[dict[str, str]]:
    records = load_official_records(config["data_source"], include_ties=bool(config.get("include_ties", True)))
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
    task = {item.task_id: item for item in load_openrs_tasks(config)}.get(sample_id)
    if task is None:
        raise KeyError(f"OpenRS task not found: {sample_id}")
    skill_package = load_skill_package(config)
    profile = judgebench_system_prompt_profile(task, config) if str(task.benchmark).lower().startswith("judgebench") else None
    user_prompt = format_pairwise_user_prompt(task, with_skill_hint=True)
    if str(task.benchmark).lower().startswith("judgebench") and profile == "none":
        return [{"role": "user", "content": user_prompt}]
    system_prompt = format_self_select_skill_system_prompt(skill_package, config)
    if profile == "minimal":
        system_prompt = format_judgebench_minimal_skill_system_prompt(skill_package, config)
    elif profile == "legacy":
        system_prompt = format_legacy_pairwise_skill_system_prompt(skill_package, config)
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def strip_identity(trace: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in trace.items() if key not in {"sample_id", "trace_id", "original_sample_id"}}


def write_form(path: Path, audit_ids: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audit_id", *DIMENSIONS, "error_type", "brief_reason"])
        writer.writeheader()
        for audit_id in audit_ids:
            writer.writerow({"audit_id": audit_id})


def write_instructions(path: Path, input_hash: str) -> None:
    path.write_text(
        "\n".join(
            [
                "# Two-Author Human Trace Evaluation",
                "",
                "Two paper authors independently annotate all 30 cases before inspecting `sealed_outcome_key.jsonl`.",
                "Use only the visible task, candidate responses, trace, tool/resource observations, and recorded verdict.",
                f"Input SHA256: `{input_hash}`",
                "",
                "Allowed dimension labels: `Yes`, `Partial`, `No`, `N/A`.",
                "",
                "- action_relevance: the invoked skill/resource/tool is relevant to the decision.",
                "- evidence_validity: the produced observation is correct and checkable from visible material.",
                "- evidence_faithfulness: the final reasoning uses the observation without distortion.",
                "- verdict_support: the recorded evidence supports the final verdict.",
                "- auditability: a reader can reconstruct why the verdict was selected.",
                "",
                "Allowed error_type values: `none`, `irrelevant_action`, `invalid_evidence`, `unsupported_claim`, "
                "`tool_failure`, `reasoning_gap`, `insufficient_trace`, `other`.",
                "Do not discuss cases or inspect the other form until both annotation files are complete.",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a blinded two-author trace evaluation from matched compute runs.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--audit-dir")
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    summary = json.loads((run_root / "compute_summary.json").read_text(encoding="utf-8"))
    if not summary.get("all_runs_complete"):
        raise ValueError("All six matched runs with complete usage are required.")
    audit_dir = Path(args.audit_dir).resolve() if args.audit_dir else run_root / "human_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    staged: list[dict[str, Any]] = []
    for benchmark in BENCHMARKS:
        staged.extend(select_cases(build_candidates(run_root, benchmark), benchmark))
    random.Random(SEED).shuffle(staged)

    inputs: list[dict[str, Any]] = []
    keys: list[dict[str, Any]] = []
    for index, case in enumerate(staged, start=1):
        audit_id = f"HUMAN-{index:03d}"
        sample_id, occurrence = case["key"]
        config = load_config(str(REPO_ROOT / "configs" / case["benchmark"] / "skill_fair.yaml"))
        messages = rb2_messages(config, sample_id) if case["benchmark"] == "rewardbench2" else openrs_messages(config, sample_id)
        inputs.append(
            {
                "audit_id": audit_id,
                "benchmark": case["benchmark"],
                "visible_messages": messages,
                "trace": strip_identity(case["trace"]),
                "recorded_final": {
                    "predicted_label": case["skill"].get("predicted_label"),
                    "skill_final_verdict": case["skill"].get("skill_final_verdict"),
                },
            }
        )
        keys.append(
            {
                "audit_id": audit_id,
                "benchmark": case["benchmark"],
                "sample_id": sample_id,
                "occurrence_index": occurrence,
                "outcome_group": case["outcome_group"],
                "evidence_types": case["evidence_types"],
                "direct_correct": case["direct"].get("correct"),
                "skill_correct": case["skill"].get("correct"),
                "direct_label": case["direct"].get("predicted_label"),
                "skill_label": case["skill"].get("predicted_label"),
            }
        )

    input_payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in inputs).encode("utf-8")
    key_payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in keys).encode("utf-8")
    input_hash = sha256_bytes(input_payload)
    (audit_dir / "audit_input.jsonl").write_bytes(input_payload)
    (audit_dir / "sealed_outcome_key.jsonl").write_bytes(key_payload)
    write_form(audit_dir / "annotator_a.csv", [row["audit_id"] for row in inputs])
    write_form(audit_dir / "annotator_b.csv", [row["audit_id"] for row in inputs])
    write_instructions(audit_dir / "instructions.md", input_hash)
    write_json(
        audit_dir / "manifest.json",
        {
            "version": "skill-rm-two-author-human-audit-v1",
            "seed": SEED,
            "cases": 30,
            "per_benchmark": 10,
            "outcome_priorities_per_benchmark": OUTCOME_PRIORITIES,
            "minimum_python_per_benchmark": 5,
            "minimum_resource_views_per_benchmark": 3,
            "input_sha256": input_hash,
            "key_sha256": sha256_bytes(key_payload),
            "blinding": "annotators see method traces but not identity, gold, baseline verdict, correctness, or outcome group",
        },
    )
    print(f"human_audit_cases={len(inputs)}")
    print(f"audit_input_sha256={input_hash}")
    print("Complete annotator_a.csv and annotator_b.csv independently before opening sealed_outcome_key.jsonl.")


if __name__ == "__main__":
    main()
